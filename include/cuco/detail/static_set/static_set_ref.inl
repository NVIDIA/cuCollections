/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuco/detail/equal_wrapper.cuh>
#include <cuco/detail/pair.cuh>
#include <cuco/operator.hpp>
#include <cuco/sentinel.cuh>

#include <thrust/distance.h>

#include <cooperative_groups.h>
#include <cuda/std/atomic>
#include <type_traits>

namespace cuco {
namespace experimental {

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_set_ref<
  Key,
  Scope,
  KeyEqual,
  ProbingScheme,
  StorageRef,
  Operators...>::static_set_ref(cuco::empty_key<Key> empty_key_sentinel,
                                KeyEqual const& predicate,
                                ProbingScheme const& probing_scheme,
                                StorageRef storage_ref) noexcept
  : empty_key_sentinel_{empty_key_sentinel},
    predicate_{empty_key_sentinel, predicate},
    probing_scheme_{probing_scheme},
    storage_ref_{storage_ref}
{
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr auto
static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::capacity()
  const noexcept
{
  return storage_ref_.capacity();
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr Key
static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::empty_key_sentinel()
  const noexcept
{
  return empty_key_sentinel_;
}

namespace detail {

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<op::insert_tag,
                    static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type  = static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type   = static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type   = typename base_type::key_type;
  using value_type = typename base_type::value_type;

  static constexpr auto cg_size      = base_type::cg_size;
  static constexpr auto window_size  = base_type::window_size;
  static constexpr auto thread_scope = base_type::thread_scope;

 public:
  /**
   * @brief Inserts an element.
   *
   * @param value The element to insert
   * @return True if the given element is successfully inserted
   */
  __device__ bool insert(value_type const& value) noexcept
  {
    ref_type& ref_    = static_cast<ref_type&>(*this);
    auto probing_iter = ref_.probing_scheme_(value, ref_.storage_ref_.num_windows());

    while (true) {
      auto const window_slots = ref_.storage_ref_[*probing_iter];

      // TODO: perf gain with #pragma unroll since num_windows is build time constant
      for (auto& slot_content : window_slots) {
        auto const eq_res = ref_.predicate_(slot_content, value);

        // If the key is already in the container, return false
        if (eq_res == detail::equal_result::EQUAL) { return false; }
        if (eq_res == detail::equal_result::EMPTY) {
          auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
          switch (attempt_insert(
            (ref_.storage_ref_.data() + *probing_iter)->data() + intra_window_index, value)) {
            case insert_result::CONTINUE: continue;
            case insert_result::SUCCESS: return true;
            case insert_result::DUPLICATE: return false;
          }
        }
      }
      ++probing_iter;
    }
  }

  /**
   * @brief Inserts an element.
   *
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   * @return True if the given element is successfully inserted
   */
  __device__ bool insert(cooperative_groups::thread_block_tile<cg_size> group,
                         value_type const& value) noexcept
  {
    auto& ref_        = static_cast<ref_type&>(*this);
    auto probing_iter = ref_.probing_scheme_(group, value, ref_.storage_ref_.num_windows());

    while (true) {
      auto const window_slots = ref_.storage_ref_[*probing_iter];

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (ref_.predicate_(window_slots[i], value)) {
            case detail::equal_result::EMPTY: return cuco::pair{detail::equal_result::EMPTY, i};
            case detail::equal_result::EQUAL: return cuco::pair{detail::equal_result::EQUAL, i};
            default: continue;
          }
        }
        return cuco::pair<detail::equal_result, int32_t>{detail::equal_result::UNEQUAL, -1};
      }();

      // If the key is already in the container, return false
      if (group.any(state == detail::equal_result::EQUAL)) { return false; }

      auto const group_contains_empty = group.ballot(state == detail::equal_result::EMPTY);

      if (group_contains_empty) {
        auto const src_lane = __ffs(group_contains_empty) - 1;
        auto const status =
          (group.thread_rank() == src_lane)
            ? attempt_insert(
                (ref_.storage_ref_.data() + *probing_iter)->data() + intra_window_index, value)
            : insert_result::CONTINUE;

        switch (group.shfl(status, src_lane)) {
          case insert_result::SUCCESS: return true;
          case insert_result::DUPLICATE: return false;
          default: continue;
        }
      } else {
        ++probing_iter;
      }
    }
  }

 private:
  enum class insert_result : int32_t { CONTINUE = 0, SUCCESS = 1, DUPLICATE = 2 };

  /**
   * @brief Attempts to insert an element into a slot.
   *
   * @note Dispatches the correct implementation depending on the container
   * type and presence of other operator mixins.
   *
   * @param slot Pointer to the slot in memory
   * @param value Element to insert
   *
   * @return Result of this operation, i.e., success/continue/duplicate
   */
  __device__ insert_result attempt_insert(value_type* slot, value_type const& value)
  {
    return cas(slot, value);
  }

  /**
   * @brief Try insert using simple atomic compare-and-swap.
   *
   * @param slot Pointer to the slot in memory
   * @param value Element to insert
   *
   * @return Result of this operation, i.e., success/continue/duplicate
   */
  __device__ insert_result cas(value_type* slot, value_type const& value)
  {
    auto& ref_ = static_cast<ref_type&>(*this);

    // temporary workaround due to performance regression
    // https://github.com/NVIDIA/libcudacxx/issues/366
    value_type const old = [&]() {
      value_type expected = ref_.empty_key_sentinel_.value;
      value_type val      = value;
      if constexpr (sizeof(value_type) == sizeof(uint32_t)) {
        auto* expected_ptr = reinterpret_cast<unsigned int*>(&expected);
        auto* value_ptr    = reinterpret_cast<unsigned int*>(&val);
        if constexpr (thread_scope == cuda::thread_scope_system) {
          return atomicCAS_system(reinterpret_cast<unsigned int*>(slot), *expected_ptr, *value_ptr);
        } else if constexpr (thread_scope == cuda::thread_scope_device) {
          return atomicCAS(reinterpret_cast<unsigned int*>(slot), *expected_ptr, *value_ptr);
        } else if constexpr (thread_scope == cuda::thread_scope_block) {
          return atomicCAS_block(reinterpret_cast<unsigned int*>(slot), *expected_ptr, *value_ptr);
        } else {
          static_assert(cuco::dependent_false<decltype(thread_scope)>, "Unsupported thread scope");
        }
      }
      if constexpr (sizeof(value_type) == sizeof(uint64_t)) {
        auto* expected_ptr = reinterpret_cast<unsigned long long int*>(&expected);
        auto* value_ptr    = reinterpret_cast<unsigned long long int*>(&val);
        if constexpr (thread_scope == cuda::thread_scope_system) {
          return atomicCAS_system(
            reinterpret_cast<unsigned long long int*>(slot), *expected_ptr, *value_ptr);
        }
        if constexpr (thread_scope == cuda::thread_scope_device) {
          return atomicCAS(
            reinterpret_cast<unsigned long long int*>(slot), *expected_ptr, *value_ptr);
        }
        if constexpr (thread_scope == cuda::thread_scope_block) {
          return atomicCAS_block(
            reinterpret_cast<unsigned long long int*>(slot), *expected_ptr, *value_ptr);
        }
      }
    }();
    if (*slot == old) {
      // Shouldn't use `predicate_` operator directly since it includes a redundant bitwise compare
      return ref_.predicate_.equal_to(old, value) == detail::equal_result::EQUAL
               ? insert_result::DUPLICATE
               : insert_result::CONTINUE;
    } else {
      return insert_result::SUCCESS;
    }
  }
};

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<op::contains_tag,
                    static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type  = static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type   = static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type   = typename base_type::key_type;
  using value_type = typename base_type::value_type;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Indicates whether the probe key `key` was inserted into the container.
   *
   * If the probe key `key` was inserted into the container, returns
   * true. Otherwise, returns false.
   *
   * @tparam ProbeKey Probe key type
   *
   * @param key The key to search for
   * @return A boolean indicating whether the probe key is present
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ bool contains(ProbeKey const& key) const noexcept
  {
    // CRTP: cast `this` to the actual reference type
    auto const& ref_ = static_cast<ref_type const&>(*this);

    auto probing_iter = ref_.probing_scheme_(key, ref_.storage_ref_.num_windows());

    while (true) {
      // TODO atomic_ref::load if insert operator is present
      auto const window_slots = ref_.storage_ref_[*probing_iter];

      for (auto& slot_content : window_slots) {
        switch (ref_.predicate_(slot_content, key)) {
          case detail::equal_result::UNEQUAL: continue;
          case detail::equal_result::EMPTY: return false;
          case detail::equal_result::EQUAL: return true;
        }
      }
      ++probing_iter;
    }
  }

  /**
   * @brief Indicates whether the probe key `key` was inserted into the container.
   *
   * If the probe key `key` was inserted into the container, returns
   * true. Otherwise, returns false.
   *
   * @tparam ProbeKey Probe key type
   *
   * @param g The Cooperative Group used to perform group contains
   * @param key The key to search for
   * @return A boolean indicating whether the probe key is present
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ bool contains(cooperative_groups::thread_block_tile<cg_size> const& g,
                                         ProbeKey const& key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);

    auto probing_iter = ref_.probing_scheme_(g, key, ref_.storage_ref_.num_windows());

    while (true) {
      auto const window_slots = ref_.storage_ref_[*probing_iter];

      auto const state = [&]() {
        for (auto& slot : window_slots) {
          switch (ref_.predicate_(slot, key)) {
            case detail::equal_result::EMPTY: return detail::equal_result::EMPTY;
            case detail::equal_result::EQUAL: return detail::equal_result::EQUAL;
            default: continue;
          }
        }
        return detail::equal_result::UNEQUAL;
      }();

      if (g.any(state == detail::equal_result::EQUAL)) { return true; }
      if (g.any(state == detail::equal_result::EMPTY)) { return false; }

      ++probing_iter;
    }
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
