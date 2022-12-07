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
#include <cuco/sentinel.cuh>  // TODO .hpp

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
template <typename... NewOperators>
auto static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::with(
  NewOperators...) const noexcept
{
  return static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, NewOperators...>(
    this->empty_key_sentinel_, this->predicate_.equal_, this->probing_scheme_, this->storage_ref_);
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

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Inserts an element.
   *
   * @param value The element to insert
   * @return True if the given element is successfully inserted
   */
  __device__ inline bool insert(value_type const& value) noexcept
  {
    ref_type& ref_    = static_cast<ref_type&>(*this);
    auto probing_iter = ref_.probing_scheme_(value, ref_.storage_ref_.num_windows());

    while (true) {
      auto const window_slots = ref_.storage_ref_.window(*probing_iter);

      for (auto& slot_content : window_slots) {
        auto const eq_res = ref_.predicate_(slot_content, value);

        // If the key is already in the map, return false
        // TODO this needs to be disabled for static_multimap
        if (eq_res == detail::equal_result::EQUAL) { return false; }
        if (eq_res == detail::equal_result::EMPTY) {
          auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
          switch (attempt_insert(
            (ref_.storage_ref_.windows() + *probing_iter)->data() + intra_window_index, value)) {
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
   * @param g The Cooperative Group used to perform group insert
   * @param value The element to insert
   * @return True if the given element is successfully inserted
   */
  __device__ inline bool insert(cooperative_groups::thread_block_tile<cg_size> group,
                                value_type const& value) noexcept
  {
    auto& ref_        = static_cast<ref_type&>(*this);
    auto probing_iter = ref_.probing_scheme_(group, value, ref_.storage_ref_.num_windows());

    while (true) {
      auto const window_slots = ref_.storage_ref_.window(*probing_iter);

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (ref_.predicate_(window_slots[i], value)) {
            case detail::equal_result::EMPTY:
              return cuco::pair<detail::equal_result, int32_t>{detail::equal_result::EMPTY, i};
            case detail::equal_result::EQUAL:
              return cuco::pair<detail::equal_result, int32_t>{detail::equal_result::EQUAL, i};
            default: continue;
          }
        }
        return cuco::pair<detail::equal_result, int32_t>{detail::equal_result::UNEQUAL, -1};
      }();

      // If the key is already in the map, return false
      if (group.any(state == detail::equal_result::EQUAL)) { return false; }

      auto const group_contains_empty = group.ballot(state == detail::equal_result::EMPTY);

      if (group_contains_empty) {
        auto const src_lane = __ffs(group_contains_empty) - 1;
        auto const status =
          (group.thread_rank() == src_lane)
            ? attempt_insert(
                (ref_.storage_ref_.windows() + *probing_iter)->data() + intra_window_index, value)
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
  __device__ inline insert_result attempt_insert(value_type* slot, value_type const& value)
  {
    // code path for static_set -> cas
    if constexpr (std::is_same_v<key_type, value_type>) { return cas(slot, value); }

    // TODO code path for static_map and static_multimap
  }

  /**
   * @brief Try insert using simple atomic compare-and-swap.
   *
   * @param slot Pointer to the slot in memory
   * @param value Element to insert
   *
   * @return Result of this operation, i.e., success/continue/duplicate
   */
  __device__ inline insert_result cas(value_type* slot, value_type const& value)
  {
    auto& ref_ = static_cast<ref_type&>(*this);

    auto ref      = cuda::atomic_ref<value_type, Scope>{*slot};
    auto expected = ref_.empty_key_sentinel_.value;
    bool result   = ref.compare_exchange_strong(expected, value, cuda::std::memory_order_relaxed);
    if (result) {
      return insert_result::SUCCESS;
    } else {
      auto old = expected;
      return ref_.predicate_(old, value) == detail::equal_result::EQUAL ? insert_result::DUPLICATE
                                                                        : insert_result::CONTINUE;
    }
  }

  // TODO packed_cas, back_to_back_cas, cas_dependent_write
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
  [[nodiscard]] __device__ inline bool contains(ProbeKey const& key) const noexcept
  {
    // CRTP: cast `this` to the actual reference type
    auto const& ref_ = static_cast<ref_type const&>(*this);

    auto probing_iter = ref_.probing_scheme_(key, ref_.storage_ref_.num_windows());

    while (true) {
      // TODO do we need to use atomic_ref::load here?
      auto const window_slots = ref_.storage_ref_.window(*probing_iter);

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
  [[nodiscard]] __device__ inline bool contains(
    cooperative_groups::thread_block_tile<cg_size> const& g, ProbeKey const& key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);

    auto probing_iter = ref_.probing_scheme_(g, key, ref_.storage_ref_.num_windows());

    while (true) {
      auto const window_slots = ref_.storage_ref_.window(*probing_iter);

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
