/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <thrust/pair.h>

#include <cuda/std/atomic>

#include <cooperative_groups.h>

#include <cstdint>
#include <type_traits>

namespace cuco {
namespace experimental {

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_map_ref<
  Key,
  T,
  Scope,
  KeyEqual,
  ProbingScheme,
  StorageRef,
  Operators...>::static_map_ref(cuco::empty_key<Key> empty_key_sentinel,
                                cuco::empty_value<T> empty_value_sentinel,
                                KeyEqual const& predicate,
                                ProbingScheme const& probing_scheme,
                                StorageRef storage_ref) noexcept
  : empty_key_sentinel_{empty_key_sentinel},
    empty_value_sentinel_{empty_value_sentinel},
    predicate_{empty_key_sentinel, predicate},
    probing_scheme_{probing_scheme},
    storage_ref_{storage_ref}
{
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr auto
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::capacity()
  const noexcept
{
  return storage_ref_.capacity();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr Key
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
  empty_key_sentinel() const noexcept
{
  return empty_key_sentinel_;
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__device__
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::insert_result
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::attempt_insert(
    value_type* slot, value_type const& value)
{
  // temporary workaround due to performance regression
  // https://github.com/NVIDIA/libcudacxx/issues/366
  value_type const old = [&]() {
    value_type expected = this->empty_key_sentinel();
    value_type val      = value;
    if constexpr (sizeof(value_type) == sizeof(unsigned int)) {
      auto* expected_ptr = reinterpret_cast<unsigned int*>(&expected);
      auto* value_ptr    = reinterpret_cast<unsigned int*>(&val);
      if constexpr (Scope == cuda::thread_scope_system) {
        return atomicCAS_system(reinterpret_cast<unsigned int*>(slot), *expected_ptr, *value_ptr);
      } else if constexpr (Scope == cuda::thread_scope_device) {
        return atomicCAS(reinterpret_cast<unsigned int*>(slot), *expected_ptr, *value_ptr);
      } else if constexpr (Scope == cuda::thread_scope_block) {
        return atomicCAS_block(reinterpret_cast<unsigned int*>(slot), *expected_ptr, *value_ptr);
      } else {
        static_assert(cuco::dependent_false<decltype(Scope)>, "Unsupported thread scope");
      }
    }
    if constexpr (sizeof(value_type) == sizeof(unsigned long long int)) {
      auto* expected_ptr = reinterpret_cast<unsigned long long int*>(&expected);
      auto* value_ptr    = reinterpret_cast<unsigned long long int*>(&val);
      if constexpr (Scope == cuda::thread_scope_system) {
        return atomicCAS_system(
          reinterpret_cast<unsigned long long int*>(slot), *expected_ptr, *value_ptr);
      } else if constexpr (Scope == cuda::thread_scope_device) {
        return atomicCAS(
          reinterpret_cast<unsigned long long int*>(slot), *expected_ptr, *value_ptr);
      } else if constexpr (Scope == cuda::thread_scope_block) {
        return atomicCAS_block(
          reinterpret_cast<unsigned long long int*>(slot), *expected_ptr, *value_ptr);
      } else {
        static_assert(cuco::dependent_false<decltype(Scope)>, "Unsupported thread scope");
      }
    }
  }();
  if (*slot == old) {
    // Shouldn't use `predicate_` operator directly since it includes a redundant bitwise compare
    return predicate_.equal_to(old, value) == detail::equal_result::EQUAL ? insert_result::DUPLICATE
                                                                          : insert_result::CONTINUE;
  } else {
    return insert_result::SUCCESS;
  }
}

namespace detail {

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<
  op::insert_tag,
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type = typename base_type::key_type;
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
  __device__ bool insert(value_type const& value) noexcept
  {
    using insert_result = typename ref_type::insert_result;

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
          switch (ref_.attempt_insert(
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
    using insert_result = typename ref_type::insert_result;

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
        // returns dummy index `-1` for UNEQUAL
        return cuco::pair<detail::equal_result, int32_t>{detail::equal_result::UNEQUAL, -1};
      }();

      // If the key is already in the container, return false
      if (group.any(state == detail::equal_result::EQUAL)) { return false; }

      auto const group_contains_empty = group.ballot(state == detail::equal_result::EMPTY);

      if (group_contains_empty) {
        auto const src_lane = __ffs(group_contains_empty) - 1;
        auto const status =
          (group.thread_rank() == src_lane)
            ? ref_.attempt_insert(
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
};

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<
  op::insert_and_find_tag,
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type = typename base_type::key_type;
  using value_type     = typename base_type::value_type;
  using iterator       = typename base_type::iterator;
  using const_iterator = typename base_type::const_iterator;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Inserts the given element into the set.
   *
   * @note This API returns a pair consisting of an iterator to the inserted element (or to the
   * element that prevented the insertion) and a `bool` denoting whether the insertion took place or
   * not.
   *
   * @param value The element to insert
   *
   * @return a pair consisting of an iterator to the element and a bool indicating whether the
   * insertion is successful or not.
   */
  __device__ thrust::pair<iterator, bool> insert_and_find(value_type const& value) noexcept
  {
    using insert_result = typename ref_type::insert_result;

    ref_type& ref_    = static_cast<ref_type&>(*this);
    auto probing_iter = ref_.probing_scheme_(value, ref_.storage_ref_.num_windows());

    while (true) {
      auto const window_slots = ref_.storage_ref_[*probing_iter];

      for (auto i = 0; i < window_size; ++i) {
        auto const eq_res = ref_.predicate_(window_slots[i], value);
        auto* window_ptr  = (ref_.storage_ref_.data() + *probing_iter)->data();

        // If the key is already in the container, return false
        if (eq_res == detail::equal_result::EQUAL) { return {iterator{&window_ptr[i]}, false}; }
        if (eq_res == detail::equal_result::EMPTY) {
          switch (ref_.attempt_insert(window_ptr + i, value)) {
            case insert_result::SUCCESS: {
              return {iterator{&window_ptr[i]}, true};
            }
            case insert_result::DUPLICATE: {
              return {iterator{&window_ptr[i]}, false};
            }
            default: continue;
          }
        }
      }
      ++probing_iter;
    };
  }

  /**
   * @brief Inserts the given element into the set.
   *
   * @note This API returns a pair consisting of an iterator to the inserted element (or to the
   * element that prevented the insertion) and a `bool` denoting whether the insertion took place or
   * not.
   *
   * @param group The Cooperative Group used to perform group insert_and_find
   * @param value The element to insert
   *
   * @return a pair consisting of an iterator to the element and a bool indicating whether the
   * insertion is successful or not.
   */
  __device__ thrust::pair<iterator, bool> insert_and_find(
    cooperative_groups::thread_block_tile<cg_size> const& group, value_type const& value) noexcept
  {
    using insert_result = typename ref_type::insert_result;

    ref_type& ref_    = static_cast<ref_type&>(*this);
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
        // returns dummy index `-1` for UNEQUAL
        return cuco::pair<detail::equal_result, int32_t>{detail::equal_result::UNEQUAL, -1};
      }();

      auto* slot_ptr = (ref_.storage_ref_.data() + *probing_iter)->data() + intra_window_index;

      // If the key is already in the container, return false
      auto const group_finds_equal = group.ballot(state == detail::equal_result::EQUAL);
      if (group_finds_equal) {
        auto const src_lane = __ffs(group_finds_equal) - 1;
        auto const res      = group.shfl(reinterpret_cast<intptr_t>(slot_ptr), src_lane);
        return {iterator{reinterpret_cast<value_type*>(res)}, false};
      }

      auto const group_contains_empty = group.ballot(state == detail::equal_result::EMPTY);
      if (group_contains_empty) {
        auto const src_lane = __ffs(group_contains_empty) - 1;
        auto const res      = group.shfl(reinterpret_cast<intptr_t>(slot_ptr), src_lane);
        auto const status = (group.thread_rank() == src_lane) ? ref_.attempt_insert(slot_ptr, value)
                                                              : insert_result::CONTINUE;

        switch (group.shfl(status, src_lane)) {
          case insert_result::SUCCESS: {
            return {iterator{reinterpret_cast<value_type*>(res)}, true};
          }
          case insert_result::DUPLICATE: {
            return {iterator{reinterpret_cast<value_type*>(res)}, false};
          }
          default: continue;
        }
      } else {
        ++probing_iter;
      }
    }
  }
};

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<
  op::contains_tag,
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type = typename base_type::key_type;
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
    // CRTP: cast `this` to the actual ref type
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
   * @param group The Cooperative Group used to perform group contains
   * @param key The key to search for
   * @return A boolean indicating whether the probe key is present
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ bool contains(
    cooperative_groups::thread_block_tile<cg_size> const& group, ProbeKey const& key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);

    auto probing_iter = ref_.probing_scheme_(group, key, ref_.storage_ref_.num_windows());

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

      if (group.any(state == detail::equal_result::EQUAL)) { return true; }
      if (group.any(state == detail::equal_result::EMPTY)) { return false; }

      ++probing_iter;
    }
  }
};

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<
  op::find_tag,
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type = typename base_type::key_type;
  using value_type     = typename base_type::value_type;
  using iterator       = typename base_type::iterator;
  using const_iterator = typename base_type::const_iterator;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Returns a const_iterator to one past the last slot.
   *
   * @note This API is available only when `find_tag` is present.
   *
   * @return A const_iterator to one past the last slot
   */
  [[nodiscard]] __host__ __device__ constexpr const_iterator end() const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return ref_.storage_ref_.end();
  }

  /**
   * @brief Returns an iterator to one past the last slot.
   *
   * @note This API is available only when `find_tag` is present.
   *
   * @return An iterator to one past the last slot
   */
  [[nodiscard]] __host__ __device__ constexpr iterator end() noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return ref_.storage_ref_.end();
  }

  /**
   * @brief Finds an element in the set with key equivalent to the probe key.
   *
   * @note Returns a un-incrementable input iterator to the element whose key is equivalent to
   * `key`. If no such element exists, returns `end()`.
   *
   * @tparam ProbeKey Probe key type
   *
   * @param key The key to search for
   *
   * @return An iterator to the position at which the equivalent key is stored
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ const_iterator find(ProbeKey const& key) const noexcept
  {
    // CRTP: cast `this` to the actual ref type
    auto const& ref_ = static_cast<ref_type const&>(*this);

    auto probing_iter = ref_.probing_scheme_(key, ref_.storage_ref_.num_windows());

    while (true) {
      // TODO atomic_ref::load if insert operator is present
      auto const window_slots = ref_.storage_ref_[*probing_iter];

      for (auto i = 0; i < window_size; ++i) {
        switch (ref_.predicate_(window_slots[i], key)) {
          case detail::equal_result::EMPTY: {
            return this->end();
          }
          case detail::equal_result::EQUAL: {
            return const_iterator{&(*(ref_.storage_ref_.data() + *probing_iter))[i]};
          }
          default: continue;
        }
      }
      ++probing_iter;
    }
  }

  /**
   * @brief Finds an element in the set with key equivalent to the probe key.
   *
   * @note Returns a un-incrementable input iterator to the element whose key is equivalent to
   * `key`. If no such element exists, returns `end()`.
   *
   * @tparam ProbeKey Probe key type
   *
   * @param group The Cooperative Group used to perform this operation
   * @param key The key to search for
   *
   * @return An iterator to the position at which the equivalent key is stored
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ const_iterator find(
    cooperative_groups::thread_block_tile<cg_size> const& group, ProbeKey const& key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);

    auto probing_iter = ref_.probing_scheme_(group, key, ref_.storage_ref_.num_windows());

    while (true) {
      auto const window_slots = ref_.storage_ref_[*probing_iter];

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (ref_.predicate_(window_slots[i], key)) {
            case detail::equal_result::EMPTY: return cuco::pair{detail::equal_result::EMPTY, i};
            case detail::equal_result::EQUAL: return cuco::pair{detail::equal_result::EQUAL, i};
            default: continue;
          }
        }
        // returns dummy index `-1` for UNEQUAL
        return cuco::pair<detail::equal_result, int32_t>{detail::equal_result::UNEQUAL, -1};
      }();

      // Find a match for the probe key, thus return an iterator to the entry
      auto const group_finds_match = group.ballot(state == detail::equal_result::EQUAL);
      if (group_finds_match) {
        auto const src_lane = __ffs(group_finds_match) - 1;
        auto const res =
          group.shfl(reinterpret_cast<intptr_t>(
                       &(*(ref_.storage_ref_.data() + *probing_iter))[intra_window_index]),
                     src_lane);
        return const_iterator{reinterpret_cast<value_type*>(res)};
      }

      // Find an empty slot, meaning that the probe key isn't present in the set
      if (group.any(state == detail::equal_result::EMPTY)) { return this->end(); }

      ++probing_iter;
    }
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco