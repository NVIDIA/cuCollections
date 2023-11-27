/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuco/operator.hpp>

#include <thrust/tuple.h>

#include <cuda/atomic>

#include <cooperative_groups.h>

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
  : impl_{
      cuco::pair{empty_key_sentinel, empty_value_sentinel}, predicate, probing_scheme, storage_ref}
{
}

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
                                cuco::erased_key<Key> erased_key_sentinel,
                                KeyEqual const& predicate,
                                ProbingScheme const& probing_scheme,
                                StorageRef storage_ref) noexcept
  : impl_{cuco::pair{empty_key_sentinel, empty_value_sentinel},
          erased_key_sentinel,
          predicate,
          probing_scheme,
          storage_ref}
{
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename... OtherOperators>
__host__ __device__ constexpr static_map_ref<Key,
                                             T,
                                             Scope,
                                             KeyEqual,
                                             ProbingScheme,
                                             StorageRef,
                                             Operators...>::
  static_map_ref(
    static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, OtherOperators...>&&
      other) noexcept
  : impl_{std::move(other.impl_)}
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
  return impl_.capacity();
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
  return impl_.empty_key_sentinel();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr T
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
  empty_value_sentinel() const noexcept
{
  return impl_.empty_value_sentinel();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename... NewOperators>
auto static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::with(
  NewOperators...) && noexcept
{
  return static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, NewOperators...>(
    std::move(*this));
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
  using value_type  = typename base_type::value_type;
  using mapped_type = T;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Inserts an element.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param value The element to insert
   *
   * @return True if the given element is successfully inserted
   */
  template <typename Value>
  __device__ bool insert(Value const& value) noexcept
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.insert(value);
  }

  /**
   * @brief Inserts an element.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   *
   * @return True if the given element is successfully inserted
   */
  template <typename Value>
  __device__ bool insert(cooperative_groups::thread_block_tile<cg_size> const& group,
                         Value const& value) noexcept
  {
    auto& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.insert(group, value);
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
  op::insert_or_assign_tag,
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type = typename base_type::key_type;
  using value_type  = typename base_type::value_type;
  using mapped_type = T;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

  static_assert(sizeof(T) == 4 or sizeof(T) == 8,
                "sizeof(mapped_type) must be either 4 bytes or 8 bytes.");

 public:
  /**
   * @brief Inserts a key-value pair `{k, v}` if it's not present in the map. Otherwise, assigns `v`
   * to the mapped_type corresponding to the key `k`.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param value The element to insert
   */
  template <typename Value>
  __device__ void insert_or_assign(Value const& value) noexcept
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");

    ref_type& ref_ = static_cast<ref_type&>(*this);

    auto const val       = ref_.impl_.heterogeneous_value(value);
    auto const key       = ref_.impl_.extract_key(val);
    auto& probing_scheme = ref_.impl_.probing_scheme();
    auto storage_ref     = ref_.impl_.storage_ref();
    auto probing_iter    = probing_scheme(key, storage_ref.window_extent());

    while (true) {
      auto const window_slots = storage_ref[*probing_iter];

      for (auto& slot_content : window_slots) {
        auto const eq_res = ref_.impl_.predicate()(slot_content.first, key);

        // If the key is already in the container, update the payload and return
        if (eq_res == detail::equal_result::EQUAL) {
          auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
          ref_.impl_.atomic_store(
            &((storage_ref.data() + *probing_iter)->data() + intra_window_index)->second,
            val.second);
          return;
        }
        if (eq_res == detail::equal_result::EMPTY or
            cuco::detail::bitwise_compare(slot_content.first, ref_.impl_.erased_key_sentinel())) {
          auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
          if (attempt_insert_or_assign(
                (storage_ref.data() + *probing_iter)->data() + intra_window_index, val)) {
            return;
          }
        }
      }
      ++probing_iter;
    }
  }

  /**
   * @brief Inserts an element.
   *
   * @brief Inserts a key-value pair `{k, v}` if it's not present in the map. Otherwise, assigns `v`
   * to the mapped_type corresponding to the key `k`.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   */
  template <typename Value>
  __device__ void insert_or_assign(cooperative_groups::thread_block_tile<cg_size> const& group,
                                   Value const& value) noexcept
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);

    auto const val       = ref_.impl_.heterogeneous_value(value);
    auto const key       = ref_.impl_.extract_key(val);
    auto& probing_scheme = ref_.impl_.probing_scheme();
    auto storage_ref     = ref_.impl_.storage_ref();
    auto probing_iter    = probing_scheme(group, key, storage_ref.window_extent());

    while (true) {
      auto const window_slots = storage_ref[*probing_iter];

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (ref_.impl_.predicate()(window_slots[i].first, key)) {
            case detail::equal_result::EMPTY:
              return detail::window_probing_results{detail::equal_result::EMPTY, i};
            case detail::equal_result::EQUAL:
              return detail::window_probing_results{detail::equal_result::EQUAL, i};
            default: {
              if (cuco::detail::bitwise_compare(window_slots[i].first,
                                                ref_.impl_.erased_key_sentinel())) {
                return window_probing_results{detail::equal_result::ERASED, i};
              } else {
                continue;
              }
            }
          }
        }
        // returns dummy index `-1` for UNEQUAL
        return detail::window_probing_results{detail::equal_result::UNEQUAL, -1};
      }();

      auto const group_contains_equal = group.ballot(state == detail::equal_result::EQUAL);
      if (group_contains_equal) {
        auto const src_lane = __ffs(group_contains_equal) - 1;
        if (group.thread_rank() == src_lane) {
          ref_.impl_.atomic_store(
            &((storage_ref.data() + *probing_iter)->data() + intra_window_index)->second,
            val.second);
        }
        group.sync();
        return;
      }

      auto const group_contains_available =
        group.ballot(state == detail::equal_result::EMPTY or state == detail::equal_result::ERASED);
      if (group_contains_available) {
        auto const src_lane = __ffs(group_contains_available) - 1;
        auto const status =
          (group.thread_rank() == src_lane)
            ? attempt_insert_or_assign(
                (storage_ref.data() + *probing_iter)->data() + intra_window_index, val)
            : false;

        // Exit if inserted or assigned
        if (group.shfl(status, src_lane)) { return; }
      } else {
        ++probing_iter;
      }
    }
  }

 private:
  /**
   * @brief Attempts to insert an element into a slot or update the matching payload with the given
   * element
   *
   * @brief Inserts a key-value pair `{k, v}` if it's not present in the map. Otherwise, assigns `v`
   * to the mapped_type corresponding to the key `k`.
   *
   * @tparam Value Input type which is implicitly convertible to 'value_type'
   *
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   *
   * @return Returns `true` if the given `value` is inserted or `value` has a match in the map.
   */
  template <typename Value>
  __device__ constexpr bool attempt_insert_or_assign(value_type* slot, Value const& value) noexcept
  {
    ref_type& ref_          = static_cast<ref_type&>(*this);
    auto const expected_key = ref_.impl_.empty_slot_sentinel().first;

    auto old_key =
      ref_.impl_.compare_and_swap(&slot->first, expected_key, static_cast<key_type>(value.first));
    auto* old_key_ptr = reinterpret_cast<key_type*>(&old_key);

    // if key success or key was already present in the map
    if (cuco::detail::bitwise_compare(*old_key_ptr, expected_key) or
        (ref_.impl_.predicate().equal_to(*old_key_ptr, value.first) ==
         detail::equal_result::EQUAL)) {
      // Update payload
      ref_.impl_.atomic_store(&slot->second, value.second);
      return true;
    }
    return false;
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
  using mapped_type    = T;
  using iterator       = typename base_type::iterator;
  using const_iterator = typename base_type::const_iterator;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Returns a const_iterator to one past the last slot.
   *
   * @note This API is available only when `find_tag` or `insert_and_find_tag` is present.
   *
   * @return A const_iterator to one past the last slot
   */
  [[nodiscard]] __host__ __device__ constexpr const_iterator end() const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return ref_.impl_.end();
  }

  /**
   * @brief Returns an iterator to one past the last slot.
   *
   * @note This API is available only when `find_tag` or `insert_and_find_tag` is present.
   *
   * @return An iterator to one past the last slot
   */
  [[nodiscard]] __host__ __device__ constexpr iterator end() noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return ref_.impl_.end();
  }

  /**
   * @brief Inserts the given element into the map.
   *
   * @note This API returns a pair consisting of an iterator to the inserted element (or to the
   * element that prevented the insertion) and a `bool` denoting whether the insertion took place or
   * not.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param value The element to insert
   *
   * @return a pair consisting of an iterator to the element and a bool indicating whether the
   * insertion is successful or not.
   */
  template <typename Value>
  __device__ thrust::pair<iterator, bool> insert_and_find(Value const& value) noexcept
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.insert_and_find(value);
  }

  /**
   * @brief Inserts the given element into the map.
   *
   * @note This API returns a pair consisting of an iterator to the inserted element (or to the
   * element that prevented the insertion) and a `bool` denoting whether the insertion took place or
   * not.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param group The Cooperative Group used to perform group insert_and_find
   * @param value The element to insert
   *
   * @return a pair consisting of an iterator to the element and a bool indicating whether the
   * insertion is successful or not.
   */
  template <typename Value>
  __device__ thrust::pair<iterator, bool> insert_and_find(
    cooperative_groups::thread_block_tile<cg_size> const& group, Value const& value) noexcept
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.insert_and_find(group, value);
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
  op::erase_tag,
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type = typename base_type::key_type;
  using value_type = typename base_type::value_type;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Erases an element.
   *
   * @tparam ProbeKey Input key type which is convertible to 'key_type'
   *
   * @param key The element to erase
   *
   * @return True if the given element is successfully erased
   */
  template <typename ProbeKey>
  __device__ bool erase(ProbeKey const& key) noexcept
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.erase(key);
  }

  /**
   * @brief Erases an element.
   *
   * @tparam ProbeKey Input key type which is convertible to 'key_type'
   *
   * @param group The Cooperative Group used to perform group insert
   * @param key The element to erase
   *
   * @return True if the given element is successfully erased
   */
  template <typename ProbeKey>
  __device__ bool erase(cooperative_groups::thread_block_tile<cg_size> const& group,
                        ProbeKey const& key) noexcept
  {
    auto& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.erase(group, key);
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
   * @note If the probe key `key` was inserted into the container, returns
   * true. Otherwise, returns false.
   *
   * @tparam ProbeKey Input key type which is convertible to 'key_type'
   *
   * @param key The key to search for
   *
   * @return A boolean indicating whether the probe key is present
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ bool contains(ProbeKey const& key) const noexcept
  {
    // CRTP: cast `this` to the actual ref type
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return ref_.impl_.contains(key);
  }

  /**
   * @brief Indicates whether the probe key `key` was inserted into the container.
   *
   * @note If the probe key `key` was inserted into the container, returns
   * true. Otherwise, returns false.
   *
   * @tparam ProbeKey Input key type which is convertible to 'key_type'
   *
   * @param group The Cooperative Group used to perform group contains
   * @param key The key to search for
   *
   * @return A boolean indicating whether the probe key is present
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ bool contains(
    cooperative_groups::thread_block_tile<cg_size> const& group, ProbeKey const& key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return ref_.impl_.contains(group, key);
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
   * @note This API is available only when `find_tag` or `insert_and_find_tag` is present.
   *
   * @return A const_iterator to one past the last slot
   */
  [[nodiscard]] __host__ __device__ constexpr const_iterator end() const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return ref_.impl_.end();
  }

  /**
   * @brief Returns an iterator to one past the last slot.
   *
   * @note This API is available only when `find_tag` or `insert_and_find_tag` is present.
   *
   * @return An iterator to one past the last slot
   */
  [[nodiscard]] __host__ __device__ constexpr iterator end() noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return ref_.impl_.end();
  }

  /**
   * @brief Finds an element in the map with key equivalent to the probe key.
   *
   * @note Returns a un-incrementable input iterator to the element whose key is equivalent to
   * `key`. If no such element exists, returns `end()`.
   *
   * @tparam ProbeKey Input key type which is convertible to 'key_type'
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
    return ref_.impl_.find(key);
  }

  /**
   * @brief Finds an element in the map with key equivalent to the probe key.
   *
   * @note Returns a un-incrementable input iterator to the element whose key is equivalent to
   * `key`. If no such element exists, returns `end()`.
   *
   * @tparam ProbeKey Input key type which is convertible to 'key_type'
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
    return ref_.impl_.find(group, key);
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
