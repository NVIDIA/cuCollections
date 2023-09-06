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
  : impl_{cuco::pair{empty_key_sentinel, empty_value_sentinel}, probing_scheme, storage_ref},
    empty_value_sentinel_{empty_value_sentinel},
    predicate_{empty_key_sentinel, predicate}
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
  return predicate_.empty_sentinel_;
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
  return empty_value_sentinel_;
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
struct static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
  predicate_wrapper {
  detail::equal_wrapper<key_type, key_equal> predicate_;

  /**
   * @brief Map predicate wrapper ctor.
   *
   * @param sentinel Sentinel value
   * @param equal Equality binary callable
   */
  __host__ __device__ constexpr predicate_wrapper(key_type empty_key_sentinel,
                                                  key_equal const& equal) noexcept
    : predicate_{empty_key_sentinel, equal}
  {
  }

  /**
   * @brief Equality check with the given equality callable.
   *
   * @tparam U Right-hand side Element type
   *
   * @param lhs Left-hand side element to check equality
   * @param rhs Right-hand side element to check equality
   *
   * @return `EQUAL` if `lhs` and `rhs` are equivalent. `UNEQUAL` otherwise.
   */
  template <typename U>
  __device__ constexpr detail::equal_result equal_to(value_type const& lhs,
                                                     U const& rhs) const noexcept
  {
    return predicate_.equal_to(lhs.first, rhs);
  }

  /**
   * @brief Equality check with the given equality callable.
   *
   * @param lhs Left-hand side element to check equality
   * @param rhs Right-hand side element to check equality
   *
   * @return `EQUAL` if `lhs` and `rhs` are equivalent. `UNEQUAL` otherwise.
   */
  __device__ constexpr detail::equal_result equal_to(value_type const& lhs,
                                                     value_type const& rhs) const noexcept
  {
    return predicate_.equal_to(lhs.first, rhs.first);
  }

  /**
   * @brief Equality check with the given equality callable.
   *
   * @param lhs Left-hand side key to check equality
   * @param rhs Right-hand side key to check equality
   *
   * @return `EQUAL` if `lhs` and `rhs` are equivalent. `UNEQUAL` otherwise.
   */
  __device__ constexpr detail::equal_result equal_to(key_type const& lhs,
                                                     key_type const& rhs) const noexcept
  {
    return predicate_.equal_to(lhs, rhs);
  }

  /**
   * @brief Order-sensitive equality operator.
   *
   * @note Container keys MUST be always on the left-hand side.
   *
   * @tparam U Right-hand side Element type
   *
   * @param lhs Left-hand side element to check equality
   * @param rhs Right-hand side element to check equality
   *
   * @return Three way equality comparison result
   */
  template <typename U>
  __device__ constexpr detail::equal_result operator()(value_type const& lhs,
                                                       U const& rhs) const noexcept
  {
    return predicate_(lhs.first, rhs);
  }
};

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
    ref_type& ref_             = static_cast<ref_type&>(*this);
    auto constexpr has_payload = false;
    return ref_.impl_.insert<has_payload>(value.first, value, ref_.predicate_);
  }

  /**
   * @brief Inserts an element.
   *
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   * @return True if the given element is successfully inserted
   */
  __device__ bool insert(cooperative_groups::thread_block_tile<cg_size> const& group,
                         value_type const& value) noexcept
  {
    auto& ref_                 = static_cast<ref_type&>(*this);
    auto constexpr has_payload = false;
    return ref_.impl_.insert<has_payload>(group, value.first, value, ref_.predicate_);
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
  using value_type = typename base_type::value_type;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

  static_assert(sizeof(T) == 4 or sizeof(T) == 8,
                "sizeof(mapped_type) must be either 4 bytes or 8 bytes.");

 public:
  /**
   * @brief Inserts a key-value pair `{k, v}` if it's not present in the map. Otherwise, assigns `v`
   * to the mapped_type corresponding to the key `k`.
   *
   * @param value The element to insert
   */
  __device__ void insert_or_assign(value_type const& value) noexcept
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");

    ref_type& ref_       = static_cast<ref_type&>(*this);
    auto const key       = value.first;
    auto& probing_scheme = ref_.impl_.probing_scheme();
    auto storage_ref     = ref_.impl_.storage_ref();
    auto probing_iter    = probing_scheme(key, storage_ref.window_extent());

    while (true) {
      auto const window_slots = storage_ref[*probing_iter];

      for (auto& slot_content : window_slots) {
        auto const eq_res = ref_.predicate_(slot_content, key);

        // If the key is already in the container, update the payload and return
        if (eq_res == detail::equal_result::EQUAL) {
          auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
          ref_.impl_.atomic_store(
            &((storage_ref.data() + *probing_iter)->data() + intra_window_index)->second,
            value.second);
          return;
        }
        if (eq_res == detail::equal_result::EMPTY) {
          auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
          if (attempt_insert_or_assign(
                (storage_ref.data() + *probing_iter)->data() + intra_window_index, value)) {
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
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   */
  __device__ void insert_or_assign(cooperative_groups::thread_block_tile<cg_size> const& group,
                                   value_type const& value) noexcept
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);

    auto const key       = value.first;
    auto& probing_scheme = ref_.impl_.probing_scheme();
    auto storage_ref     = ref_.impl_.storage_ref();
    auto probing_iter    = probing_scheme(group, key, storage_ref.window_extent());

    while (true) {
      auto const window_slots = storage_ref[*probing_iter];

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (ref_.predicate_(window_slots[i], key)) {
            case detail::equal_result::EMPTY:
              return detail::window_probing_results{detail::equal_result::EMPTY, i};
            case detail::equal_result::EQUAL:
              return detail::window_probing_results{detail::equal_result::EQUAL, i};
            default: continue;
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
            value.second);
        }
        group.sync();
        return;
      }

      auto const group_contains_empty = group.ballot(state == detail::equal_result::EMPTY);
      if (group_contains_empty) {
        auto const src_lane = __ffs(group_contains_empty) - 1;
        auto const status =
          (group.thread_rank() == src_lane)
            ? attempt_insert_or_assign(
                (storage_ref.data() + *probing_iter)->data() + intra_window_index, value)
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
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   *
   * @return Returns `true` if the given `value` is inserted or `value` has a match in the map.
   */
  __device__ constexpr bool attempt_insert_or_assign(value_type* slot,
                                                     value_type const& value) noexcept
  {
    ref_type& ref_          = static_cast<ref_type&>(*this);
    auto const expected_key = ref_.impl_.empty_slot_sentinel().first;

    auto old_key      = ref_.impl_.compare_and_swap(&slot->first, expected_key, value.first);
    auto* old_key_ptr = reinterpret_cast<key_type*>(&old_key);

    // if key success or key was already present in the map
    if (cuco::detail::bitwise_compare(*old_key_ptr, expected_key) or
        (ref_.predicate_.equal_to(*old_key_ptr, value.first) == detail::equal_result::EQUAL)) {
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
   * @param value The element to insert
   *
   * @return a pair consisting of an iterator to the element and a bool indicating whether the
   * insertion is successful or not.
   */
  __device__ thrust::pair<iterator, bool> insert_and_find(value_type const& value) noexcept
  {
    ref_type& ref_             = static_cast<ref_type&>(*this);
    auto constexpr has_payload = false;
    return ref_.impl_.insert_and_find<has_payload>(value.first, value, ref_.predicate_);
  }

  /**
   * @brief Inserts the given element into the map.
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
    ref_type& ref_             = static_cast<ref_type&>(*this);
    auto constexpr has_payload = false;
    return ref_.impl_.insert_and_find<has_payload>(group, value.first, value, ref_.predicate_);
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
   * @tparam ProbeKey Probe key type
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
    return ref_.impl_.contains(key, ref_.predicate_);
  }

  /**
   * @brief Indicates whether the probe key `key` was inserted into the container.
   *
   * @note If the probe key `key` was inserted into the container, returns
   * true. Otherwise, returns false.
   *
   * @tparam ProbeKey Probe key type
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
    return ref_.impl_.contains(group, key, ref_.predicate_);
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
    return ref_.impl_.find(key, ref_.predicate_);
  }

  /**
   * @brief Finds an element in the map with key equivalent to the probe key.
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
    return ref_.impl_.find(group, key, ref_.predicate_);
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
