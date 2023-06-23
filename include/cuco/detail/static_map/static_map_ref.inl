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

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__device__ constexpr auto
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::back_to_back_cas(
  value_type* slot, value_type const& value) noexcept
{
  auto expected_key     = this->get_empty_key_sentinel();
  auto expected_payload = this->get_empty_value_sentinel();

  auto old_key     = compare_and_swap(&slot->first, expected_key, value.first);
  auto old_payload = compare_and_swap(&slot->second, expected_payload, value.second);

  auto* old_key_ptr     = reinterpret_cast<key_type*>(&old_key);
  auto* old_payload_ptr = reinterpret_cast<mapped_type*>(&old_payload);

  // if key success
  if (cuco::detail::bitwise_compare(*old_key_ptr, expected_key)) {
    while (not cuco::detail::bitwise_compare(*old_payload_ptr, expected_payload)) {
      old_payload = compare_and_swap(&slot->second, expected_payload, value.second);
    }
    return insert_result::SUCCESS;
  } else if (cuco::detail::bitwise_compare(*old_payload_ptr, expected_payload)) {
    auto const desired = this->get_empty_value_sentinel();
    if constexpr (sizeof(mapped_type) == sizeof(unsigned int)) {
      if constexpr (Scope == cuda::thread_scope_system) {
        atomicExch_system(reinterpret_cast<unsigned int*>(&slot->second), desired);
      } else if constexpr (Scope == cuda::thread_scope_device) {
        atomicExch(reinterpret_cast<unsigned int*>(&slot->second), desired);
      } else if constexpr (Scope == cuda::thread_scope_block) {
        atomicExch_block(reinterpret_cast<unsigned int*>(&slot->second), desired);
      } else {
        static_assert(cuco::dependent_false<decltype(Scope)>, "Unsupported thread scope");
      }
    } else if constexpr (sizeof(mapped_type) == sizeof(unsigned long long int)) {
      if constexpr (Scope == cuda::thread_scope_system) {
        atomicExch_system(reinterpret_cast<unsigned long long int*>(&slot->second), desired);
      } else if constexpr (Scope == cuda::thread_scope_device) {
        atomicExch(reinterpret_cast<unsigned long long int*>(&slot->second), desired);
      } else if constexpr (Scope == cuda::thread_scope_block) {
        atomicExch_block(reinterpret_cast<unsigned long long int*>(&slot->second), desired);
      } else {
        static_assert(cuco::dependent_false<decltype(Scope)>, "Unsupported thread scope");
      }
    }
  }

  // Our key was already present in the slot, so our key is a duplicate
  // Shouldn't use `predicate` operator directly since it includes a redundant bitwise compare
  if (predicate_.equal_to(*old_key_ptr, value.first) == detail::equal_result::EQUAL) {
    return insert_result::DUPLICATE;
  }

  return insert_result::CONTINUE;
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
    ref_type& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.insert(value.first, value, ref_.predicate_);
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
    auto& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.insert(group, value.first, value, ref_.predicate_);
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
    ref_type& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.insert_and_find(value.first, value, ref_.predicate_);
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
    ref_type& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.insert_and_find(group, value.first, value, ref_.predicate_);
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
