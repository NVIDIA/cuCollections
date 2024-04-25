/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
                                cuda_thread_scope<Scope>,
                                StorageRef storage_ref) noexcept
  : impl_{empty_key_sentinel, predicate, probing_scheme, storage_ref}
{
}

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
                                cuco::erased_key<Key> erased_key_sentinel,
                                KeyEqual const& predicate,
                                ProbingScheme const& probing_scheme,
                                cuda_thread_scope<Scope>,
                                StorageRef storage_ref) noexcept
  : impl_{empty_key_sentinel, erased_key_sentinel, predicate, probing_scheme, storage_ref}
{
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename... OtherOperators>
__host__ __device__ constexpr static_set_ref<Key,
                                             Scope,
                                             KeyEqual,
                                             ProbingScheme,
                                             StorageRef,
                                             Operators...>::
  static_set_ref(
    static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, OtherOperators...>&&
      other) noexcept
  : impl_{std::move(other.impl_)}
{
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_set_ref<Key,
                                             Scope,
                                             KeyEqual,
                                             ProbingScheme,
                                             StorageRef,
                                             Operators...>::key_equal
static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::key_eq()
  const noexcept
{
  return this->impl_.key_eq();
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_set_ref<Key,
                                             Scope,
                                             KeyEqual,
                                             ProbingScheme,
                                             StorageRef,
                                             Operators...>::const_iterator
static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::end() const noexcept
{
  return this->impl_.end();
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_set_ref<Key,
                                             Scope,
                                             KeyEqual,
                                             ProbingScheme,
                                             StorageRef,
                                             Operators...>::iterator
static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::end() noexcept
{
  return this->impl_.end();
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
  return impl_.capacity();
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_set_ref<Key,
                                             Scope,
                                             KeyEqual,
                                             ProbingScheme,
                                             StorageRef,
                                             Operators...>::extent_type
static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::window_extent()
  const noexcept
{
  return impl_.window_extent();
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
  return impl_.empty_key_sentinel();
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr Key
static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::erased_key_sentinel()
  const noexcept
{
  return impl_.erased_key_sentinel();
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename... NewOperators>
auto static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::with(
  NewOperators...) && noexcept
{
  return static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, NewOperators...>{
    std::move(*this)};
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename NewKeyEqual>
__host__ __device__ constexpr auto
static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::with_key_eq(
  NewKeyEqual const& key_equal) const noexcept
{
  return static_set_ref<Key, Scope, NewKeyEqual, ProbingScheme, StorageRef, Operators...>{
    cuco::empty_key<Key>{this->empty_key_sentinel()},
    key_equal,
    this->impl_.probing_scheme(),
    {},
    this->impl_.storage_ref()};
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename NewHash>
__host__ __device__ constexpr auto
static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::with_hash_function(
  NewHash const& hash) const noexcept
{
  auto const probing_scheme = this->impl_.probing_scheme().make_copy(hash);
  return static_set_ref<Key, Scope, KeyEqual, decltype(probing_scheme), StorageRef, Operators...>{
    cuco::empty_key<Key>{this->empty_key_sentinel()},
    this->impl_.key_eq(),
    probing_scheme,
    {},
    this->impl_.storage_ref()};
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename CG, cuda::thread_scope NewScope>
__device__ constexpr auto
static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::make_copy(
  CG const& tile,
  window_type* const memory_to_use,
  cuda_thread_scope<NewScope> scope) const noexcept
{
  this->impl_.make_copy(tile, memory_to_use);
  return static_set_ref<Key, NewScope, KeyEqual, ProbingScheme, StorageRef, Operators...>{
    cuco::empty_key<Key>{this->empty_key_sentinel()},
    cuco::erased_key<Key>{this->erased_key_sentinel()},
    this->key_eq(),
    this->impl_.probing_scheme(),
    scope,
    storage_ref_type{this->window_extent(), memory_to_use}};
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename CG>
__device__ constexpr void
static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::initialize(
  CG const& tile) noexcept
{
  this->impl_.initialize(tile);
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
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<op::insert_and_find_tag,
                    static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type  = static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type   = static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type   = typename base_type::key_type;
  using value_type = typename base_type::value_type;
  using iterator   = typename base_type::iterator;
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
   * @brief Inserts the given element into the set.
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
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<op::erase_tag,
                    static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type  = static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type   = static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type   = typename base_type::key_type;
  using value_type = typename base_type::value_type;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Erases an element.
   *
   * @tparam ProbeKey Input type which is convertible to 'key_type'
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
   * @tparam ProbeKey Input type which is convertible to 'key_type'
   *
   * @param group The Cooperative Group used to perform group erase
   * @param value The element to erase
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
   * @note If the probe key `key` was inserted into the container, returns true. Otherwise, returns
   * false.
   *
   * @tparam ProbeKey Input type which is convertible to 'key_type'
   *
   * @param key The key to search for
   *
   * @return A boolean indicating whether the probe key is present
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ bool contains(ProbeKey const& key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return ref_.impl_.contains(key);
  }

  /**
   * @brief Indicates whether the probe key `key` was inserted into the container.
   *
   * @note If the probe key `key` was inserted into the container, returns true. Otherwise, returns
   * false.
   *
   * @tparam ProbeKey Input type which is convertible to 'key_type'
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
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<op::find_tag,
                    static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type  = static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type   = static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type   = typename base_type::key_type;
  using value_type = typename base_type::value_type;
  using iterator   = typename base_type::iterator;
  using const_iterator = typename base_type::const_iterator;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Finds an element in the set with key equivalent to the probe key.
   *
   * @note Returns a un-incrementable input iterator to the element whose key is equivalent to
   * `key`. If no such element exists, returns `end()`.
   *
   * @tparam ProbeKey Input type which is convertible to 'key_type'
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
   * @brief Finds an element in the set with key equivalent to the probe key.
   *
   * @note Returns a un-incrementable input iterator to the element whose key is equivalent to
   * `key`. If no such element exists, returns `end()`.
   *
   * @tparam ProbeKey Input type which is convertible to 'key_type'
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
}  // namespace cuco
