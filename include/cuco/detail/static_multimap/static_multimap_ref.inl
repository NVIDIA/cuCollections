/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cuco/operator.hpp>

#include <cuda/atomic>
#include <cuda/std/functional>

#include <cooperative_groups.h>

namespace cuco {

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_multimap_ref<
  Key,
  T,
  Scope,
  KeyEqual,
  ProbingScheme,
  StorageRef,
  Operators...>::static_multimap_ref(cuco::empty_key<Key> empty_key_sentinel,
                                     cuco::empty_value<T> empty_value_sentinel,
                                     KeyEqual const& predicate,
                                     ProbingScheme const& probing_scheme,
                                     cuda_thread_scope<Scope>,
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
__host__ __device__ constexpr static_multimap_ref<
  Key,
  T,
  Scope,
  KeyEqual,
  ProbingScheme,
  StorageRef,
  Operators...>::static_multimap_ref(cuco::empty_key<Key> empty_key_sentinel,
                                     cuco::empty_value<T> empty_value_sentinel,
                                     cuco::erased_key<Key> erased_key_sentinel,
                                     KeyEqual const& predicate,
                                     ProbingScheme const& probing_scheme,
                                     cuda_thread_scope<Scope>,
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
__host__ __device__ constexpr static_multimap_ref<Key,
                                                  T,
                                                  Scope,
                                                  KeyEqual,
                                                  ProbingScheme,
                                                  StorageRef,
                                                  Operators...>::
  static_multimap_ref(
    static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, OtherOperators...>&&
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
__host__ __device__ constexpr static_multimap_ref<Key,
                                                  T,
                                                  Scope,
                                                  KeyEqual,
                                                  ProbingScheme,
                                                  StorageRef,
                                                  Operators...>::key_equal
static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::key_eq()
  const noexcept
{
  return impl_.key_eq();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_multimap_ref<Key,
                                                  T,
                                                  Scope,
                                                  KeyEqual,
                                                  ProbingScheme,
                                                  StorageRef,
                                                  Operators...>::const_iterator
static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::end()
  const noexcept
{
  return impl_.end();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_multimap_ref<Key,
                                                  T,
                                                  Scope,
                                                  KeyEqual,
                                                  ProbingScheme,
                                                  StorageRef,
                                                  Operators...>::iterator
static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
  end() noexcept
{
  return this->impl_.end();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr auto
static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::capacity()
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
__host__ __device__ constexpr auto
static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::storage_ref()
  const noexcept
{
  return this->impl_.storage_ref();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr auto
static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
  probing_scheme() const noexcept
{
  return this->impl_.probing_scheme();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_multimap_ref<Key,
                                                  T,
                                                  Scope,
                                                  KeyEqual,
                                                  ProbingScheme,
                                                  StorageRef,
                                                  Operators...>::extent_type
static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
  window_extent() const noexcept
{
  return impl_.window_extent();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr Key
static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
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
static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
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
__host__ __device__ constexpr Key
static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
  erased_key_sentinel() const noexcept
{
  return impl_.erased_key_sentinel();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename... NewOperators>
auto static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::with(
  NewOperators...) && noexcept
{
  return static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, NewOperators...>{
    std::move(*this)};
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename... NewOperators>
__host__ __device__ auto constexpr static_multimap_ref<
  Key,
  T,
  Scope,
  KeyEqual,
  ProbingScheme,
  StorageRef,
  Operators...>::with_operators(NewOperators...) const noexcept
{
  return static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, NewOperators...>{
    cuco::empty_key<Key>{this->empty_key_sentinel()},
    cuco::empty_value<T>{this->empty_value_sentinel()},
    this->key_eq(),
    impl_.probing_scheme(),
    {},
    impl_.storage_ref()};
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename CG, cuda::thread_scope NewScope>
__device__ constexpr auto
static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::make_copy(
  CG const& tile,
  window_type* const memory_to_use,
  cuda_thread_scope<NewScope> scope) const noexcept
{
  impl_.make_copy(tile, memory_to_use);
  return static_multimap_ref<Key, T, NewScope, KeyEqual, ProbingScheme, StorageRef, Operators...>{
    cuco::empty_key<Key>{this->empty_key_sentinel()},
    cuco::empty_value<T>{this->empty_value_sentinel()},
    cuco::erased_key<Key>{this->erased_key_sentinel()},
    this->key_eq(),
    impl_.probing_scheme(),
    scope,
    storage_ref_type{this->window_extent(), memory_to_use}};
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename CG>
__device__ constexpr void
static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::initialize(
  CG const& tile) noexcept
{
  impl_.initialize(tile);
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
  static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type =
    static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type    = typename base_type::key_type;
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
  op::contains_tag,
  static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type =
    static_multimap_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type   = typename base_type::key_type;
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
    return ref_.impl_.contains(key);
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
    return ref_.impl_.contains(group, key);
  }
};
}  // namespace detail
}  // namespace cuco
