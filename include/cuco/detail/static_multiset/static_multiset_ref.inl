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
__host__ __device__ constexpr static_multiset_ref<
  Key,
  Scope,
  KeyEqual,
  ProbingScheme,
  StorageRef,
  Operators...>::static_multiset_ref(cuco::empty_key<Key> empty_key_sentinel,
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
__host__ __device__ constexpr static_multiset_ref<
  Key,
  Scope,
  KeyEqual,
  ProbingScheme,
  StorageRef,
  Operators...>::static_multiset_ref(cuco::empty_key<Key> empty_key_sentinel,
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
__host__ __device__ constexpr static_multiset_ref<Key,
                                                  Scope,
                                                  KeyEqual,
                                                  ProbingScheme,
                                                  StorageRef,
                                                  Operators...>::
  static_multiset_ref(
    static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, OtherOperators...>&&
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
__host__ __device__ constexpr static_multiset_ref<Key,
                                                  Scope,
                                                  KeyEqual,
                                                  ProbingScheme,
                                                  StorageRef,
                                                  Operators...>::key_equal
static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::key_eq()
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
__host__ __device__ constexpr auto
static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::capacity()
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
__host__ __device__ constexpr static_multiset_ref<Key,
                                                  Scope,
                                                  KeyEqual,
                                                  ProbingScheme,
                                                  StorageRef,
                                                  Operators...>::extent_type
static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::window_extent()
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
static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
  empty_key_sentinel() const noexcept
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
static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
  erased_key_sentinel() const noexcept
{
  return impl_.erased_key_sentinel();
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_multiset_ref<Key,
                                                  Scope,
                                                  KeyEqual,
                                                  ProbingScheme,
                                                  StorageRef,
                                                  Operators...>::const_iterator
static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::end()
  const noexcept
{
  return this->impl_.end();
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_multiset_ref<Key,
                                                  Scope,
                                                  KeyEqual,
                                                  ProbingScheme,
                                                  StorageRef,
                                                  Operators...>::iterator
static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::end() noexcept
{
  return this->impl_.end();
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename... NewOperators>
auto static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::with(
  NewOperators...) && noexcept
{
  return static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, NewOperators...>{
    std::move(*this)};
}

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename... NewOperators>
__host__ __device__ constexpr auto
static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::with_operators(
  NewOperators...) const noexcept
{
  return static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, NewOperators...>{
    cuco::empty_key<Key>{this->empty_key_sentinel()},
    this->key_eq(),
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
template <typename NewKeyEqual>
__host__ __device__ constexpr auto
static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::with_key_eq(
  NewKeyEqual const& key_equal) const noexcept
{
  return static_multiset_ref<Key, Scope, NewKeyEqual, ProbingScheme, StorageRef, Operators...>{
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
static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
  with_hash_function(NewHash const& hash) const noexcept
{
  auto const probing_scheme = this->impl_.probing_scheme().with_hash_function(hash);
  return static_multiset_ref<Key,
                             Scope,
                             KeyEqual,
                             decltype(probing_scheme),
                             StorageRef,
                             Operators...>{cuco::empty_key<Key>{this->empty_key_sentinel()},
                                           this->impl_.key_eq(),
                                           probing_scheme,
                                           {},
                                           this->impl_.storage_ref()};
}

namespace detail {

template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<
  op::insert_tag,
  static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type =
    static_multiset_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
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

}  // namespace detail
}  // namespace cuco
