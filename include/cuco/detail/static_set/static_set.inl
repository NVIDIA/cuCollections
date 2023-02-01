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

#include <cuco/detail/error.hpp>
#include <cuco/detail/prime.hpp>
#include <cuco/detail/static_set/kernels.cuh>
#include <cuco/detail/tuning.cuh>
#include <cuco/detail/utils.hpp>
#include <cuco/operator.hpp>
#include <cuco/static_set_ref.cuh>

#include <cstddef>

namespace cuco {
namespace experimental {

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
constexpr static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::static_set(
  Extent capacity,
  empty_key<Key> empty_key_sentinel,
  KeyEqual pred,
  ProbingScheme const& probing_scheme,
  Allocator const& alloc,
  cudaStream_t stream)
  : empty_key_sentinel_{empty_key_sentinel.value},
    predicate_{pred},
    probing_scheme_{probing_scheme},
    allocator_{alloc},
    storage_{capacity.template valid_extent<cg_size, window_size>(), allocator_}
{
  storage_.initialize(empty_key_sentinel_, stream);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt>
void static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::insert(
  InputIt first, InputIt last, cudaStream_t stream)
{
  auto const num_keys = cuco::detail::distance(first, last);
  if (num_keys == 0) { return; }

  auto const grid_size =
    (cg_size * num_keys + detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
    (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE);

  if constexpr (cg_size == 1) {
    detail::insert<detail::CUCO_DEFAULT_BLOCK_SIZE>
      <<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
        first, num_keys, ref_with(op::insert));
  } else {
    detail::insert<cg_size, detail::CUCO_DEFAULT_BLOCK_SIZE>
      <<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
        first, num_keys, ref_with(op::insert));
  }
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename OutputIt>
void static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::contains(
  InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream) const
{
  auto const num_keys = cuco::detail::distance(first, last);
  if (num_keys == 0) { return; }

  auto const grid_size =
    (cg_size * num_keys + detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
    (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE);

  if constexpr (cg_size == 1) {
    detail::contains<detail::CUCO_DEFAULT_BLOCK_SIZE>
      <<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
        first, num_keys, output_begin, ref_with(op::contains));
  } else {
    detail::contains<cg_size, detail::CUCO_DEFAULT_BLOCK_SIZE>
      <<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
        first, num_keys, output_begin, ref_with(op::contains));
  }
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::size_type
static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::size(
  cudaStream_t stream) const
{
  return storage_.size(empty_key_sentinel_, stream);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
constexpr auto
static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::capacity()
  const noexcept
{
  return storage_.capacity();
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
constexpr static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::key_type
static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::empty_key_sentinel()
  const noexcept
{
  return empty_key_sentinel_;
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename... Operators>
auto static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::ref_with(
  Operators...) const noexcept
{
  return static_set_ref<key_type,
                        thread_scope,
                        key_equal,
                        probing_scheme_type,
                        storage_ref_type,
                        Operators...>{
    cuco::empty_key<key_type>(empty_key_sentinel_), predicate_, probing_scheme_, storage_.ref()};
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
auto static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::ref()
  const noexcept
{
  return ref_with();
}
}  // namespace experimental
}  // namespace cuco
