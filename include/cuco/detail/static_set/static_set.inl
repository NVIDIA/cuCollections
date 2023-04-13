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
#include <cuco/detail/storage/counter_storage.cuh>
#include <cuco/detail/tuning.cuh>
#include <cuco/detail/utils.hpp>
#include <cuco/operator.hpp>
#include <cuco/static_set_ref.cuh>

#include <thrust/iterator/transform_iterator.h>

#include <cub/device/device_reduce.cuh>

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
  : empty_key_sentinel_{empty_key_sentinel},
    predicate_{pred},
    probing_scheme_{probing_scheme},
    allocator_{alloc},
    storage_{make_valid_extent<cg_size, window_size>(capacity), allocator_}
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
static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::size_type
static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::insert(
  InputIt first, InputIt last, cudaStream_t stream)
{
  auto const num_keys = cuco::detail::distance(first, last);
  if (num_keys == 0) { return 0; }

  auto counter = detail::counter_storage<size_type, thread_scope, allocator_type>{allocator_};
  counter.reset(stream);

  auto const grid_size =
    (cg_size * num_keys + detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
    (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE);

  detail::insert<cg_size, detail::CUCO_DEFAULT_BLOCK_SIZE>
    <<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
      first, num_keys, counter.data(), ref(op::insert));

  return counter.load_to_host(stream);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt>
void static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::insert_async(
  InputIt first, InputIt last, cudaStream_t stream)
{
  auto const num_keys = cuco::detail::distance(first, last);
  if (num_keys == 0) { return; }

  auto const grid_size =
    (cg_size * num_keys + detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
    (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE);

  detail::insert<cg_size, detail::CUCO_DEFAULT_BLOCK_SIZE>
    <<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(first, num_keys, ref(op::insert));
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename StencilIt, typename Predicate>
static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::size_type
static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::insert_if(
  InputIt first, InputIt last, StencilIt stencil, Predicate pred, cudaStream_t stream)
{
  auto const num_keys = cuco::detail::distance(first, last);
  if (num_keys == 0) { return 0; }

  auto counter = detail::counter_storage<size_type, thread_scope, allocator_type>{allocator_};
  counter.reset(stream);

  auto const grid_size =
    (cg_size * num_keys + detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
    (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename StencilIt, typename Predicate>
void static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::insert_if_async(
  InputIt first, InputIt last, StencilIt stencil, Predicate pred, cudaStream_t stream)
{
  auto const num_keys = cuco::detail::distance(first, last);
  if (num_keys == 0) { return; }

  auto const grid_size =
    (cg_size * num_keys + detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
    (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE);
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
  contains_async(first, last, output_begin, stream);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename OutputIt>
void static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::contains_async(
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
        first, num_keys, output_begin, ref(op::contains));
  } else {
    detail::contains<cg_size, detail::CUCO_DEFAULT_BLOCK_SIZE>
      <<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
        first, num_keys, output_begin, ref(op::contains));
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
  auto counter = detail::counter_storage<size_type, thread_scope, allocator_type>{allocator_};
  counter.reset(stream);

  auto const grid_size =
    (storage_.num_windows() + detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
    (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE);

  // TODO: custom kernel to be replaced by cub::DeviceReduce::Sum when cub version is bumped to
  // v2.1.0
  detail::size<detail::CUCO_DEFAULT_BLOCK_SIZE>
    <<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
      storage_.ref(), this->empty_key_sentinel(), counter.data());

  return counter.load_to_host(stream);
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
auto static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::ref(
  Operators...) const noexcept
{
  static_assert(sizeof...(Operators), "No operators specified");
  return ref_type<Operators...>{
    cuco::empty_key<key_type>(empty_key_sentinel_), predicate_, probing_scheme_, storage_.ref()};
}
}  // namespace experimental
}  // namespace cuco
