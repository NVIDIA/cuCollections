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

#include <cuco/utility/cuda_thread_scope.cuh>

#include <cuda/atomic>
#include <cuda/stream_ref>

#include <cstdint>

namespace cuco {

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
__host__ __device__
bloom_filter_ref<Key, Block, Extent, Scope, Hash>::bloom_filter_ref(word_type* data,
                                                                    Extent num_blocks,
                                                                    std::uint32_t pattern_bits,
                                                                    cuda_thread_scope<Scope>,
                                                                    Hash const& hash)
  : impl_{data, num_blocks, pattern_bits, {}, hash}
{
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
template <class CG>
__device__ void bloom_filter_ref<Key, Block, Extent, Scope, Hash>::clear(CG const& group)
{
  impl_.clear(group);
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
__host__ void bloom_filter_ref<Key, Block, Extent, Scope, Hash>::clear(cuda::stream_ref stream)
{
  impl_.clear(stream);
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
__host__ void bloom_filter_ref<Key, Block, Extent, Scope, Hash>::clear_async(
  cuda::stream_ref stream)
{
  impl_.clear_async(stream);
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
template <class ProbeKey>
__device__ void bloom_filter_ref<Key, Block, Extent, Scope, Hash>::add(ProbeKey const& key)
{
  impl_.add(key);
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
template <class CG, class ProbeKey>
__device__ void bloom_filter_ref<Key, Block, Extent, Scope, Hash>::add(CG const& group,
                                                                       ProbeKey const& key)
{
  impl_.add(group, key);
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
template <class InputIt>
__host__ void bloom_filter_ref<Key, Block, Extent, Scope, Hash>::add(InputIt first,
                                                                     InputIt last,
                                                                     cuda::stream_ref stream)
{
  impl_.add(first, last, stream);
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
template <class InputIt>
__host__ void bloom_filter_ref<Key, Block, Extent, Scope, Hash>::add_async(InputIt first,
                                                                           InputIt last,
                                                                           cuda::stream_ref stream)
{
  impl_.add_async(first, last, stream);
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
template <class InputIt, class StencilIt, class Predicate>
__host__ void bloom_filter_ref<Key, Block, Extent, Scope, Hash>::add_if(
  InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream)
{
  impl_.add_if(first, last, stencil, pred, stream);
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
template <class InputIt, class StencilIt, class Predicate>
__host__ void bloom_filter_ref<Key, Block, Extent, Scope, Hash>::add_if_async(
  InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream) noexcept
{
  impl_.add_if_async(first, last, stencil, pred, stream);
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
template <class ProbeKey>
[[nodiscard]] __device__ bool bloom_filter_ref<Key, Block, Extent, Scope, Hash>::contains(
  ProbeKey const& key) const
{
  return impl_.contains(key);
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
template <class InputIt, class OutputIt>
__host__ void bloom_filter_ref<Key, Block, Extent, Scope, Hash>::contains(
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const
{
  impl_.contains(first, last, output_begin, stream);
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
template <class InputIt, class OutputIt>
__host__ void bloom_filter_ref<Key, Block, Extent, Scope, Hash>::contains_async(
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const noexcept
{
  impl_.contains_async(first, last, output_begin, stream);
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
template <class InputIt, class StencilIt, class Predicate, class OutputIt>
__host__ void bloom_filter_ref<Key, Block, Extent, Scope, Hash>::contains_if(
  InputIt first,
  InputIt last,
  StencilIt stencil,
  Predicate pred,
  OutputIt output_begin,
  cuda::stream_ref stream) const
{
  impl_.contains_if(first, last, stencil, pred, output_begin, stream);
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
template <class InputIt, class StencilIt, class Predicate, class OutputIt>
__host__ void bloom_filter_ref<Key, Block, Extent, Scope, Hash>::contains_if_async(
  InputIt first,
  InputIt last,
  StencilIt stencil,
  Predicate pred,
  OutputIt output_begin,
  cuda::stream_ref stream) const noexcept
{
  impl_.contains_if_async(first, last, stencil, pred, output_begin, stream);
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
[[nodiscard]] __host__ __device__
  typename bloom_filter_ref<Key, Block, Extent, Scope, Hash>::word_type*
  bloom_filter_ref<Key, Block, Extent, Scope, Hash>::data() noexcept
{
  return impl_.data();
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
[[nodiscard]] __host__ __device__
  typename bloom_filter_ref<Key, Block, Extent, Scope, Hash>::word_type const*
  bloom_filter_ref<Key, Block, Extent, Scope, Hash>::data() const noexcept
{
  return impl_.data();
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
[[nodiscard]] __host__ __device__
  typename bloom_filter_ref<Key, Block, Extent, Scope, Hash>::extent_type
  bloom_filter_ref<Key, Block, Extent, Scope, Hash>::block_extent() const noexcept
{
  return impl_.block_extent();
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
[[nodiscard]] __host__ __device__ uint32_t
bloom_filter_ref<Key, Block, Extent, Scope, Hash>::pattern_bits() const noexcept
{
  return impl_.pattern_bits();
}

template <class Key, class Block, class Extent, cuda::thread_scope Scope, class Hash>
[[nodiscard]] __host__ __device__ typename bloom_filter_ref<Key, Block, Extent, Scope, Hash>::hasher
bloom_filter_ref<Key, Block, Extent, Scope, Hash>::hash_function() const noexcept
{
  return impl_.hash_function();
}

}  // namespace cuco