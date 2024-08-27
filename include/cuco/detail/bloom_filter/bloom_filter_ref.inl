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

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
__host__ __device__ bloom_filter_ref<Key, Extent, Scope, Hash, BlockWords, Word>::bloom_filter_ref(
  word_type* data,
  Extent num_blocks,
  std::uint32_t pattern_bits,
  cuda_thread_scope<Scope>,
  Hash const& hash)
  : impl_{data, num_blocks, pattern_bits, {}, hash}
{
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
template <class CG>
__device__ void bloom_filter_ref<Key, Extent, Scope, Hash, BlockWords, Word>::clear(CG const& group)
{
  impl_.clear(group);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
__host__ void bloom_filter_ref<Key, Extent, Scope, Hash, BlockWords, Word>::clear(
  cuda::stream_ref stream)
{
  impl_.clear(stream);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
__host__ void bloom_filter_ref<Key, Extent, Scope, Hash, BlockWords, Word>::clear_async(
  cuda::stream_ref stream)
{
  impl_.clear_async(stream);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
template <class CG, class ProbeKey>
__device__ void bloom_filter_ref<Key, Extent, Scope, Hash, BlockWords, Word>::add(
  CG const& group, ProbeKey const& key)
{
  impl_.add(group, key);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
template <class InputIt>
__host__ void bloom_filter_ref<Key, Extent, Scope, Hash, BlockWords, Word>::add(
  InputIt first, InputIt last, cuda::stream_ref stream)
{
  impl_.add(first, last, stream);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
template <class InputIt>
__host__ void bloom_filter_ref<Key, Extent, Scope, Hash, BlockWords, Word>::add_async(
  InputIt first, InputIt last, cuda::stream_ref stream)
{
  impl_.add_async(first, last, stream);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
template <class InputIt, class StencilIt, class Predicate>
__host__ void bloom_filter_ref<Key, Extent, Scope, Hash, BlockWords, Word>::add_if(
  InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream)
{
  impl_.add_if(first, last, stencil, pred, stream);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
template <class InputIt, class StencilIt, class Predicate>
__host__ void bloom_filter_ref<Key, Extent, Scope, Hash, BlockWords, Word>::add_if_async(
  InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream) noexcept
{
  impl_.add_if_async(first, last, stencil, pred, stream);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
template <class ProbeKey>
[[nodiscard]] __device__ bool bloom_filter_ref<Key, Extent, Scope, Hash, BlockWords, Word>::test(
  ProbeKey const& key) const
{
  return impl_.test(key);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
template <class InputIt, class OutputIt>
__host__ void bloom_filter_ref<Key, Extent, Scope, Hash, BlockWords, Word>::test(
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const
{
  impl_.test(first, last, output_begin, stream);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
template <class InputIt, class OutputIt>
__host__ void bloom_filter_ref<Key, Extent, Scope, Hash, BlockWords, Word>::test_async(
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const noexcept
{
  impl_.test_async(first, last, output_begin, stream);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
template <class InputIt, class StencilIt, class Predicate, class OutputIt>
__host__ void bloom_filter_ref<Key, Extent, Scope, Hash, BlockWords, Word>::test_if(
  InputIt first,
  InputIt last,
  StencilIt stencil,
  Predicate pred,
  OutputIt output_begin,
  cuda::stream_ref stream) const
{
  impl_.test_if(first, last, stencil, pred, output_begin, stream);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          std::uint32_t BlockWords,
          class Word>
template <class InputIt, class StencilIt, class Predicate, class OutputIt>
__host__ void bloom_filter_ref<Key, Extent, Scope, Hash, BlockWords, Word>::test_if_async(
  InputIt first,
  InputIt last,
  StencilIt stencil,
  Predicate pred,
  OutputIt output_begin,
  cuda::stream_ref stream) const noexcept
{
  impl_.test_if_async(first, last, stencil, pred, output_begin, stream);
}

}  // namespace cuco