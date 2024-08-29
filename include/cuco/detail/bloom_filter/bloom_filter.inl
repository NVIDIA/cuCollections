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

#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cuda/atomic>
#include <cuda/stream_ref>

#include <cstdint>

namespace cuco {

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
__host__ bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::bloom_filter(
  Extent num_blocks,
  uint32_t pattern_bits,
  cuda_thread_scope<Scope>,
  Hash const& hash,
  Allocator const& alloc,
  cuda::stream_ref stream)
  : allocator_{alloc},
    data_{
      allocator_.allocate(num_blocks * block_words),
      detail::custom_deleter<std::size_t, allocator_type>{num_blocks * block_words, allocator_}},
    ref_{data_.get(), num_blocks, pattern_bits, {}, hash}
{
  this->clear_async(stream);
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
__host__ void bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::clear(
  cuda::stream_ref stream)
{
  ref_.clear(stream);
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
__host__ void bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::clear_async(
  cuda::stream_ref stream)
{
  ref_.clear_async(stream);
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
template <class InputIt>
__host__ void bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::add(InputIt first,
                                                                            InputIt last,
                                                                            cuda::stream_ref stream)
{
  ref_.add(first, last, stream);
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
template <class InputIt>
__host__ void bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::add_async(
  InputIt first, InputIt last, cuda::stream_ref stream)
{
  ref_.add_async(first, last, stream);
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
template <class InputIt, class StencilIt, class Predicate>
__host__ void bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::add_if(
  InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream)
{
  ref_.add_if(first, last, stencil, pred, stream);
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
template <class InputIt, class StencilIt, class Predicate>
__host__ void bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::add_if_async(
  InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream) noexcept
{
  ref_.add_if_async(first, last, stencil, pred, stream);
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
template <class InputIt, class OutputIt>
__host__ void bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::test(
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const
{
  ref_.test(first, last, output_begin, stream);
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
template <class InputIt, class OutputIt>
__host__ void bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::test_async(
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const noexcept
{
  ref_.test_async(first, last, output_begin, stream);
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
template <class InputIt, class StencilIt, class Predicate, class OutputIt>
__host__ void bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::test_if(
  InputIt first,
  InputIt last,
  StencilIt stencil,
  Predicate pred,
  OutputIt output_begin,
  cuda::stream_ref stream) const
{
  ref_.test_if(first, last, stencil, pred, output_begin, stream);
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
template <class InputIt, class StencilIt, class Predicate, class OutputIt>
__host__ void bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::test_if_async(
  InputIt first,
  InputIt last,
  StencilIt stencil,
  Predicate pred,
  OutputIt output_begin,
  cuda::stream_ref stream) const noexcept
{
  ref_.test_if_async(first, last, stencil, pred, output_begin, stream);
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
[[nodiscard]] __host__ typename bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::word_type*
bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::data() noexcept
{
  return ref_.data();
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
[[nodiscard]] __host__
  typename bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::word_type const*
  bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::data() const noexcept
{
  return ref_.data();
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
[[nodiscard]] __host__
  typename bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::extent_type
  bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::block_extent() const noexcept
{
  return ref_.block_extent();
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
[[nodiscard]] __host__ uint32_t
bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::pattern_bits() const noexcept
{
  return ref_.pattern_bits();
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
[[nodiscard]] __host__ typename bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::hasher
bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::hash_function() const noexcept
{
  return ref_.hash_function();
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
[[nodiscard]] __host__
  typename bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::allocator_type
  bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::allocator() const noexcept
{
  return allocator_;
}

template <class Key,
          class Block,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          class Allocator>
[[nodiscard]] __host__ typename bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::ref_type<>
bloom_filter<Key, Block, Extent, Scope, Hash, Allocator>::ref() const noexcept
{
  return ref_;
}

}  // namespace cuco