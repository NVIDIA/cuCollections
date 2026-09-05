/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/error.hpp>
#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cuda/atomic>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/limits>
#include <cuda/stream_ref>

#include <cstddef>

namespace cuco {

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
__host__ bloom_filter<Key, Extent, Scope, Policy, Allocator>::bloom_filter(
  bloom_filter_size_bytes size_bytes,
  cuda_thread_scope<Scope> scope,
  Policy const& policy,
  Allocator const& alloc,
  cuda::stream_ref stream)
  : bloom_filter{make_block_extent(size_bytes, Extent{0}), scope, policy, alloc, stream}
{
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
[[nodiscard]] __host__ constexpr std::size_t
bloom_filter<Key, Extent, Scope, Policy, Allocator>::max_size() noexcept
{
  constexpr auto block_bytes = words_per_block * sizeof(word_type);
  constexpr auto max_blocks  = cuda::std::min(
    static_cast<std::size_t>(Policy::max_filter_blocks),
    cuda::std::min(static_cast<std::size_t>(cuda::std::numeric_limits<size_type>::max()),
                   cuda::std::numeric_limits<std::size_t>::max() / block_bytes));
  return max_blocks * block_bytes;
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
[[nodiscard]] __host__ constexpr std::size_t
bloom_filter<Key, Extent, Scope, Policy, Allocator>::aligned_size(std::size_t size_bytes)
{
  constexpr auto block_bytes = words_per_block * sizeof(word_type);
  CUCO_EXPECTS(size_bytes >= block_bytes,
               "Storage size must accommodate at least one filter block");
  auto const capped_bytes = cuda::std::min(size_bytes, max_size());
  return capped_bytes - capped_bytes % block_bytes;
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
template <typename SizeType, std::size_t N>
__host__ constexpr typename bloom_filter<Key, Extent, Scope, Policy, Allocator>::extent_type
bloom_filter<Key, Extent, Scope, Policy, Allocator>::make_block_extent(
  bloom_filter_size_bytes size_bytes, cuco::extent<SizeType, N>)
{
  static_assert(N == cuco::dynamic_extent || detail::is_static_extent_representable<SizeType, N>(),
                "Static extent must be representable by its size type");
  constexpr auto block_bytes = words_per_block * sizeof(word_type);
  CUCO_EXPECTS(size_bytes.value > 0, "Storage size must be positive");
  CUCO_EXPECTS(size_bytes.value % block_bytes == 0,
               "Storage size must be a multiple of the filter block size");
  CUCO_EXPECTS(size_bytes.value <= max_size(), "Storage size exceeds the maximum filter size");
  auto const num_blocks = size_bytes.value / block_bytes;
  auto const extent     = extent_type{static_cast<size_type>(num_blocks)};
  CUCO_EXPECTS(static_cast<std::size_t>(static_cast<size_type>(extent)) == num_blocks,
               "Storage size must match the static block extent");
  return extent;
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
__host__ bloom_filter<Key, Extent, Scope, Policy, Allocator>::bloom_filter(Extent num_blocks,
                                                                           cuda_thread_scope<Scope>,
                                                                           Policy const& policy,
                                                                           Allocator const& alloc,
                                                                           cuda::stream_ref stream)
  : allocator_{alloc},
    data_{allocator_.allocate(static_cast<size_type>(num_blocks), stream),
          detail::custom_deleter<std::size_t, allocator_type>{
            static_cast<size_type>(num_blocks), allocator_, stream}},
    ref_{data_.get(), num_blocks, {}, policy}
{
  this->clear_async(stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
__host__ constexpr void bloom_filter<Key, Extent, Scope, Policy, Allocator>::clear(
  cuda::stream_ref stream)
{
  ref_.clear(stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
__host__ constexpr void bloom_filter<Key, Extent, Scope, Policy, Allocator>::clear_async(
  cuda::stream_ref stream)
{
  ref_.clear_async(stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
template <class InputIt>
__host__ constexpr void bloom_filter<Key, Extent, Scope, Policy, Allocator>::add(
  InputIt first, InputIt last, cuda::stream_ref stream)
{
  ref_.add(first, last, stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
template <class InputIt>
__host__ constexpr void bloom_filter<Key, Extent, Scope, Policy, Allocator>::add_async(
  InputIt first, InputIt last, cuda::stream_ref stream) noexcept
{
  ref_.add_async(first, last, stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
template <class InputIt, class StencilIt, class Predicate>
__host__ constexpr void bloom_filter<Key, Extent, Scope, Policy, Allocator>::add_if(
  InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream)
{
  ref_.add_if(first, last, stencil, pred, stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
template <class InputIt, class StencilIt, class Predicate>
__host__ constexpr void bloom_filter<Key, Extent, Scope, Policy, Allocator>::add_if_async(
  InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream) noexcept
{
  ref_.add_if_async(first, last, stencil, pred, stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
template <class InputIt, class OutputIt>
__host__ constexpr void bloom_filter<Key, Extent, Scope, Policy, Allocator>::contains(
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const
{
  ref_.contains(first, last, output_begin, stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
template <class InputIt, class OutputIt>
__host__ constexpr void bloom_filter<Key, Extent, Scope, Policy, Allocator>::contains_async(
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const noexcept
{
  ref_.contains_async(first, last, output_begin, stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
template <class InputIt, class StencilIt, class Predicate, class OutputIt>
__host__ constexpr void bloom_filter<Key, Extent, Scope, Policy, Allocator>::contains_if(
  InputIt first,
  InputIt last,
  StencilIt stencil,
  Predicate pred,
  OutputIt output_begin,
  cuda::stream_ref stream) const
{
  ref_.contains_if(first, last, stencil, pred, output_begin, stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
template <class InputIt, class StencilIt, class Predicate, class OutputIt>
__host__ constexpr void bloom_filter<Key, Extent, Scope, Policy, Allocator>::contains_if_async(
  InputIt first,
  InputIt last,
  StencilIt stencil,
  Predicate pred,
  OutputIt output_begin,
  cuda::stream_ref stream) const noexcept
{
  ref_.contains_if_async(first, last, stencil, pred, output_begin, stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
__host__ constexpr void bloom_filter<Key, Extent, Scope, Policy, Allocator>::merge(
  bloom_filter<Key, Extent, Scope, Policy, Allocator> const& other, cuda::stream_ref stream)
{
  ref_.merge(other.ref_, stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
__host__ constexpr void bloom_filter<Key, Extent, Scope, Policy, Allocator>::merge_async(
  bloom_filter<Key, Extent, Scope, Policy, Allocator> const& other, cuda::stream_ref stream)
{
  ref_.merge_async(other.ref_, stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
__host__ constexpr void bloom_filter<Key, Extent, Scope, Policy, Allocator>::intersect(
  bloom_filter<Key, Extent, Scope, Policy, Allocator> const& other, cuda::stream_ref stream)
{
  ref_.intersect(other.ref_, stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
__host__ constexpr void bloom_filter<Key, Extent, Scope, Policy, Allocator>::intersect_async(
  bloom_filter<Key, Extent, Scope, Policy, Allocator> const& other, cuda::stream_ref stream)
{
  ref_.intersect_async(other.ref_, stream);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
[[nodiscard]] __host__ constexpr
  typename bloom_filter<Key, Extent, Scope, Policy, Allocator>::word_type*
  bloom_filter<Key, Extent, Scope, Policy, Allocator>::data() noexcept
{
  return ref_.data();
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
[[nodiscard]] __host__ constexpr
  typename bloom_filter<Key, Extent, Scope, Policy, Allocator>::word_type const*
  bloom_filter<Key, Extent, Scope, Policy, Allocator>::data() const noexcept
{
  return ref_.data();
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
[[nodiscard]] __host__ constexpr
  typename bloom_filter<Key, Extent, Scope, Policy, Allocator>::extent_type
  bloom_filter<Key, Extent, Scope, Policy, Allocator>::block_extent() const noexcept
{
  return ref_.block_extent();
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
[[nodiscard]] __host__ constexpr
  typename bloom_filter<Key, Extent, Scope, Policy, Allocator>::allocator_type
  bloom_filter<Key, Extent, Scope, Policy, Allocator>::allocator() const noexcept
{
  return allocator_;
}

template <class Key, class Extent, cuda::thread_scope Scope, class Policy, class Allocator>
[[nodiscard]] __host__ constexpr
  typename bloom_filter<Key, Extent, Scope, Policy, Allocator>::template ref_type<>
  bloom_filter<Key, Extent, Scope, Policy, Allocator>::ref() const noexcept
{
  return ref_;
}

}  // namespace cuco
