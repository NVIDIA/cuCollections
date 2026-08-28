/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/stream_ref>

namespace cuco::experimental {

template <class T, class Allocator>
roaring_bitmap<T, Allocator>::roaring_bitmap(cuda::std::byte const* bitmap,
                                             Allocator const& alloc,
                                             cuda::stream_ref stream)
  : storage_{bitmap, alloc, stream}
{
}

template <class T, class Allocator>
roaring_bitmap<T, Allocator> roaring_bitmap<T, Allocator>::from_serialized(
  cuda::std::byte const* bitmap, Allocator const& alloc, cuda::stream_ref stream)
{
  static_assert(cuda::std::is_same_v<T, cuda::std::uint32_t>,
                "roaring_bitmap::from_serialized currently supports only uint32_t");
  return roaring_bitmap{bitmap, alloc, stream};
}

template <class T, class Allocator>
template <class InputIt, class OutputIt>
void roaring_bitmap<T, Allocator>::contains(InputIt first,
                                            InputIt last,
                                            OutputIt output,
                                            cuda::stream_ref stream) const
{
  ref_type{storage_.ref()}.contains(first, last, output, stream);
}

template <class T, class Allocator>
template <class InputIt, class OutputIt>
void roaring_bitmap<T, Allocator>::contains_async(InputIt first,
                                                  InputIt last,
                                                  OutputIt output,
                                                  cuda::stream_ref stream) const noexcept
{
  ref_type{storage_.ref()}.contains_async(first, last, output, stream);
}

template <class T, class Allocator>
cuda::std::size_t roaring_bitmap<T, Allocator>::size() const noexcept
{
  return ref_type{storage_.ref()}.size();
}

template <class T, class Allocator>
bool roaring_bitmap<T, Allocator>::empty() const noexcept
{
  return ref_type{storage_.ref()}.empty();
}

template <class T, class Allocator>
cuda::std::byte const* roaring_bitmap<T, Allocator>::data() const noexcept
{
  return ref_type{storage_.ref()}.data();
}

template <class T, class Allocator>
cuda::std::size_t roaring_bitmap<T, Allocator>::size_bytes() const noexcept
{
  return ref_type{storage_.ref()}.size_bytes();
}

template <class T, class Allocator>
typename roaring_bitmap<T, Allocator>::allocator_type roaring_bitmap<T, Allocator>::allocator()
  const noexcept
{
  return storage_.allocator();
}

template <class T, class Allocator>
typename roaring_bitmap<T, Allocator>::ref_type roaring_bitmap<T, Allocator>::ref() const noexcept
{
  return ref_type{storage_.ref()};
}
}  // namespace cuco::experimental