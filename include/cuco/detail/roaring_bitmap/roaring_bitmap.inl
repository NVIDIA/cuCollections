/*
 * Copyright (c) 2025 NVIDIA CORPORATION.
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

#include <cuco/detail/error.hpp>
#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cuda/std/cstddef>
#include <cuda/std/span>
#include <cuda/stream_ref>

namespace cuco {

template <class T, cuda::thread_scope Scope, class Allocator>
__host__ roaring_bitmap<T, Scope, Allocator>::roaring_bitmap(cuda::std::byte const* bitmap,
                                                             cuda_thread_scope<Scope> scope,
                                                             Allocator const& alloc,
                                                             cuda::stream_ref stream)
  : allocator_{alloc},
    metadata_{ref_type<>::read_metadata(bitmap)},
    data_{
      allocator_.allocate(metadata_.size_bytes),
      detail::custom_deleter<cuda::std::size_t, allocator_type>{metadata_.size_bytes, allocator_}},
    ref_{data_.get(), metadata_, scope}
{
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    data_.get(), bitmap, metadata_.size_bytes, cudaMemcpyHostToDevice, stream.get()));
  // stream.wait();  // TODO check if this is necessary
}

template <class T, cuda::thread_scope Scope, class Allocator>
template <class InputIt, class OutputIt>
__host__ void roaring_bitmap<T, Scope, Allocator>::contains(InputIt first,
                                                            InputIt last,
                                                            OutputIt output,
                                                            cuda::stream_ref stream) const
{
  ref_.contains(first, last, output, stream);
}

template <class T, cuda::thread_scope Scope, class Allocator>
template <class InputIt, class OutputIt>
__host__ void roaring_bitmap<T, Scope, Allocator>::contains_async(
  InputIt first, InputIt last, OutputIt output, cuda::stream_ref stream) const noexcept
{
  ref_.contains_async(first, last, output, stream);
}

template <class T, cuda::thread_scope Scope, class Allocator>
__host__ cuda::std::size_t roaring_bitmap<T, Scope, Allocator>::size() const noexcept
{
  return ref_.size();
}

template <class T, cuda::thread_scope Scope, class Allocator>
__host__ cuda::std::span<cuda::std::byte const> roaring_bitmap<T, Scope, Allocator>::data()
  const noexcept
{
  return ref_.data();
}

template <class T, cuda::thread_scope Scope, class Allocator>
__host__ typename roaring_bitmap<T, Scope, Allocator>::allocator_type
roaring_bitmap<T, Scope, Allocator>::allocator() const noexcept
{
  return allocator_;
}

template <class T, cuda::thread_scope Scope, class Allocator>
__host__ typename roaring_bitmap<T, Scope, Allocator>::ref_type<>
roaring_bitmap<T, Scope, Allocator>::ref() const noexcept
{
  return ref_;
}
}  // namespace cuco