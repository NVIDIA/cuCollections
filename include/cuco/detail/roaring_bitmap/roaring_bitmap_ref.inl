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

#include <cuco/utility/cuda_thread_scope.cuh>

#include <cuda/std/cstddef>
#include <cuda/std/span>
#include <cuda/stream_ref>

namespace cuco {

template <class T, cuda::thread_scope Scope>
__host__ roaring_bitmap_ref<T, Scope>::roaring_bitmap_ref(
  cuda::std::span<cuda::std::byte const> compressed_bitmap_h,
  cuda::std::span<cuda::std::byte const> compressed_bitmap_d,
  cuda_thread_scope<Scope> scope)
  : impl_{compressed_bitmap_h, compressed_bitmap_d, scope}
{
}

template <class T, cuda::thread_scope Scope>
__device__ roaring_bitmap_ref<T, Scope>::roaring_bitmap_ref(
  cuda::std::span<cuda::std::byte const> compressed_bitmap, cuda_thread_scope<Scope> scope)
  : impl_{compressed_bitmap, scope}
{
}

template <class T, cuda::thread_scope Scope>
template <class InputIt, class OutputIt>
__host__ void roaring_bitmap_ref<T, Scope>::contains(InputIt first,
                                                     InputIt last,
                                                     OutputIt output,
                                                     cuda::stream_ref stream) const
{
  impl_.contains(first, last, output, stream);
}

template <class T, cuda::thread_scope Scope>
template <class InputIt, class OutputIt>
__host__ void roaring_bitmap_ref<T, Scope>::contains_async(InputIt first,
                                                           InputIt last,
                                                           OutputIt output,
                                                           cuda::stream_ref stream) const noexcept
{
  impl_.contains_async(first, last, output, stream);
}

template <class T, cuda::thread_scope Scope>
__device__ bool roaring_bitmap_ref<T, Scope>::contains(T value) const
{
  return impl_.contains(value);
}

template <class T, cuda::thread_scope Scope>
__host__ __device__ cuda::std::size_t roaring_bitmap_ref<T, Scope>::size() const noexcept
{
  return impl_.size();
}

template <class T, cuda::thread_scope Scope>
__host__ __device__ cuda::std::span<cuda::std::byte const> roaring_bitmap_ref<T, Scope>::data()
  const noexcept
{
  return impl_.data();
}
}  // namespace cuco