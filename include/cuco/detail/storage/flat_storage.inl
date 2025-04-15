/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cuco/detail/storage/kernels.cuh>
#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/extent.cuh>

#include <cuda/std/array>
#include <cuda/stream_ref>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>

namespace cuco {

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
constexpr flat_storage<T, BucketSize, Extent, Allocator>::flat_storage(Extent size,
                                                                       Allocator const& allocator)
  : extent_{size},
    allocator_{allocator},
    slot_deleter_{capacity(), allocator_},
    slots_{allocator_.allocate(capacity()), slot_deleter_}
{
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
constexpr flat_storage<T, BucketSize, Extent, Allocator>::value_type*
flat_storage<T, BucketSize, Extent, Allocator>::data() const noexcept
{
  return slots_.get();
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
constexpr flat_storage<T, BucketSize, Extent, Allocator>::allocator_type
flat_storage<T, BucketSize, Extent, Allocator>::allocator() const noexcept
{
  return allocator_;
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
constexpr flat_storage<T, BucketSize, Extent, Allocator>::ref_type
flat_storage<T, BucketSize, Extent, Allocator>::ref() const noexcept
{
  return ref_type{this->extent(), this->data()};
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
void flat_storage<T, BucketSize, Extent, Allocator>::initialize(value_type key,
                                                                cuda::stream_ref stream)
{
  this->initialize_async(key, stream);
  stream.wait();
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
void flat_storage<T, BucketSize, Extent, Allocator>::initialize_async(
  value_type key, cuda::stream_ref stream) noexcept
{
  if (this->capacity() == 0) { return; }

  auto constexpr cg_size = 1;
  auto constexpr stride  = 4;
  auto const grid_size   = cuco::detail::grid_size(this->capacity(), cg_size, stride);

  detail::initialize<<<grid_size, cuco::detail::default_block_size(), 0, stream.get()>>>(
    this->data(), this->capacity(), key);
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
__host__ __device__ constexpr typename flat_storage<T, BucketSize, Extent, Allocator>::size_type
flat_storage<T, BucketSize, Extent, Allocator>::num_buckets() const noexcept
{
  return static_cast<size_type>(extent_) / bucket_size;
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
__host__ __device__ constexpr typename flat_storage<T, BucketSize, Extent, Allocator>::size_type
flat_storage<T, BucketSize, Extent, Allocator>::capacity() const noexcept
{
  return static_cast<size_type>(extent_);
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
__host__ __device__ constexpr typename flat_storage<T, BucketSize, Extent, Allocator>::extent_type
flat_storage<T, BucketSize, Extent, Allocator>::extent() const noexcept
{
  return extent_;
}

template <typename T, int32_t BucketSize, typename Extent>
__host__ __device__ constexpr flat_storage_ref<T, BucketSize, Extent>::flat_storage_ref(
  Extent size, value_type* slots) noexcept
  : extent_{size}, slots_{slots}
{
}

template <typename T, int32_t BucketSize, typename Extent>
__device__ constexpr flat_storage_ref<T, BucketSize, Extent>::iterator
flat_storage_ref<T, BucketSize, Extent>::end() noexcept
{
  return iterator{reinterpret_cast<value_type*>(this->data() + this->capacity())};
}

template <typename T, int32_t BucketSize, typename Extent>
__device__ constexpr flat_storage_ref<T, BucketSize, Extent>::iterator
flat_storage_ref<T, BucketSize, Extent>::end() const noexcept
{
  return iterator{reinterpret_cast<value_type*>(this->data() + this->capacity())};
}

template <typename T, int32_t BucketSize, typename Extent>
__device__ constexpr flat_storage_ref<T, BucketSize, Extent>::value_type*
flat_storage_ref<T, BucketSize, Extent>::data() noexcept
{
  return slots_;
}

template <typename T, int32_t BucketSize, typename Extent>
__device__ constexpr flat_storage_ref<T, BucketSize, Extent>::value_type*
flat_storage_ref<T, BucketSize, Extent>::data() const noexcept
{
  return slots_;
}

template <typename T, int32_t BucketSize, typename Extent>
__device__ constexpr flat_storage_ref<T, BucketSize, Extent>::bucket_type
flat_storage_ref<T, BucketSize, Extent>::operator[](size_type index) const noexcept
{
  return *reinterpret_cast<bucket_type*>(this->data() + index);
  /*
    bucket_type res;
    memcpy(res.data(), this->data() + index, sizeof(value_type) * bucket_size);
    return res;
    */
}

template <typename T, int32_t BucketSize, typename Extent>
__host__ __device__ constexpr typename flat_storage_ref<T, BucketSize, Extent>::size_type
flat_storage_ref<T, BucketSize, Extent>::num_buckets() const noexcept
{
  return static_cast<size_type>(extent_) / bucket_size;
}

template <typename T, int32_t BucketSize, typename Extent>
__host__ __device__ constexpr typename flat_storage_ref<T, BucketSize, Extent>::size_type
flat_storage_ref<T, BucketSize, Extent>::capacity() const noexcept
{
  return static_cast<size_type>(extent_);
}

template <typename T, int32_t BucketSize, typename Extent>
__host__ __device__ constexpr typename flat_storage_ref<T, BucketSize, Extent>::extent_type
flat_storage_ref<T, BucketSize, Extent>::extent() const noexcept
{
  return extent_;
}

}  // namespace cuco
