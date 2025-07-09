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
#include <cuco/detail/roaring_bitmap/util.cuh>
#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/utility/traits.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>

#include <memory>
#include <nv/target>

namespace cuco::detail {

template <class T>
struct roaring_bitmap_storage_ref {
  static_assert(cuco::dependent_false<T>, "T must be either uint32_t or uint64_t");
};

template <>
class roaring_bitmap_storage_ref<cuda::std::uint32_t> {
 public:
  using metadata_type = roaring_bitmap_metadata<cuda::std::uint32_t>;
  __host__ __device__ roaring_bitmap_storage_ref(cuda::std::byte const* bitmap,
                                                 metadata_type const& metadata)
    : data_{bitmap}, metadata_{metadata}
  {
  }

  __device__ roaring_bitmap_storage_ref(cuda::std::byte const* bitmap)
    : data_{bitmap}, metadata_{metadata_type{bitmap}}
  {
  }

  __host__ __device__ metadata_type const& metadata() const noexcept { return metadata_; }

  __host__ __device__ cuda::std::byte const* data() const noexcept { return data_; }

 private:
  cuda::std::byte const* data_;
  metadata_type metadata_;
};

template <class T, class Allocator>
struct roaring_bitmap_storage {
  static_assert(cuco::dependent_false<T>, "T must be either uint32_t or uint64_t");
};

template <class Allocator>
class roaring_bitmap_storage<cuda::std::uint32_t, Allocator> {
 public:
  using allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<cuda::std::byte>;
  using ref_type = roaring_bitmap_storage_ref<cuda::std::uint32_t>;

  roaring_bitmap_storage(roaring_bitmap_storage const& other)            = default;
  roaring_bitmap_storage(roaring_bitmap_storage&& other)                 = default;
  roaring_bitmap_storage& operator=(roaring_bitmap_storage const& other) = default;
  roaring_bitmap_storage& operator=(roaring_bitmap_storage&& other)      = default;

  ~roaring_bitmap_storage() = default;

  roaring_bitmap_storage(cuda::std::byte const* bitmap,
                         Allocator const& alloc,
                         cuda::stream_ref stream)
    : allocator_{alloc},
      metadata_{bitmap},
      data_{allocator_.allocate(metadata_.size_bytes),
            detail::custom_deleter<cuda::std::size_t, allocator_type>{metadata_.size_bytes,
                                                                      allocator_}},
      ref_{data_.get(), metadata_}
  {
    CUCO_CUDA_TRY(cudaMemcpyAsync(
      data_.get(), bitmap, metadata_.size_bytes, cudaMemcpyHostToDevice, stream.get()));
    // stream.wait();  // TODO check if this is necessary
  }

  ref_type ref() const noexcept { return ref_; }

 private:
  allocator_type allocator_;
  typename ref_type::metadata_type metadata_;
  std::unique_ptr<cuda::std::byte, custom_deleter<cuda::std::size_t, allocator_type>> data_;
  ref_type ref_;
};

// TODO implement roaring_bitmap_metadata<cuda::std::uint64_t>

}  // namespace cuco::detail