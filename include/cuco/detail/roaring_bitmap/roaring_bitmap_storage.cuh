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
#include <utility>
#include <vector>

namespace cuco::experimental::detail {

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
    : metadata_{metadata},
      data_{bitmap},
      run_container_bitmap_{bitmap + metadata.run_container_bitmap},
      key_cards_{bitmap + metadata.key_cards},
      container_offsets_{bitmap + metadata.container_offsets}
  {
    assert(metadata.valid);
  }

  __device__ roaring_bitmap_storage_ref(cuda::std::byte const* bitmap)
    : roaring_bitmap_storage_ref{bitmap, metadata_type{bitmap}}
  {
  }

  __host__ __device__ metadata_type const& metadata() const noexcept { return metadata_; }

  __host__ __device__ cuda::std::byte const* data() const noexcept { return data_; }

  __host__ __device__ cuda::std::size_t size_bytes() const noexcept { return metadata_.size_bytes; }

  __host__ __device__ cuda::std::byte const* run_container_bitmap() const noexcept
  {
    return run_container_bitmap_;
  }

  __host__ __device__ cuda::std::byte const* key_cards() const noexcept { return key_cards_; }

  __host__ __device__ cuda::std::byte const* container_offsets() const noexcept
  {
    return container_offsets_;
  }

 private:
  metadata_type metadata_;
  cuda::std::byte const* data_;
  cuda::std::byte const* run_container_bitmap_;
  cuda::std::byte const* key_cards_;
  cuda::std::byte const* container_offsets_;
};

template <>
class roaring_bitmap_storage_ref<cuda::std::uint64_t> {
 public:
  using metadata_type = roaring_bitmap_metadata<cuda::std::uint64_t>;

  __host__ __device__ roaring_bitmap_storage_ref(
    cuda::std::byte const* bitmap,
    metadata_type const& metadata,
    cuda::std::pair<cuda::std::uint32_t, roaring_bitmap_storage_ref<cuda::std::uint32_t>>* buckets)
    : metadata_{metadata}, data_{bitmap}, buckets_{buckets}
  {
  }

  __host__ __device__ metadata_type const& metadata() const noexcept { return metadata_; }

  __host__ __device__ cuda::std::byte const* data() const noexcept { return data_; }

  __host__ __device__ cuda::std::size_t size_bytes() const noexcept { return metadata_.size_bytes; }

  __host__ __device__
    cuda::std::pair<cuda::std::uint32_t, roaring_bitmap_storage_ref<cuda::std::uint32_t>>*
    buckets() const noexcept
  {
    return buckets_;
  }

 private:
  metadata_type metadata_;
  cuda::std::byte const* data_;
  cuda::std::pair<cuda::std::uint32_t, roaring_bitmap_storage_ref<cuda::std::uint32_t>>* buckets_;
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
            cuco::detail::custom_deleter<cuda::std::size_t, allocator_type>{metadata_.size_bytes,
                                                                            allocator_}},
      ref_{data_.get(), metadata_}
  {
    CUCO_CUDA_TRY(cudaMemcpyAsync(
      data_.get(), bitmap, metadata_.size_bytes, cudaMemcpyHostToDevice, stream.get()));
  }

  ref_type ref() const noexcept { return ref_; }

 private:
  allocator_type allocator_;
  typename ref_type::metadata_type metadata_;
  std::unique_ptr<cuda::std::byte, cuco::detail::custom_deleter<cuda::std::size_t, allocator_type>>
    data_;
  ref_type ref_;
};

template <class Allocator>
class roaring_bitmap_storage<cuda::std::uint64_t, Allocator> {
 public:
  using allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<cuda::std::byte>;
  using ref_type              = roaring_bitmap_storage_ref<cuda::std::uint64_t>;
  using bucket_ref_type       = roaring_bitmap_storage_ref<cuda::std::uint32_t>;
  using bucket_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<
    cuda::std::pair<cuda::std::uint32_t, bucket_ref_type>>;

  roaring_bitmap_storage(roaring_bitmap_storage const& other)            = default;
  roaring_bitmap_storage(roaring_bitmap_storage&& other)                 = default;
  roaring_bitmap_storage& operator=(roaring_bitmap_storage const& other) = default;
  roaring_bitmap_storage& operator=(roaring_bitmap_storage&& other)      = default;

  ~roaring_bitmap_storage() = default;

  roaring_bitmap_storage(cuda::std::byte const* bitmap,
                         Allocator const& alloc,
                         cuda::stream_ref stream)
    : allocator_{alloc},
      bucket_allocator_{alloc},
      bucket_metadata_{},
      buckets_h_{},
      metadata_{
        [bitmap](std::vector<typename ref_type::metadata_type::bucket_metadata>& bucket_metadata) {
          return typename ref_type::metadata_type{bitmap, bucket_metadata};
        }(bucket_metadata_)},
      data_{allocator_.allocate(metadata_.size_bytes),
            cuco::detail::custom_deleter<cuda::std::size_t, allocator_type>{metadata_.size_bytes,
                                                                            allocator_}},
      buckets_{bucket_allocator_.allocate(metadata_.num_buckets),
               cuco::detail::custom_deleter<cuda::std::size_t, bucket_allocator_type>{
                 metadata_.num_buckets, bucket_allocator_}},
      ref_{data_.get(), metadata_, buckets_.get()}
  {
    assert(metadata_.valid);
    buckets_h_.reserve(bucket_metadata_.size());
    for (auto const& meta : bucket_metadata_) {
      buckets_h_.emplace_back(meta.key,
                              bucket_ref_type{data_.get() + meta.byte_offset, meta.metadata});
    }
    CUCO_CUDA_TRY(cudaMemcpyAsync(
      data_.get(), bitmap, metadata_.size_bytes, cudaMemcpyHostToDevice, stream.get()));
    CUCO_CUDA_TRY(cudaMemcpyAsync(
      buckets_.get(),
      buckets_h_.data(),
      metadata_.num_buckets * sizeof(cuda::std::pair<cuda::std::uint32_t, bucket_ref_type>),
      cudaMemcpyHostToDevice,
      stream.get()));
  }

  ref_type ref() const noexcept { return ref_; }

 private:
  allocator_type allocator_;
  bucket_allocator_type bucket_allocator_;
  std::vector<typename ref_type::metadata_type::bucket_metadata> bucket_metadata_;
  std::vector<cuda::std::pair<cuda::std::uint32_t, bucket_ref_type>> buckets_h_;
  typename ref_type::metadata_type metadata_;
  std::unique_ptr<cuda::std::byte, cuco::detail::custom_deleter<cuda::std::size_t, allocator_type>>
    data_;
  std::unique_ptr<cuda::std::pair<cuda::std::uint32_t, bucket_ref_type>,
                  cuco::detail::custom_deleter<cuda::std::size_t, bucket_allocator_type>>
    buckets_;
  ref_type ref_;
};

}  // namespace cuco::experimental::detail