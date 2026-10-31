/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/pair/helpers.cuh>
#include <cuco/detail/storage/functors.cuh>
#include <cuco/detail/utility/assert.cuh>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/extent.cuh>

#include <cub/device/device_for.cuh>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/stream>
#include <cuda/utility>

#include <cassert>
#include <memory>

namespace cuco::detail {
/**
 * @brief Forces `word` to be materialized in registers at this point without emitting code.
 *
 * @tparam Word Unsigned integral type of 1, 2, 4, 8 or 16 bytes
 *
 * @param word The value to keep
 */
template <typename Word>
__device__ __forceinline__ void keep_in_register(Word word) noexcept
{
  if constexpr (sizeof(Word) <= 4) {
    asm volatile("" ::"r"(static_cast<cuda::std::uint32_t>(word)));
  } else if constexpr (sizeof(Word) == 8) {
    asm volatile("" ::"l"(static_cast<cuda::std::uint64_t>(word)));
  } else {
    static_assert(sizeof(Word) == 16);
    asm volatile("" ::"l"(static_cast<cuda::std::uint64_t>(word)),
                 "l"(static_cast<cuda::std::uint64_t>(word >> 64)));
  }
}

/**
 * @brief Whether a bucket of `T` aligned to `Alignment` bytes can be loaded as one word per slot.
 *
 * @tparam T Slot type
 * @tparam Alignment Bucket alignment in bytes
 *
 * @return `true` if `T` has a packed word type of the same size whose alignment `Alignment` covers
 */
template <typename T, cuda::std::size_t Alignment>
__host__ __device__ constexpr bool loads_as_slot_words() noexcept
{
  using word_type = cuco::detail::packed_t<T>;
  if constexpr (cuda::std::is_void_v<word_type> or not cuda::std::is_trivially_copyable_v<T>) {
    return false;
  } else {
    return Alignment >= alignof(word_type);
  }
}
}  // namespace cuco::detail

namespace cuco {

template <typename T, int BucketSize, typename Extent>
__host__ __device__ constexpr bucket_storage_ref<T, BucketSize, Extent>::bucket_storage_ref(
  Extent size, value_type* slots) noexcept
  : extent_{size}, slots_{slots}
{
  static_assert(cuda::std::has_single_bit(alignment), "Alignment must be a power of 2");
  assert(reinterpret_cast<cuda::std::uintptr_t>(slots) % alignment == 0 &&
         "Storage must be properly aligned");
}

template <typename T, int BucketSize, typename Extent>
__host__ __device__ constexpr bucket_storage_ref<T, BucketSize, Extent>::iterator
bucket_storage_ref<T, BucketSize, Extent>::end() noexcept
{
  return iterator{reinterpret_cast<value_type*>(this->data() + this->capacity())};
}

template <typename T, int BucketSize, typename Extent>
__host__ __device__ constexpr bucket_storage_ref<T, BucketSize, Extent>::iterator
bucket_storage_ref<T, BucketSize, Extent>::end() const noexcept
{
  return iterator{reinterpret_cast<value_type*>(this->data() + this->capacity())};
}

template <typename T, int BucketSize, typename Extent>
__host__ __device__ constexpr bucket_storage_ref<T, BucketSize, Extent>::value_type*
bucket_storage_ref<T, BucketSize, Extent>::data() noexcept
{
  return slots_;
}

template <typename T, int BucketSize, typename Extent>
__host__ __device__ constexpr bucket_storage_ref<T, BucketSize, Extent>::value_type*
bucket_storage_ref<T, BucketSize, Extent>::data() const noexcept
{
  return slots_;
}

template <typename T, int BucketSize, typename Extent>
__device__ constexpr bucket_storage_ref<T, BucketSize, Extent>::bucket_type
bucket_storage_ref<T, BucketSize, Extent>::operator[](size_type index) const noexcept
{
#if defined(CUCO_DEBUG)
  if constexpr (cuda::std::is_signed_v<size_type>) {
    CUCO_DEBUG_ASSERT(index >= size_type{0}, "Bucket access precedes storage");
  }
  CUCO_DEBUG_ASSERT(index <= this->capacity() && bucket_size <= this->capacity() - index,
                    "Bucket access exceeds storage capacity");
#endif
  return *reinterpret_cast<bucket_type*>(this->data() + index);
}

template <typename T, int BucketSize, typename Extent>
__device__ constexpr bucket_storage_ref<T, BucketSize, Extent>::bucket_type
bucket_storage_ref<T, BucketSize, Extent>::load_bucket(size_type index) const noexcept
{
  CUCO_DEBUG_ASSERT(index % bucket_size == 0, "Bucket index must be a multiple of bucket_size");
  CUCO_DEBUG_ASSERT(index <= this->capacity() && bucket_size <= this->capacity() - index,
                    "Bucket access exceeds storage capacity");
  if constexpr (cuco::detail::loads_as_slot_words<T, alignment>()) {
    using word_type = cuco::detail::packed_t<T>;
    // Load each slot as one word and keep every word in a register here. A field-wise copy lets the
    // compiler split the bucket load and sink slots (or slot fields) that only a branch reads (a
    // later key compare, the payload expected by an insert CAS) into that branch, where they are
    // loaded from memory a second time.
    struct alignas(alignment) words_type {
      word_type slots[bucket_size];
    };
    static_assert(sizeof(words_type) == sizeof(bucket_type));
    auto const words = *reinterpret_cast<words_type const*>(this->data() + index);
    cuda::static_for<bucket_size>(
      [&] __device__(auto i) { cuco::detail::keep_in_register(words.slots[i()]); });
    return cuda::std::bit_cast<bucket_type>(words);
  } else if constexpr (alignment <= alignof(bucket_type)) {
    return (*this)[index];
  } else {
    return *reinterpret_cast<bucket_type*>(
      __builtin_assume_aligned(this->data() + index, alignment));
  }
}

template <typename T, int BucketSize, typename Extent>
__host__ __device__ constexpr typename bucket_storage_ref<T, BucketSize, Extent>::size_type
bucket_storage_ref<T, BucketSize, Extent>::num_buckets() const noexcept
{
  return static_cast<size_type>(extent_) / bucket_size;
}

template <typename T, int BucketSize, typename Extent>
__host__ __device__ constexpr typename bucket_storage_ref<T, BucketSize, Extent>::size_type
bucket_storage_ref<T, BucketSize, Extent>::capacity() const noexcept
{
  return static_cast<size_type>(extent_);
}

template <typename T, int BucketSize, typename Extent>
__host__ __device__ constexpr typename bucket_storage_ref<T, BucketSize, Extent>::extent_type
bucket_storage_ref<T, BucketSize, Extent>::extent() const noexcept
{
  return extent_;
}

template <typename T, int BucketSize, typename Extent, typename Allocator>
constexpr bucket_storage<T, BucketSize, Extent, Allocator>::bucket_storage(
  Extent size, Allocator const& allocator, cuda::stream_ref stream)
  : extent_{size}, allocator_{allocator}, slots_{[this, &stream]() {
      constexpr std::size_t align  = ref_type::alignment;
      constexpr std::size_t extra  = (align - 1) / sizeof(value_type) + 1;
      std::size_t const alloc_size = static_cast<std::size_t>(capacity()) + extra;
      auto* const raw_ptr          = allocator_.allocate(alloc_size, stream);
      auto* const aligned_ptr      = reinterpret_cast<value_type*>(
        (reinterpret_cast<cuda::std::uintptr_t>(raw_ptr) + align - 1) & ~(align - 1));
      return std::unique_ptr<value_type, aligned_deleter>{
        aligned_ptr, aligned_deleter{raw_ptr, alloc_size, allocator_, stream}};
    }()}
{
}

template <typename T, int BucketSize, typename Extent, typename Allocator>
constexpr bucket_storage<T, BucketSize, Extent, Allocator>::value_type*
bucket_storage<T, BucketSize, Extent, Allocator>::data() const noexcept
{
  return slots_.get();
}

template <typename T, int BucketSize, typename Extent, typename Allocator>
constexpr bucket_storage<T, BucketSize, Extent, Allocator>::allocator_type
bucket_storage<T, BucketSize, Extent, Allocator>::allocator() const noexcept
{
  return allocator_;
}

template <typename T, int BucketSize, typename Extent, typename Allocator>
constexpr bucket_storage<T, BucketSize, Extent, Allocator>::ref_type
bucket_storage<T, BucketSize, Extent, Allocator>::ref() const noexcept
{
  return ref_type{this->extent(), this->data()};
}

template <typename T, int BucketSize, typename Extent, typename Allocator>
void bucket_storage<T, BucketSize, Extent, Allocator>::initialize(value_type key,
                                                                  cuda::stream_ref stream)
{
  this->initialize_async(key, stream);
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
  stream.sync();
#else
  stream.wait();
#endif
}

template <typename T, int BucketSize, typename Extent, typename Allocator>
void bucket_storage<T, BucketSize, Extent, Allocator>::initialize_async(value_type key,
                                                                        cuda::stream_ref stream)
{
  if (this->capacity() == 0) { return; }

  auto ftor = cuco::detail::initialize_functor<size_type, T>{this->data(), key};
  CUCO_CUDA_TRY(cub::DeviceFor::Bulk(this->capacity(), ftor, stream.get()));
}

template <typename T, int BucketSize, typename Extent, typename Allocator>
__host__ __device__ constexpr typename bucket_storage<T, BucketSize, Extent, Allocator>::size_type
bucket_storage<T, BucketSize, Extent, Allocator>::num_buckets() const noexcept
{
  return static_cast<size_type>(extent_) / bucket_size;
}

template <typename T, int BucketSize, typename Extent, typename Allocator>
__host__ __device__ constexpr typename bucket_storage<T, BucketSize, Extent, Allocator>::size_type
bucket_storage<T, BucketSize, Extent, Allocator>::capacity() const noexcept
{
  return static_cast<size_type>(extent_);
}

template <typename T, int BucketSize, typename Extent, typename Allocator>
__host__ __device__ constexpr typename bucket_storage<T, BucketSize, Extent, Allocator>::extent_type
bucket_storage<T, BucketSize, Extent, Allocator>::extent() const noexcept
{
  return extent_;
}

}  // namespace cuco
