/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
constexpr bucket_storage<T, BucketSize, Extent, Allocator>::bucket_storage(
  Extent size, Allocator const& allocator)
  : detail::bucket_storage_base<T, BucketSize, Extent>{size},
    allocator_{allocator},
    bucket_deleter_{capacity(), allocator_},
    buckets_{allocator_.allocate(capacity()), bucket_deleter_}
{
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
constexpr bucket_storage<T, BucketSize, Extent, Allocator>::bucket_type*
bucket_storage<T, BucketSize, Extent, Allocator>::data() const noexcept
{
  return buckets_.get();
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
constexpr bucket_storage<T, BucketSize, Extent, Allocator>::allocator_type
bucket_storage<T, BucketSize, Extent, Allocator>::allocator() const noexcept
{
  return allocator_;
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
constexpr bucket_storage<T, BucketSize, Extent, Allocator>::ref_type
bucket_storage<T, BucketSize, Extent, Allocator>::ref() const noexcept
{
  return ref_type{this->bucket_extent(), this->data()};
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
void bucket_storage<T, BucketSize, Extent, Allocator>::initialize(value_type key,
                                                                  cuda::stream_ref stream)
{
  this->initialize_async(key, stream);
  stream.wait();
}

template <typename T, int32_t BucketSize, typename Extent, typename Allocator>
void bucket_storage<T, BucketSize, Extent, Allocator>::initialize_async(
  value_type key, cuda::stream_ref stream) noexcept
{
  if (this->num_buckets() == 0) { return; }

  auto constexpr cg_size = 1;
  auto constexpr stride  = 4;
  auto const grid_size   = cuco::detail::grid_size(this->num_buckets(), cg_size, stride);

  detail::initialize<<<grid_size, cuco::detail::default_block_size(), 0, stream.get()>>>(
    this->data(), this->num_buckets(), key);
}

template <typename T, int32_t BucketSize, typename Extent>
__host__ __device__ constexpr bucket_storage_ref<T, BucketSize, Extent>::bucket_storage_ref(
  Extent size, bucket_type* buckets) noexcept
  : detail::bucket_storage_base<T, BucketSize, Extent>{size}, buckets_{buckets}
{
}

template <typename T, int32_t BucketSize, typename Extent>
struct bucket_storage_ref<T, BucketSize, Extent>::iterator {
 public:
  using iterator_category = std::input_iterator_tag;  ///< iterator category
  using reference         = value_type&;              ///< iterator reference type

  /**
   * @brief Constructs a device side input iterator of the given slot.
   *
   * @param current The slot pointer
   */
  __device__ constexpr explicit iterator(value_type* current) noexcept : current_{current} {}

  /**
   * @brief Prefix increment operator
   *
   * @throw This code path should never be chosen.
   *
   * @return Current iterator
   */
  __device__ constexpr iterator& operator++() noexcept
  {
    static_assert("Un-incrementable input iterator");
  }

  /**
   * @brief Postfix increment operator
   *
   * @throw This code path should never be chosen.
   *
   * @return Current iterator
   */
  __device__ constexpr iterator operator++(int32_t) noexcept
  {
    static_assert("Un-incrementable input iterator");
  }

  /**
   * @brief Dereference operator
   *
   * @return Reference to the current slot
   */
  __device__ constexpr reference operator*() const { return *current_; }

  /**
   * @brief Access operator
   *
   * @return Pointer to the current slot
   */
  __device__ constexpr value_type* operator->() const { return current_; }

  /**
   * Equality operator
   *
   * @return True if two iterators are identical
   */
  friend __device__ constexpr bool operator==(iterator const& lhs, iterator const& rhs) noexcept
  {
    return lhs.current_ == rhs.current_;
  }

  /**
   * Inequality operator
   *
   * @return True if two iterators are not identical
   */
  friend __device__ constexpr bool operator!=(iterator const& lhs, iterator const& rhs) noexcept
  {
    return not(lhs == rhs);
  }

 private:
  value_type* current_{};  ///< Pointer to the current slot
};

template <typename T, int32_t BucketSize, typename Extent>
__device__ constexpr bucket_storage_ref<T, BucketSize, Extent>::iterator
bucket_storage_ref<T, BucketSize, Extent>::end() noexcept
{
  return iterator{reinterpret_cast<value_type*>(this->data() + this->capacity())};
}

template <typename T, int32_t BucketSize, typename Extent>
__device__ constexpr bucket_storage_ref<T, BucketSize, Extent>::const_iterator
bucket_storage_ref<T, BucketSize, Extent>::end() const noexcept
{
  return const_iterator{reinterpret_cast<value_type*>(this->data() + this->capacity())};
}

template <typename T, int32_t BucketSize, typename Extent>
__device__ constexpr bucket_storage_ref<T, BucketSize, Extent>::bucket_type*
bucket_storage_ref<T, BucketSize, Extent>::data() noexcept
{
  return buckets_;
}

template <typename T, int32_t BucketSize, typename Extent>
__device__ constexpr bucket_storage_ref<T, BucketSize, Extent>::bucket_type*
bucket_storage_ref<T, BucketSize, Extent>::data() const noexcept
{
  return buckets_;
}

template <typename T, int32_t BucketSize, typename Extent>
__device__ constexpr bucket_storage_ref<T, BucketSize, Extent>::bucket_type
bucket_storage_ref<T, BucketSize, Extent>::operator[](size_type index) const noexcept
{
  return *reinterpret_cast<bucket_type*>(
    __builtin_assume_aligned(this->data() + index, sizeof(value_type) * bucket_size));
}

}  // namespace cuco
