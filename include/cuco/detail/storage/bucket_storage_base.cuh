/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuda/std/array>

#include <cstddef>
#include <cstdint>

namespace cuco {
namespace detail {
/**
￼ * @brief Bucket data structure type
￼ *
￼ * @tparam T Bucket slot type
￼ * @tparam BucketSize Number of elements per bucket
￼ */
template <typename T, int32_t BucketSize>
struct bucket : public cuda::std::array<T, BucketSize> {
 public:
  static int32_t constexpr bucket_size = BucketSize;  ///< Number of slots per bucket
};

/**
 * @brief Base class of array of slot buckets open addressing storage.
 *
 * @note This should NOT be used directly.
 *
 * @tparam T Slot type
 * @tparam BucketSize Number of slots in each bucket
 * @tparam Extent Type of extent denoting the number of buckets
 */
template <typename T, int32_t BucketSize, typename Extent>
class bucket_storage_base : public storage_base<Extent> {
 public:
  /**
   * @brief The number of elements (slots) processed per bucket.
   */
  static constexpr int32_t bucket_size = BucketSize;

  using extent_type = typename storage_base<Extent>::extent_type;  ///< Storage extent type
  using size_type   = typename storage_base<Extent>::size_type;    ///< Storage size type

  using value_type  = T;                                ///< Slot type
  using bucket_type = bucket<value_type, bucket_size>;  ///< Slot bucket type

  /**
   * @brief Constructor of array of bucket base storage.
   *
   * @param size Number of buckets to store
   */
  __host__ __device__ explicit constexpr bucket_storage_base(Extent size)
    : storage_base<Extent>{size}
  {
  }

  /**
   * @brief Gets the total number of slot buckets in the current storage.
   *
   * @return The total number of slot buckets
   */
  [[nodiscard]] __host__ __device__ constexpr size_type num_buckets() const noexcept
  {
    return storage_base<Extent>::capacity();
  }

  /**
   * @brief Gets the total number of slots in the current storage.
   *
   * @return The total number of slots
   */
  [[nodiscard]] __host__ __device__ constexpr size_type capacity() const noexcept
  {
    return storage_base<Extent>::capacity() * bucket_size;
  }

  /**
   * @brief Gets the bucket extent of the current storage.
   *
   * @return The bucket extent.
   */
  [[nodiscard]] __host__ __device__ constexpr extent_type bucket_extent() const noexcept
  {
    return storage_base<Extent>::extent();
  }
};

}  // namespace detail
}  // namespace cuco
