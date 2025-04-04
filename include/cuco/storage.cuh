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

#include <cuco/detail/storage/storage.cuh>

namespace cuco {

/**
 * @brief Public storage class.
 *
 * @note This is a public interface used to control storage bucket size. A bucket consists of one
 * or multiple contiguous slots. The bucket size defines the workload granularity for each CUDA
 * thread, i.e., how many slots a thread would concurrently operate on when performing modify or
 * lookup operations. cuCollections uses the array of bucket storage to supersede the raw flat slot
 * storage due to its superior granularity control: When bucket size equals one, array of buckets
 * performs the same as the flat storage. If the underlying operation is more memory bandwidth
 * bound, e.g., high occupancy multimap operations, a larger bucket size can reduce the length of
 * probing sequences thus improve runtime performance.
 *
 * @tparam BucketSize Number of elements per bucket storage
 */
template <int32_t BucketSize>
class storage {
 public:
  /// Number of slots per bucket storage
  static constexpr int32_t bucket_size = BucketSize;

  /// Type of implementation details
  template <class T, class Extent, class Allocator>
  using impl = bucket_storage<T, bucket_size, Extent, Allocator>;
};

/**
 * @brief Trait to determine if a storage is bucket-based
 *
 * @tparam T Storage class
 */
template <typename T>
struct is_bucket_storage : cuda::std::false_type {};

/**
 * @brief Specialization for bucket_storage
 */
template <typename T, int BucketSize, typename Extent, typename Allocator>
struct is_bucket_storage<cuco::bucket_storage<T, BucketSize, Extent, Allocator>>
  : cuda::std::true_type {};

/**
 * @brief Specialization for bucket_storage_ref
 */
template <typename T, int BucketSize, typename Extent>
struct is_bucket_storage<cuco::bucket_storage_ref<T, BucketSize, Extent>> : cuda::std::true_type {};

/**
 * @brief Specialization for bucket_storage_ref
 */
template <int BucketSize>
struct is_bucket_storage<cuco::storage<BucketSize>> : cuda::std::true_type {};

/**
 * @brief Helper variable template for is_bucket_storage
 */
template <typename T>
constexpr bool is_bucket_storage_v = is_bucket_storage<T>::value;

}  // namespace cuco
