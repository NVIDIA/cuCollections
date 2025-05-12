/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cuco/detail/storage/bucket_storage_base.cuh>
#include <cuco/extent.cuh>
#include <cuco/utility/allocator.hpp>

#include <cuda/std/array>
#include <cuda/stream_ref>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>

namespace cuco {
/// Bucket type alias
template <typename T, int32_t BucketSize>
using bucket = detail::bucket<T, BucketSize>;

/**
 * @brief Non-owning array of buckets storage reference type.
 *
 * @tparam T Storage element type
 * @tparam BucketSize Number of slots in each bucket
 * @tparam Extent Type of extent denoting storage capacity
 */
template <typename T, int32_t BucketSize, typename Extent = cuco::extent<std::size_t>>
class bucket_storage_ref : public detail::bucket_storage_base<T, BucketSize, Extent> {
 public:
  /// Array of buckets base class type
  using base_type = detail::bucket_storage_base<T, BucketSize, Extent>;

  using base_type::bucket_size;  ///< Number of elements processed per bucket

  using extent_type = typename base_type::extent_type;  ///< Storage extent type
  using size_type   = typename base_type::size_type;    ///< Storage size type
  using value_type  = typename base_type::value_type;   ///< Slot type
  using bucket_type = typename base_type::bucket_type;  ///< Slot bucket type

  using base_type::capacity;
  using base_type::num_buckets;

  /**
   * @brief Constructor of AoS storage ref.
   *
   * @param size Number of buckets
   * @param buckets Pointer to the buckets array
   */
  __host__ __device__ explicit constexpr bucket_storage_ref(Extent size,
                                                            bucket_type* buckets) noexcept;

  /**
   * @brief Custom un-incrementable input iterator for the convenience of `find` operations.
   *
   * @note This iterator is for read only and NOT incrementable.
   */
  struct iterator;
  using const_iterator = iterator const;  ///< Const forward iterator type

  /**
   * @brief Returns an iterator to one past the last slot.
   *
   * This is provided for convenience for those familiar with checking
   * an iterator returned from `find()` against the `end()` iterator.
   *
   * @return An iterator to one past the last slot
   */
  [[nodiscard]] __device__ constexpr iterator end() noexcept;

  /**
   * @brief Returns a const_iterator to one past the last slot.
   *
   * This is provided for convenience for those familiar with checking
   * an iterator returned from `find()` against the `end()` iterator.
   *
   * @return A const_iterator to one past the last slot
   */
  [[nodiscard]] __device__ constexpr const_iterator end() const noexcept;

  /**
   * @brief Gets buckets array.
   *
   * @return Pointer to the first bucket
   */
  [[nodiscard]] __device__ constexpr bucket_type* data() noexcept;

  /**
   * @brief Gets bucket array.
   *
   * @return Pointer to the first bucket
   */
  [[nodiscard]] __device__ constexpr bucket_type* data() const noexcept;

  /**
   * @brief Returns an array of slots (or a bucket) for a given index.
   *
   * @param index Index of the bucket
   * @return An array of slots
   */
  [[nodiscard]] __device__ constexpr bucket_type operator[](size_type index) const noexcept;

 private:
  bucket_type* buckets_;  ///< Pointer to the buckets array
};

/**
 * @brief Array of buckets open addressing storage class.
 *
 * @tparam T Slot type
 * @tparam BucketSize Number of slots in each bucket
 * @tparam Extent Type of extent denoting number of buckets
 * @tparam Allocator Type of allocator used for device storage (de)allocation
 */
template <typename T,
          int32_t BucketSize,
          typename Extent    = cuco::extent<std::size_t>,
          typename Allocator = cuco::cuda_allocator<cuco::bucket<T, BucketSize>>>
class bucket_storage : public detail::bucket_storage_base<T, BucketSize, Extent> {
 public:
  /// Array of buckets base class type
  using base_type = detail::bucket_storage_base<T, BucketSize, Extent>;

  using base_type::bucket_size;  ///< Number of elements processed per bucket

  using extent_type = typename base_type::extent_type;  ///< Storage extent type
  using size_type   = typename base_type::size_type;    ///< Storage size type
  using value_type  = typename base_type::value_type;   ///< Slot type
  using bucket_type = typename base_type::bucket_type;  ///< Slot bucket type

  using base_type::capacity;
  using base_type::num_buckets;

  /// Type of the allocator to (de)allocate buckets
  using allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<bucket_type>;
  using bucket_deleter_type =
    detail::custom_deleter<size_type, allocator_type>;  ///< Type of bucket deleter
  using ref_type = bucket_storage_ref<value_type, bucket_size, extent_type>;  ///< Storage ref type

  /**
   * @brief Constructor of bucket storage.
   *
   * @note The input `size` should be exclusively determined by the return value of
   * `make_bucket_extent` since it depends on the requested low-bound value, the probing scheme, and
   * the storage.
   *
   * @param size Number of buckets to (de)allocate
   * @param allocator Allocator used for (de)allocating device storage
   */
  explicit constexpr bucket_storage(Extent size, Allocator const& allocator = {});

  bucket_storage(bucket_storage&&) = default;  ///< Move constructor
  /**
   * @brief Replaces the contents of the storage with another storage.
   *
   * @return Reference of the current storage object
   */
  bucket_storage& operator=(bucket_storage&&) = default;
  ~bucket_storage()                           = default;  ///< Destructor

  bucket_storage(bucket_storage const&)            = delete;
  bucket_storage& operator=(bucket_storage const&) = delete;

  /**
   * @brief Gets buckets array.
   *
   * @return Pointer to the first bucket
   */
  [[nodiscard]] constexpr bucket_type* data() const noexcept;

  /**
   * @brief Gets the storage allocator.
   *
   * @return The storage allocator
   */
  [[nodiscard]] constexpr allocator_type allocator() const noexcept;

  /**
   * @brief Gets bucket storage reference.
   *
   * @return Reference of bucket storage
   */
  [[nodiscard]] constexpr ref_type ref() const noexcept;

  /**
   * @brief Initializes each slot in the bucket storage to contain `key`.
   *
   * @param key Key to which all keys in `slots` are initialized
   * @param stream Stream used for executing the kernel
   */
  void initialize(value_type key, cuda::stream_ref stream = {});

  /**
   * @brief Asynchronously initializes each slot in the bucket storage to contain `key`.
   *
   * @param key Key to which all keys in `slots` are initialized
   * @param stream Stream used for executing the kernel
   */
  void initialize_async(value_type key, cuda::stream_ref stream = {}) noexcept;

 private:
  allocator_type allocator_;            ///< Allocator used to (de)allocate buckets
  bucket_deleter_type bucket_deleter_;  ///< Custom buckets deleter
  /// Pointer to the bucket storage
  std::unique_ptr<bucket_type, bucket_deleter_type> buckets_;
};
}  // namespace cuco

#include <cuco/detail/storage/bucket_storage.inl>
