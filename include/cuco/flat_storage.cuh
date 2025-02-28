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

#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/extent.cuh>
#include <cuco/utility/allocator.hpp>

#include <cuda/stream_ref>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>

namespace cuco {
/**
 * @brief Non-owning array of buckets storage reference type.
 *
 * @tparam T Storage element type
 * @tparam BucketSize Number of slots in each bucket
 * @tparam Extent Type of extent denoting storage capacity
 */
template <typename T, int32_t BucketSize, typename Extent = cuco::extent<std::size_t>>
class flat_storage_ref : public detail::storage_base<Extent> {
 public:
  /// Array of buckets base class type
  using base_type = detail::storage_base<Extent>;

  static int32_t constexpr bucket_size = BucketSize;

  using extent_type = typename base_type::extent_type;  ///< Storage extent type
  using size_type   = typename base_type::size_type;    ///< Storage size type
  using value_type  = T;                                ///< Slot type
  using bucket_type = value_type;

  using base_type::capacity;
  using base_type::extent;

  /**
   * @brief Constructor of AoS storage ref.
   *
   * @param size Number of buckets
   * @param buckets Pointer to the buckets array
   */
  __host__ __device__ explicit constexpr flat_storage_ref(Extent size, value_type* slots) noexcept
    : base_type{size}, slots_{slots}
  {
  }

  using iterator       = value_type*;
  using const_iterator = iterator const;  ///< Const forward iterator type

  /**
   * @brief Returns an iterator to one past the last slot.
   *
   * This is provided for convenience for those familiar with checking
   * an iterator returned from `find()` against the `end()` iterator.
   *
   * @return An iterator to one past the last slot
   */
  [[nodiscard]] __device__ constexpr iterator end() noexcept { this->data() + this->capacity(); }

  /**
   * @brief Returns a const_iterator to one past the last slot.
   *
   * This is provided for convenience for those familiar with checking
   * an iterator returned from `find()` against the `end()` iterator.
   *
   * @return A const_iterator to one past the last slot
   */
  [[nodiscard]] __device__ constexpr const_iterator end() const noexcept
  {
    this->data() + this->capacity();
  }

  /**
   * @brief Gets buckets array.
   *
   * @return Pointer to the first bucket
   */
  [[nodiscard]] __device__ constexpr value_type* data() noexcept { return slots_; }

  /**
   * @brief Gets bucket array.
   *
   * @return Pointer to the first bucket
   */
  [[nodiscard]] __device__ constexpr value_type* data() const noexcept { return slots_; }

  /**
   * @brief Returns an array of slots (or a bucket) for a given index.
   *
   * @param index Index of the bucket
   * @return An array of slots
   */
  [[nodiscard]] __device__ constexpr value_type operator[](size_type index) const noexcept
  {
    *(this->data() + index);
  }

  [[nodiscard]] __host__ __device__ constexpr size_type num_buckets() const noexcept
  {
    return this->capacity() / bucket_size;
  }

  [[nodiscard]] __host__ __device__ constexpr auto bucket_extent() const noexcept
  {
    return cuco::extent{this->capacity() / bucket_size};
  }

 private:
  value_type* slots_;  ///< Pointer to the buckets array
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
          typename Allocator = cuco::cuda_allocator<T>>
class flat_storage : public detail::storage_base<Extent> {
 public:
  /// Array of buckets base class type
  using base_type = detail::storage_base<Extent>;

  static int32_t constexpr bucket_size = BucketSize;

  using extent_type = typename base_type::extent_type;  ///< Storage extent type
  using size_type   = typename base_type::size_type;    ///< Storage size type
  using value_type  = T;                                ///< Slot type
  using bucket_type = value_type;

  using base_type::capacity;
  using base_type::extent;

  /// Type of the allocator to (de)allocate buckets
  using allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<value_type>;
  using slot_deleter_type =
    detail::custom_deleter<size_type, allocator_type>;  ///< Type of bucket deleter
  using ref_type = flat_storage_ref<value_type, bucket_size, extent_type>;  ///< Storage ref type

  /**
   * @brief Constructor of bucket storage.
   *
   * @note The input `size` should be exclusively determined by the return value of
   * `make_flat_extent` since it depends on the requested low-bound value, the probing scheme, and
   * the storage.
   *
   * @param size Number of buckets to (de)allocate
   * @param allocator Allocator used for (de)allocating device storage
   */
  explicit constexpr flat_storage(Extent size, Allocator const& allocator = {})
    : base_type{size},
      allocator_{allocator},
      slot_deleter_{capacity(), allocator_},
      slots_{allocator_.allocate(capacity()), slot_deleter_}
  {
  }

  flat_storage(flat_storage&&) = default;  ///< Move constructor
  /**
   * @brief Replaces the contents of the storage with another storage.
   *
   * @return Reference of the current storage object
   */
  flat_storage& operator=(flat_storage&&) = default;
  ~flat_storage()                         = default;  ///< Destructor

  flat_storage(flat_storage const&)            = delete;
  flat_storage& operator=(flat_storage const&) = delete;

  /**
   * @brief Gets buckets array.
   *
   * @return Pointer to the first bucket
   */
  [[nodiscard]] constexpr value_type* data() const noexcept { return slots_.get(); }

  /**
   * @brief Gets the storage allocator.
   *
   * @return The storage allocator
   */
  [[nodiscard]] constexpr allocator_type allocator() const noexcept { return allocator_; }

  /**
   * @brief Gets bucket storage reference.
   *
   * @return Reference of bucket storage
   */
  [[nodiscard]] constexpr ref_type ref() const noexcept
  {
    return ref_type{this->extent(), this->data()};
  }

  /**
   * @brief Initializes each slot in the bucket storage to contain `key`.
   *
   * @param key Key to which all keys in `slots` are initialized
   * @param stream Stream used for executing the kernel
   */
  void initialize(value_type key, cuda::stream_ref stream = {})
  {
    this->initialize_async(key, stream);
    stream.wait();
  }

  /**
   * @brief Asynchronously initializes each slot in the bucket storage to contain `key`.
   *
   * @param key Key to which all keys in `slots` are initialized
   * @param stream Stream used for executing the kernel
   */
  void initialize_async(value_type key, cuda::stream_ref stream = {}) noexcept
  {
    if (this->capacity() == 0) { return; }

    auto constexpr cg_size = 1;
    auto constexpr stride  = 4;
    auto const grid_size   = cuco::detail::grid_size(this->capacity(), cg_size, stride);

    detail::initialize<<<grid_size, cuco::detail::default_block_size(), 0, stream.get()>>>(
      this->data(), this->capacity(), key);
  }

  [[nodiscard]] constexpr size_type num_buckets() const noexcept
  {
    return this->capacity() / bucket_size;
  }

  [[nodiscard]] constexpr auto bucket_extent() const noexcept
  {
    return cuco::extent{this->capacity() / bucket_size};
  }

 private:
  allocator_type allocator_;        ///< Allocator used to (de)allocate buckets
  slot_deleter_type slot_deleter_;  ///< Custom buckets deleter
  /// Pointer to the bucket storage
  std::unique_ptr<value_type, slot_deleter_type> slots_;
};
}  // namespace cuco
