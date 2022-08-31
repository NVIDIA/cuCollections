/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuco/allocator.hpp>
#include <cuco/detail/error.hpp>
#include <cuco/detail/pair.cuh>
#include <cuco/extent.cuh>

#include <cuda/atomic>

#include <cstddef>
#include <memory>

namespace cuco {
namespace experimental {
namespace detail {
/**
 * @brief Custom deleter for unique pointer of slots.
 *
 * @tparam Allocator Type of allocator used for device storage
 */
template <typename Allocator>
struct custom_deleter {
  using pointer = typename Allocator::value_type*;  ///< Value pointer type

  /**
   * @brief Constructor of custom deleter.
   *
   * @param size Number of values to deallocate
   * @param allocator Allocator used for deallocating device storage
   */
  custom_deleter(std::size_t const size, Allocator& allocator) : size_{size}, allocator_{allocator}
  {
  }

  /**
   * @brief Operator for deallocation
   *
   * @param ptr Pointer to the first value for deallocation
   */
  void operator()(pointer ptr) { allocator_.deallocate(ptr, size_); }

  std::size_t size_;      ///< Number of values to delete
  Allocator& allocator_;  ///< Allocator used deallocating values
};

/**
 * @brief Base class of open addressing storage. This class should not be used directly.
 *
 * @tparam Extent Type of extent denoting storage capacity
 */
template <typename Extent>
class storage_base {
 public:
  using extent_type = Extent;                            ///< Storage extent type
  using size_type   = typename extent_type::value_type;  ///< Storage size type

  /**
   * @brief Constructor of base storage.
   *
   * @param capacity Number of slots to (de)allocate
   */
  storage_base(Extent capacity) : capacity_{capacity} {}

  /**
   * @brief Gets the total number of slots in the current storage.
   *
   * @return The total number of slots
   */
  __host__ __device__ size_type capacity() const noexcept { return capacity_; }

 protected:
  extent_type capacity_;  ///< Total number of slots
};

/**
 * @brief Device counter storage class.
 *
 * @tparam SizeType Type of storage size
 * @tparam Scope The scope in which the counter will be used by individual threads
 * @tparam Allocator Type of allocator used for device storage
 */
template <typename SizeType, cuda::thread_scope Scope, typename Allocator>
class counter_storage : public storage_base<cuco::experimental::extent<SizeType, 1>> {
 public:
  using storage_base<cuco::experimental::extent<SizeType, 1>>::capacity_;  ///< Storage capacity

  using size_type      = SizeType;                        ///< Size type
  using counter_type   = cuda::atomic<size_type, Scope>;  ///< Type of the counter
  using allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<
    counter_type>;  ///< Type of the allocator to (de)allocate counter
  using counter_deleter_type = custom_deleter<allocator_type>;  ///< Type of counter deleter

  /**
   * @brief Constructor of counter storage.
   *
   * @param allocator Allocator used for (de)allocating device storage
   */
  counter_storage(Allocator const& allocator)
    : storage_base<cuco::experimental::extent<SizeType, 1>>{cuco::experimental::extent<size_type,
                                                                                       1>{}},
      allocator_{allocator},
      counter_deleter_{capacity_, allocator_},
      counter_{allocator_.allocate(capacity_), counter_deleter_}
  {
  }

  /**
   * @brief Asynchronously resets counter to zero.
   *
   * @param stream CUDA stream used to reset
   */
  inline void reset(cudaStream_t stream)
  {
    static_assert(sizeof(size_type) == sizeof(counter_type));
    CUCO_CUDA_TRY(cudaMemsetAsync(counter_.get(), 0, sizeof(counter_type), stream));
  }

  /**
   * @brief Gets slots array.
   *
   * @return Pointer to the first slot
   */
  counter_type* get() noexcept { return counter_.get(); }

  /**
   * @brief Gets slots array.
   *
   * @return Pointer to the first slot
   */
  counter_type const* get() const noexcept { return counter_.get(); }

 private:
  allocator_type allocator_;              ///< Allocator used to (de)allocate counter
  counter_deleter_type counter_deleter_;  ///< Custom counter deleter
  std::unique_ptr<counter_type, counter_deleter_type> counter_;  ///< Pointer to counter storage
};

/**
 * @brief Non-owning AoS storage view type.
 *
 * @tparam T Storage element type
 * @tparam Extent Type of extent denoting storage capacity
 */
template <typename T, typename Extent>
class aos_storage_view {
 public:
  using value_type  = T;                                 ///< Storage element type
  using extent_type = Extent;                            ///< Extent type
  using size_type   = typename extent_type::value_type;  ///< Size type

  /**
   * @brief Constructor of AoS storage view.
   *
   * @param slots Pointer to the slots array
   * @param capacity Size of the slots array
   */
  explicit aos_storage_view(value_type* slots, Extent const capacity) noexcept
    : slots_{slots}, capacity_{capacity}
  {
  }

  /**
   * @brief Gets slots array.
   *
   * @return Pointer to the first slot
   */
  __device__ inline value_type* slots() noexcept { return slots_; }

  /**
   * @brief Gets slots array.
   *
   * @return Pointer to the first slot
   */
  __device__ inline value_type const* slots() const noexcept { return slots_; }

  /**
   * @brief Gets the total number of slots in the current storage.
   *
   * @return The total number of slots
   */
  __device__ inline size_type capacity() const noexcept { return capacity_; }

 private:
  value_type* slots_;     ///< Pointer to the slots array
  extent_type capacity_;  ///< Size of the slots array
};

/**
 * @brief Array of structure open addressing storage class.
 *
 * @tparam T struct type
 * @tparam Extent Type of extent denoting storage capacity
 * @tparam Allocator Type of allocator used for device storage
 */
template <typename T, class Extent, typename Allocator>
class aos_storage : public storage_base<Extent> {
 public:
  using extent_type = typename storage_base<Extent>::extent_type;  ///< Storage extent type
  using size_type   = typename storage_base<Extent>::size_type;    ///< Storage size type
  using storage_base<Extent>::capacity_;                           ///< Storage capacity

  using value_type = T;  ///< Type of structs
  using allocator_type =
    typename std::allocator_traits<Allocator>::rebind_alloc<value_type>;  ///< Type of the allocator
                                                                          ///< to (de)allocate slots
  using slot_deleter_type = custom_deleter<allocator_type>;               ///< Type of slot deleter
  using view_type         = aos_storage_view<value_type, extent_type>;    ///< Storage view type

  /**
   * @brief Constructor of AoS storage.
   *
   * @param size Number of slots to (de)allocate
   * @param allocator Allocator used for (de)allocating device storage
   */
  aos_storage(Extent const size, Allocator const& allocator)
    : storage_base<Extent>{size},
      allocator_{allocator},
      slot_deleter_{capacity_, allocator_},
      slots_{allocator_.allocate(capacity_), slot_deleter_}
  {
  }

  aos_storage(aos_storage&&) = default;  ///< Move constructor
  /**
   * @brief Replaces the contents of the storage with another storage.
   *
   * @return Reference of the current storage object
   */
  aos_storage& operator=(aos_storage&&) = default;
  ~aos_storage()                        = default;  ///< Destructor

  aos_storage(aos_storage const&) = delete;
  aos_storage& operator=(aos_storage const&) = delete;

  /**
   * @brief Gets slots array.
   *
   * @return Pointer to the first slot
   */
  value_type* slots() noexcept { return slots_.get(); }

  /**
   * @brief Gets slots array.
   *
   * @return Pointer to the first slot
   */
  value_type const* slots() const noexcept { return slots_.get(); }

  /**
   * @brief Gets view of slot storage.
   *
   * @return View of slot storage
   */
  view_type view() noexcept { return view_type{this->slots(), this->capacity()}; }

 private:
  allocator_type allocator_;                              ///< Allocator used to (de)allocate slots
  slot_deleter_type slot_deleter_;                        ///< Custom slots deleter
  std::unique_ptr<value_type, slot_deleter_type> slots_;  ///< Pointer to AoS slots storage
};
}  // namespace detail
}  // namespace experimental
}  // namespace cuco