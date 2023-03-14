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
#include <cuco/detail/common_kernels.cuh>
#include <cuco/detail/error.hpp>
#include <cuco/detail/pair.cuh>
#include <cuco/detail/tuning.cuh>
#include <cuco/detail/utils.cuh>
#include <cuco/extent.cuh>

#include <cuda/atomic>
#include <cuda/std/array>

#include <cstddef>
#include <memory>

namespace cuco {
namespace experimental {
namespace detail {
/**
 * @brief Custom deleter for unique pointer.
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
  explicit constexpr custom_deleter(std::size_t const size, Allocator& allocator)
    : size_{size}, allocator_{allocator}
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
 * @brief Base class of open addressing storage.
 *
 * This class should not be used directly.
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
   * @param size Number of elements to (de)allocate
   */
  explicit constexpr storage_base(Extent size) : capacity_{size} {}

  /**
   * @brief Gets the total number of elements in the current storage.
   *
   * @return The total number of elements
   */
  [[nodiscard]] __host__ __device__ inline constexpr extent_type capacity() const noexcept
  {
    return capacity_;
  }

 protected:
  extent_type capacity_;  ///< Total number of elements
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
  using storage_base<cuco::experimental::extent<SizeType, 1>>::size_;  ///< Storage size

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
  explicit constexpr counter_storage(Allocator const& allocator)
    : storage_base<cuco::experimental::extent<SizeType, 1>>{cuco::experimental::extent<size_type,
                                                                                       1>{}},
      allocator_{allocator},
      counter_deleter_{size_, allocator_},
      counter_{allocator_.allocate(size_), counter_deleter_}
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
   * @brief Gets counter pointer.
   *
   * @return Pointer to the counter
   */
  [[nodiscard]] inline constexpr counter_type* get() noexcept { return counter_.get(); }

  /**
   * @brief Gets counter array.
   *
   * @return Pointer to the counter
   */
  [[nodiscard]] inline constexpr counter_type* get() const noexcept { return counter_.get(); }

 private:
  allocator_type allocator_;              ///< Allocator used to (de)allocate counter
  counter_deleter_type counter_deleter_;  ///< Custom counter deleter
  std::unique_ptr<counter_type, counter_deleter_type> counter_;  ///< Pointer to counter storage
};

/**
 * @brief Base class of array of window structure open addressing storage.
 *
 * This should NOT be used directly.
 *
 * @tparam WindowSize Number of elements in each window
 * @tparam T Element type
 * @tparam Extent Type of extent denoting number of windows
 */
template <int32_t WindowSize, typename T, typename Extent>
class aow_storage_base : public storage_base<Extent> {
 public:
  /**
   * @brief The number of elements processed per window.
   */
  static constexpr int32_t window_size = WindowSize;

  using extent_type = typename storage_base<Extent>::extent_type;  ///< Storage extent type
  using size_type   = typename storage_base<Extent>::size_type;    ///< Storage size type

  using value_type  = T;                                          ///< Type of structs
  using window_type = cuda::std::array<value_type, window_size>;  ///< Type of struct windows

  /**
   * @brief Constructor of AoW base storage.
   *
   * @param size Number of elemets to store
   */
  explicit constexpr aow_storage_base(Extent size) : storage_base<Extent>{size} {}

  /**
   * @brief Gets the total number of slot windows in the current storage.
   *
   * @return The total number of slot windows
   */
  [[nodiscard]] __host__ __device__ inline constexpr extent_type num_windows() const noexcept
  {
    return storage_base<Extent>::capacity();
  }

  /**
   * @brief Gets the total number of slots in the current storage.
   *
   * @return The total number of slots
   */
  [[nodiscard]] __host__ __device__ inline constexpr auto capacity() const noexcept
  {
    return storage_base<Extent>::capacity().template multiply<window_size>();
  }
};

/**
 * @brief Non-owning AoW storage reference type.
 *
 * @tparam WindowSize Number of slots in each window
 * @tparam T Storage element type
 * @tparam Extent Type of extent denoting storage capacity
 */
template <int32_t WindowSize, typename T, typename Extent>
class aow_storage_ref : public aow_storage_base<WindowSize, T, Extent> {
 public:
  using base_type = aow_storage_base<WindowSize, T, Extent>;  ///< AoW base class type

  using base_type::window_size;  ///< Number of elements processed per window

  using extent_type = typename base_type::extent_type;  ///< Storage extent type
  using size_type   = typename base_type::size_type;    ///< Storage size type
  using value_type  = typename base_type::value_type;   ///< Storage element type
  using window_type = typename base_type::window_type;  ///< Window storage type

  using base_type::capacity;
  using base_type::num_windows;

  /**
   * @brief Constructor of AoS storage ref.
   *
   * @param windows Pointer to the windows array
   * @param num_windows Number of slots
   */
  explicit constexpr aow_storage_ref(Extent num_windows, window_type* windows) noexcept
    : aow_storage_base<WindowSize, T, Extent>{num_windows}, windows_{windows}
  {
  }

  /**
   * @brief Gets windows array.
   *
   * @return Pointer to the first window
   */
  [[nodiscard]] __device__ inline constexpr window_type* windows() noexcept { return windows_; }

  /**
   * @brief Gets windows array.
   *
   * @return Pointer to the first window
   */
  [[nodiscard]] __device__ inline constexpr window_type* windows() const noexcept
  {
    return windows_;
  }

  /**
   * @brief Returns an array of elements (window) for a given index.
   *
   * @param index Index of the first element of the window
   * @return An array of elements
   */
  [[nodiscard]] __device__ inline constexpr window_type window(size_type index) const noexcept
  {
    return *(windows_ + index);
  }

 private:
  window_type* windows_;  ///< Pointer to the windows array
};

/**
 * @brief Array of window structure open addressing storage class.
 *
 * @tparam WindowSize Number of slots in each window
 * @tparam T struct type
 * @tparam Extent Type of extent denoting number of windows
 * @tparam Allocator Type of allocator used for device storage
 */
template <int32_t WindowSize, typename T, typename Extent, typename Allocator>
class aow_storage : public aow_storage_base<WindowSize, T, Extent> {
 public:
  using base_type = aow_storage_base<WindowSize, T, Extent>;  ///< AoW base class type

  using base_type::window_size;  ///< Number of elements processed per window

  using extent_type = typename base_type::extent_type;  ///< Storage extent type
  using size_type   = typename base_type::size_type;    ///< Storage size type
  using value_type  = typename base_type::value_type;   ///< Storage element type
  using window_type = typename base_type::window_type;  ///< Window storage type

  using base_type::capacity;
  using base_type::num_windows;

  using allocator_type =
    typename std::allocator_traits<Allocator>::rebind_alloc<window_type>;  ///< Type of the
                                                                           ///< allocator to
                                                                           ///< (de)allocate windows
  using window_deleter_type = custom_deleter<allocator_type>;  ///< Type of window deleter
  using ref_type = aow_storage_ref<window_size, value_type, extent_type>;  ///< Storage ref type

  /**
   * @brief Constructor of AoW storage.
   *
   * @param size Number of slots to (de)allocate
   * @param allocator Allocator used for (de)allocating device storage
   */
  explicit constexpr aow_storage(Extent size, Allocator const& allocator)
    : aow_storage_base<WindowSize, T, Extent>{size},
      allocator_{allocator},
      window_deleter_{capacity(), allocator_},
      windows_{allocator_.allocate(capacity()), window_deleter_}
  {
  }

  aow_storage(aow_storage&&) = default;  ///< Move constructor
  /**
   * @brief Replaces the contents of the storage with another storage.
   *
   * @return Reference of the current storage object
   */
  aow_storage& operator=(aow_storage&&) = default;
  ~aow_storage()                        = default;  ///< Destructor

  aow_storage(aow_storage const&) = delete;
  aow_storage& operator=(aow_storage const&) = delete;

  /**
   * @brief Gets windows array.
   *
   * @return Pointer to the first window
   */
  [[nodiscard]] inline constexpr window_type* windows() const noexcept { return windows_.get(); }

  /**
   * @brief Gets window storage reference.
   *
   * @return Reference of window storage
   */
  [[nodiscard]] inline constexpr ref_type ref() const noexcept
  {
    return ref_type{this->num_windows(), this->windows()};
  }

  /**
   * @brief Initializes each slot in the flat storage to contain `key`.
   *
   * @param key Key to which all keys in `slots` are initialized
   * @param stream Stream used for executing the kernels
   */
  void initialize(value_type const key, cudaStream_t stream) noexcept
  {
    auto constexpr stride = 4;
    auto const grid_size  = (this->num_windows() + stride * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
                           (stride * detail::CUCO_DEFAULT_BLOCK_SIZE);

    detail::initialize<window_size><<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
      this->windows(), key, this->num_windows());
  }

 private:
  allocator_type allocator_;            ///< Allocator used to (de)allocate windows
  window_deleter_type window_deleter_;  ///< Custom windows deleter
  std::unique_ptr<window_type, window_deleter_type> windows_;  ///< Pointer to AoS windows storage
};

/**
 * @brief Intermediate class internally used by data structures
 *
 * @tparam StorageImpl Storage implementation class
 * @tparam T Storage element type
 * @tparam Extent Type of extent denoting number of windows
 * @tparam Allocator Type of allocator used for device storage
 */
template <class StorageImpl, class T, class Extent, class Allocator>
class storage : StorageImpl::template impl<T, Extent, Allocator> {
 public:
  /// Storage implementation type
  using impl_type  = typename StorageImpl::template impl<T, Extent, Allocator>;
  using ref_type   = typename impl_type::ref_type;  ///< Storage ref type
  using value_type = typename impl_type::value_type;

  /// Number of elements per window
  static constexpr int window_size = impl_type::window_size;

  using impl_type::capacity;
  using impl_type::initialize;
  using impl_type::num_windows;
  using impl_type::ref;
  using impl_type::windows;

  /**
   * @brief Constructs storage.
   *
   * @param size Number of slots to (de)allocate
   * @param allocator Allocator used for (de)allocating device storage
   */
  explicit constexpr storage(Extent size, Allocator const& allocator) : impl_type{size, allocator}
  {
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
