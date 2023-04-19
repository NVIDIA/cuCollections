/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cuco/detail/tuning.cuh>
#include <cuco/extent.cuh>

#include <cuda/std/array>

#include <cstddef>
#include <iterator>
#include <memory>

namespace cuco {
namespace experimental {
namespace detail {
/**
 * @brief Base class of array of slot windows open addressing storage.
 *
 * This should NOT be used directly.
 *
 * @tparam WindowSize Number of elements in each window
 * @tparam T Element type
 * @tparam Extent Type of extent denoting the number of windows
 */
template <int32_t WindowSize, typename T, typename Extent>
class aow_storage_base : public storage_base<Extent> {
 public:
  /**
   * @brief The number of elements (slots) processed per window.
   */
  static constexpr int32_t window_size = WindowSize;

  using extent_type = typename storage_base<Extent>::extent_type;  ///< Storage extent type
  using size_type   = typename storage_base<Extent>::size_type;    ///< Storage size type

  using value_type  = T;                                          ///< Slot type
  using window_type = cuda::std::array<value_type, window_size>;  ///< Slot window type

  /**
   * @brief Constructor of AoW base storage.
   *
   * @param size Number of windows to store
   */
  explicit constexpr aow_storage_base(Extent size) : storage_base<Extent>{size} {}

  /**
   * @brief Gets the total number of slot windows in the current storage.
   *
   * @return The total number of slot windows
   */
  [[nodiscard]] __host__ __device__ constexpr extent_type num_windows() const noexcept
  {
    return storage_base<Extent>::capacity();
  }

  /**
   * @brief Gets the total number of slots in the current storage.
   *
   * @return The total number of slots
   */
  [[nodiscard]] __host__ __device__ constexpr auto capacity() const noexcept
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
  using value_type  = typename base_type::value_type;   ///< Slot type
  using window_type = typename base_type::window_type;  ///< Slot window type

  using base_type::capacity;
  using base_type::num_windows;

  /**
   * @brief Constructor of AoS storage ref.
   *
   * @param windows Pointer to the windows array
   * @param num_windows Number of windows
   */
  explicit constexpr aow_storage_ref(Extent num_windows, window_type* windows) noexcept
    : aow_storage_base<WindowSize, T, Extent>{num_windows}, windows_{windows}
  {
  }

  /**
   * @brief Custom forward iterator for the convenience of `find` operations.
   */
  struct Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;  ///< Iterator category
    using difference_type   = std::ptrdiff_t;             ///< Iterator difference type
    using reference         = value_type&;                ///< Iterator reference type

    /**
     * @brief Constructs a device side forward iterator for slots.
     *
     * @param current The slot index
     * @param data Pointer to the window storage
     */
    __device__ constexpr Iterator(size_type current, window_type* data) noexcept
      : current_{current}, data_{data}
    {
    }

    /**
     * @brief Prefix increment operator
     *
     * @return Current iterator
     */
    __device__ constexpr Iterator& operator++() noexcept
    {
      current_++;
      return *this;
    }

    /**
     * @brief Postfix increment operator
     *
     * @return Old iterator before increment
     */
    __device__ constexpr Iterator operator++(int32_t) noexcept
    {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }

    /**
     * @brief Dereference operator
     *
     * @return Reference to the current slot
     */
    __device__ constexpr reference operator*() const
    {
      auto const window_index       = current_ / window_size;
      auto const intra_window_index = current_ % window_size;
      return data_[window_index][intra_window_index];
    }

    /**
     * Equality operator
     *
     * @return True if two iterators are identical
     */
    friend __device__ constexpr bool operator==(Iterator const& lhs, Iterator const& rhs) noexcept
    {
      return lhs.current_ == rhs.current_ and lhs.data_ == rhs.data_;
    }

    /**
     * Inequality operator
     *
     * @return True if two iterators are not identical
     */
    friend __device__ constexpr bool operator!=(Iterator const& lhs, Iterator const& rhs) noexcept
    {
      return not lhs == rhs;
    }

   private:
    size_type current_{};  ///< Current index
    window_type* data_{};  ///< Pointer to the window data
  };
  using iterator       = Iterator;        ///< Forward iterator type
  using const_iterator = Iterator const;  ///< Const forward iterator type

  /**
   * @brief Returns an iterator to one past the last slot.
   *
   * This is provided for convenience for those familiar with checking
   * an iterator returned from `find()` against the `end()` iterator.
   *
   * @return An iterator to one past the last slot
   */
  [[nodiscard]] __device__ constexpr iterator end() noexcept
  {
    return {this->capacity(), this->data()};
  }

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
    return {this->capacity(), this->data()};
  }

  /**
   * @brief Gets windows array.
   *
   * @return Pointer to the first window
   */
  [[nodiscard]] __device__ constexpr window_type* data() noexcept { return windows_; }

  /**
   * @brief Gets windows array.
   *
   * @return Pointer to the first window
   */
  [[nodiscard]] __device__ constexpr window_type* data() const noexcept { return windows_; }

  /**
   * @brief Returns an array of slots (or a window) for a given index.
   *
   * @param index Index of the window
   * @return An array of slots
   */
  [[nodiscard]] __device__ constexpr window_type operator[](size_type index) const noexcept
  {
    return *reinterpret_cast<window_type*>(
      __builtin_assume_aligned(this->data() + index, sizeof(value_type) * window_size));
  }

 private:
  window_type* windows_;  ///< Pointer to the windows array
};

/**
 * @brief Array of slot Window open addressing storage class.
 *
 * @tparam WindowSize Number of slots in each window
 * @tparam T Slot type
 * @tparam Extent Type of extent denoting number of windows
 * @tparam Allocator Type of allocator used for device storage (de)allocation
 */
template <int32_t WindowSize, typename T, typename Extent, typename Allocator>
class aow_storage : public aow_storage_base<WindowSize, T, Extent> {
 public:
  using base_type = aow_storage_base<WindowSize, T, Extent>;  ///< AoW base class type

  using base_type::window_size;  ///< Number of elements processed per window

  using extent_type = typename base_type::extent_type;  ///< Storage extent type
  using size_type   = typename base_type::size_type;    ///< Storage size type
  using value_type  = typename base_type::value_type;   ///< Slot type
  using window_type = typename base_type::window_type;  ///< Slot window type

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
   * @note The input `size` should be exclusively determined by the return value of
   * `make_valid_extent` since it depends on the requested low-bound value, the probing scheme, and
   * the storage.
   *
   * @param size Number of windows to (de)allocate
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
  [[nodiscard]] constexpr window_type* data() const noexcept { return windows_.get(); }

  /**
   * @brief Gets window storage reference.
   *
   * @return Reference of window storage
   */
  [[nodiscard]] constexpr ref_type ref() const noexcept
  {
    return ref_type{this->num_windows(), this->data()};
  }

  /**
   * @brief Initializes each slot in the AoW storage to contain `key`.
   *
   * @param key Key to which all keys in `slots` are initialized
   * @param stream Stream used for executing the kernel
   */
  void initialize(value_type key, cudaStream_t stream) noexcept
  {
    auto constexpr stride = 4;
    auto const grid_size  = (this->num_windows() + stride * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
                           (stride * detail::CUCO_DEFAULT_BLOCK_SIZE);

    detail::initialize<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
      this->data(), this->num_windows(), key);
  }

 private:
  allocator_type allocator_;            ///< Allocator used to (de)allocate windows
  window_deleter_type window_deleter_;  ///< Custom windows deleter
  std::unique_ptr<window_type, window_deleter_type> windows_;  ///< Pointer to AoW storage
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
