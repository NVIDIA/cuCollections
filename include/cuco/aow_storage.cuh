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

#include <cuco/cuda_stream_ref.hpp>
#include <cuco/detail/storage/aow_storage_base.cuh>
#include <cuco/extent.cuh>
#include <cuco/utility/allocator.hpp>

#include <cuda/std/array>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>

namespace cuco {
namespace experimental {

/// Window type alias
template <typename T, int32_t WindowSize>
using window = detail::window<T, WindowSize>;

/// forward declaration
template <typename T, int32_t WindowSize, typename Extent>
class aow_storage_ref;

/**
 * @brief Array of Window open addressing storage class.
 *
 * @tparam T Slot type
 * @tparam WindowSize Number of slots in each window
 * @tparam Extent Type of extent denoting number of windows
 * @tparam Allocator Type of allocator used for device storage (de)allocation
 */
template <typename T,
          int32_t WindowSize,
          typename Extent    = cuco::experimental::extent<std::size_t>,
          typename Allocator = cuco::cuda_allocator<cuco::experimental::window<T, WindowSize>>>
class aow_storage : public detail::aow_storage_base<T, WindowSize, Extent> {
 public:
  using base_type = detail::aow_storage_base<T, WindowSize, Extent>;  ///< AoW base class type

  using base_type::window_size;  ///< Number of elements processed per window

  using extent_type = typename base_type::extent_type;  ///< Storage extent type
  using size_type   = typename base_type::size_type;    ///< Storage size type
  using value_type  = typename base_type::value_type;   ///< Slot type
  using window_type = typename base_type::window_type;  ///< Slot window type

  using base_type::capacity;
  using base_type::num_windows;

  /// Type of the allocator to (de)allocate windows
  using allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<window_type>;
  using window_deleter_type =
    detail::custom_deleter<size_type, allocator_type>;  ///< Type of window deleter
  using ref_type = aow_storage_ref<value_type, window_size, extent_type>;  ///< Storage ref type

  /**
   * @brief Constructor of AoW storage.
   *
   * @note The input `size` should be exclusively determined by the return value of
   * `make_window_extent` since it depends on the requested low-bound value, the probing scheme, and
   * the storage.
   *
   * @param size Number of windows to (de)allocate
   * @param allocator Allocator used for (de)allocating device storage
   */
  explicit constexpr aow_storage(Extent size, Allocator const& allocator = {}) noexcept;

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
  [[nodiscard]] constexpr window_type* data() const noexcept;

  /**
   * @brief Gets the storage allocator.
   *
   * @return The storage allocator
   */
  [[nodiscard]] constexpr allocator_type allocator() const noexcept;

  /**
   * @brief Gets window storage reference.
   *
   * @return Reference of window storage
   */
  [[nodiscard]] constexpr ref_type ref() const noexcept;

  /**
   * @brief Initializes each slot in the AoW storage to contain `key`.
   *
   * @param key Key to which all keys in `slots` are initialized
   * @param stream Stream used for executing the kernel
   */
  void initialize(value_type key, cuda_stream_ref stream = {}) noexcept;

  /**
   * @brief Asynchronously initializes each slot in the AoW storage to contain `key`.
   *
   * @param key Key to which all keys in `slots` are initialized
   * @param stream Stream used for executing the kernel
   */
  void initialize_async(value_type key, cuda_stream_ref stream = {}) noexcept;

 private:
  allocator_type allocator_;            ///< Allocator used to (de)allocate windows
  window_deleter_type window_deleter_;  ///< Custom windows deleter
  std::unique_ptr<window_type, window_deleter_type> windows_;  ///< Pointer to AoW storage
};

/**
 * @brief Non-owning AoW storage reference type.
 *
 * @tparam T Storage element type
 * @tparam WindowSize Number of slots in each window
 * @tparam Extent Type of extent denoting storage capacity
 */
template <typename T, int32_t WindowSize, typename Extent = cuco::experimental::extent<std::size_t>>
class aow_storage_ref : public detail::aow_storage_base<T, WindowSize, Extent> {
 public:
  using base_type = detail::aow_storage_base<T, WindowSize, Extent>;  ///< AoW base class type

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
   * @param size Number of windows
   * @param windows Pointer to the windows array
   */
  __host__ __device__ explicit constexpr aow_storage_ref(Extent size,
                                                         window_type* windows) noexcept;

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
   * @brief Gets windows array.
   *
   * @return Pointer to the first window
   */
  [[nodiscard]] __device__ constexpr window_type* data() noexcept;

  /**
   * @brief Gets windows array.
   *
   * @return Pointer to the first window
   */
  [[nodiscard]] __device__ constexpr window_type* data() const noexcept;

  /**
   * @brief Returns an array of slots (or a window) for a given index.
   *
   * @param index Index of the window
   * @return An array of slots
   */
  [[nodiscard]] __device__ constexpr window_type operator[](size_type index) const noexcept;

 private:
  window_type* windows_;  ///< Pointer to the windows array
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/storage/aow_storage.inl>
