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
#include <cuco/detail/storage/kernels.cuh>
#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/extent.cuh>

#include <cuda/std/array>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>

namespace cuco {
namespace experimental {

template <typename T, int32_t WindowSize, typename Extent, typename Allocator>
constexpr aow_storage<T, WindowSize, Extent, Allocator>::aow_storage(
  Extent size, Allocator const& allocator) noexcept
  : detail::aow_storage_base<T, WindowSize, Extent>{size},
    allocator_{allocator},
    window_deleter_{capacity(), allocator_},
    windows_{allocator_.allocate(capacity()), window_deleter_}
{
}

template <typename T, int32_t WindowSize, typename Extent, typename Allocator>
constexpr aow_storage<T, WindowSize, Extent, Allocator>::window_type*
aow_storage<T, WindowSize, Extent, Allocator>::data() const noexcept
{
  return windows_.get();
}

template <typename T, int32_t WindowSize, typename Extent, typename Allocator>
constexpr aow_storage<T, WindowSize, Extent, Allocator>::allocator_type
aow_storage<T, WindowSize, Extent, Allocator>::allocator() const noexcept
{
  return allocator_;
}

template <typename T, int32_t WindowSize, typename Extent, typename Allocator>
constexpr aow_storage<T, WindowSize, Extent, Allocator>::ref_type
aow_storage<T, WindowSize, Extent, Allocator>::ref() const noexcept
{
  return ref_type{this->window_extent(), this->data()};
}

template <typename T, int32_t WindowSize, typename Extent, typename Allocator>
void aow_storage<T, WindowSize, Extent, Allocator>::initialize(value_type key,
                                                               cuda_stream_ref stream) noexcept
{
  this->initialize_async(key, stream);
  stream.synchronize();
}

template <typename T, int32_t WindowSize, typename Extent, typename Allocator>
void aow_storage<T, WindowSize, Extent, Allocator>::initialize_async(
  value_type key, cuda_stream_ref stream) noexcept
{
  auto constexpr cg_size = 1;
  auto constexpr stride  = 4;
  auto const grid_size   = cuco::detail::grid_size(this->num_windows(), cg_size, stride);

  detail::initialize<<<grid_size, cuco::detail::default_block_size(), 0, stream>>>(
    this->data(), this->num_windows(), key);
}

template <typename T, int32_t WindowSize, typename Extent>
__host__ __device__ constexpr aow_storage_ref<T, WindowSize, Extent>::aow_storage_ref(
  Extent size, window_type* windows) noexcept
  : detail::aow_storage_base<T, WindowSize, Extent>{size}, windows_{windows}
{
}

template <typename T, int32_t WindowSize, typename Extent>
struct aow_storage_ref<T, WindowSize, Extent>::iterator {
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

template <typename T, int32_t WindowSize, typename Extent>
__device__ constexpr aow_storage_ref<T, WindowSize, Extent>::iterator
aow_storage_ref<T, WindowSize, Extent>::end() noexcept
{
  return iterator{reinterpret_cast<value_type*>(this->data() + this->capacity())};
}

template <typename T, int32_t WindowSize, typename Extent>
__device__ constexpr aow_storage_ref<T, WindowSize, Extent>::const_iterator
aow_storage_ref<T, WindowSize, Extent>::end() const noexcept
{
  return const_iterator{reinterpret_cast<value_type*>(this->data() + this->capacity())};
}

template <typename T, int32_t WindowSize, typename Extent>
__device__ constexpr aow_storage_ref<T, WindowSize, Extent>::window_type*
aow_storage_ref<T, WindowSize, Extent>::data() noexcept
{
  return windows_;
}

template <typename T, int32_t WindowSize, typename Extent>
__device__ constexpr aow_storage_ref<T, WindowSize, Extent>::window_type*
aow_storage_ref<T, WindowSize, Extent>::data() const noexcept
{
  return windows_;
}

template <typename T, int32_t WindowSize, typename Extent>
__device__ constexpr aow_storage_ref<T, WindowSize, Extent>::window_type
aow_storage_ref<T, WindowSize, Extent>::operator[](size_type index) const noexcept
{
  return *reinterpret_cast<window_type*>(
    __builtin_assume_aligned(this->data() + index, sizeof(value_type) * window_size));
}

}  // namespace experimental
}  // namespace cuco
