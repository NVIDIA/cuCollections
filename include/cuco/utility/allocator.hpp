/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/error.hpp>

#include <cuda/stream_ref>

#include <cstddef>

namespace cuco {
/**
 * @brief A stream-ordered device allocator using `cudaMallocAsync`/`cudaFreeAsync`.
 *
 * @tparam T The allocator's value type
 */
template <typename T>
class cuda_allocator {
 public:
  using value_type = T;  ///< Allocator's value type

  cuda_allocator() = default;

  /**
   * @brief Copy constructor.
   */
  template <class U>
  cuda_allocator(cuda_allocator<U> const&) noexcept
  {
  }

  /**
   * @brief Allocates storage for `n` objects of type `T` using `cudaMallocAsync`.
   *
   * @param n The number of objects to allocate storage for
   * @param stream The stream to order the allocation on
   * @return Pointer to the allocated storage
   */
  value_type* allocate(std::size_t n, cuda::stream_ref stream)
  {
    value_type* p;
    CUCO_CUDA_TRY(cudaMallocAsync(&p, sizeof(value_type) * n, stream.get()));
    return p;
  }

  /**
   * @brief Deallocates storage pointed to by `p` using `cudaFreeAsync`.
   *
   * @param p Pointer to memory to deallocate
   * @param stream The stream to order the deallocation on
   */
  void deallocate(value_type* p, std::size_t, cuda::stream_ref stream)
  {
    CUCO_CUDA_TRY(cudaFreeAsync(p, stream.get()));
  }
};

/**
 * @brief Equality comparison operator.
 *
 * @tparam T Value type of LHS object
 * @tparam U Value type of RHS object
 *
 * @return `true` iff given arguments are equal
 */
template <typename T, typename U>
bool operator==(cuda_allocator<T> const&, cuda_allocator<U> const&) noexcept
{
  return true;
}

/**
 * @brief Inequality comparison operator.
 *
 * @tparam T Value type of LHS object
 * @tparam U Value type of RHS object
 *
 * @param lhs Left-hand side object to compare
 * @param rhs Right-hand side object to compare
 *
 * @return `true` iff given arguments are not equal
 */
template <typename T, typename U>
bool operator!=(cuda_allocator<T> const& lhs, cuda_allocator<U> const& rhs) noexcept
{
  return not(lhs == rhs);
}

}  // namespace cuco
