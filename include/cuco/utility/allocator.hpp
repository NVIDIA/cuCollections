/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cuco/detail/error.hpp>

namespace cuco {
/**
 * @brief A device allocator using `cudaMalloc`/`cudaFree` to satisfy (de)allocations.
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
   * @brief Allocates storage for `n` objects of type `T` using `cudaMalloc`.
   *
   * @param n The number of objects to allocate storage for
   * @return Pointer to the allocated storage
   */
  value_type* allocate(std::size_t n)
  {
    value_type* p;
    CUCO_CUDA_TRY(cudaMalloc(&p, sizeof(value_type) * n));
    return p;
  }

  /**
   * @brief Deallocates storage pointed to by `p`.
   *
   * @param p Pointer to memory to deallocate
   */
  void deallocate(value_type* p, std::size_t) { CUCO_CUDA_TRY(cudaFree(p)); }
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
