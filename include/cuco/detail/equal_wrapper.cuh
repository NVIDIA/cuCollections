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

#include <cuco/detail/bitwise_compare.cuh>

#include <cstddef>

namespace cuco {
namespace experimental {
namespace detail {

/**
 * @brief Enum of equality comparison results.
 */
enum class equal_result : int32_t { UNEQUAL = 0, EMPTY = 1, EQUAL = 2 };

/**
 * @brief Key equality wrapper.
 *
 * User-provided equality binary callable cannot be used to compare against sentinel value.
 *
 * @tparam T Right-hand side Element type
 * @tparam Equal Type of user-provided equality binary callable
 */
template <typename T, typename Equal>
struct equal_wrapper {
  T empty_sentinel_;  ///< Sentinel value
  Equal equal_;       ///< Custom equality callable

  /**
   * @brief Equality wrapper ctor.
   *
   * @param sentinel Sentinel value
   * @param equal Equality binary callable
   */
  __host__ __device__ constexpr equal_wrapper(T sentinel, Equal const& equal) noexcept
    : empty_sentinel_{sentinel}, equal_{equal}
  {
  }

  /**
   * @brief Equality check with the given equality callable.
   *
   * @tparam U Right-hand side Element type
   *
   * @param lhs Left-hand side element to check equality
   * @param rhs Right-hand side element to check equality
   *
   * @return `EQUAL` if `lhs` and `rhs` are equivalent. `UNEQUAL` otherwise.
   */
  template <typename U>
  __device__ constexpr equal_result equal_to(T const& lhs, U const& rhs) const noexcept
  {
    return equal_(lhs, rhs) ? equal_result::EQUAL : equal_result::UNEQUAL;
  }

  /**
   * @brief Order-sensitive equality operator.
   *
   * @note This function always compares the left-hand side element against `empty_sentinel_` value
   * first then perform a equality check with the given `equal_` callable, i.e., `equal_(lhs, rhs)`.
   * @note Container (like set or map) keys MUST be always on the left-hand side.
   *
   * @tparam U Right-hand side Element type
   *
   * @param lhs Left-hand side element to check equality
   * @param rhs Right-hand side element to check equality
   *
   * @return Three way equality comparison result
   */
  template <typename U>
  __device__ constexpr equal_result operator()(T const& lhs, U const& rhs) const noexcept
  {
    return cuco::detail::bitwise_compare(lhs, empty_sentinel_) ? equal_result::EMPTY
                                                               : this->equal_to(lhs, rhs);
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
