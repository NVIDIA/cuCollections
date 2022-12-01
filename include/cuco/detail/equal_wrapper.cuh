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
 * @brief Equality wrapper.
 *
 * User-provided equality binary callable cannot be used to compared against sentinel value.
 *
 * @tparam T Right-hand side Element type
 * @tparam Equal Type of user-provided equality binary callable
 */
template <typename T, typename Equal>
struct equal_wrapper {
  T sentinel_;   ///< Sentinel value
  Equal equal_;  ///< Custom equality callable

  /**
   * @brief Equality wrapper ctor.
   *
   * @param sentinel Sentinel value
   * @param equal Equality binary callable
   */
  __host__ __device__ equal_wrapper(T const sentinel, Equal const& equal)
    : sentinel_{sentinel}, equal_{equal}
  {
  }

  /**
   * @brief Equality operator.
   *
   * @tparam U Left-hand side Element type
   *
   * @param lhs Left-hand side element to check equality
   * @param rhs Right-hand side element to check equality
   * @return Equality comparison result
   */
  template <typename U>
  __device__ inline equal_result operator()(T const& lhs, U const& rhs) const noexcept
  {
    return cuco::detail::bitwise_compare(lhs, sentinel_)
             ? equal_result::EMPTY
             : ((equal_(lhs, rhs)) ? equal_result::EQUAL : equal_result::UNEQUAL);
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco