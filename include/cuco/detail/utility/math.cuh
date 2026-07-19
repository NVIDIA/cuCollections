/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/type_traits>

namespace cuco {
namespace detail {

/**
 * @brief Ceiling of an integer division
 *
 * @tparam T Type of dividend
 * @tparam U Type of divisor
 *
 * @throw If `T` is not an integral type
 * @throw If `U` is not an integral type
 *
 * @param dividend Numerator
 * @param divisor Denominator
 *
 * @return Ceiling of the integer division
 */
template <typename T, typename U>
__host__ __device__ constexpr T int_div_ceil(T dividend, U divisor) noexcept
{
  static_assert(cuda::std::is_integral_v<T>);
  static_assert(cuda::std::is_integral_v<U>);
  return (dividend + divisor - 1) / divisor;
}

}  // namespace detail
}  // namespace cuco
