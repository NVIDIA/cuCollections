/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
 */

#pragma once

namespace cuco::detail {

/**
 * @brief A strong type wrapper
 *
 * @tparam T Type of the value
 */
template <typename T>
struct strong_type {
  /**
   * @brief Constructs a strong type
   *
   * @param v Value to be wrapped as a strong type
   */
  __host__ __device__ explicit constexpr strong_type(T v) : value{v} {}

  /**
   * @brief Implicit conversion operator to the underlying value.
   *
   * @return Underlying value
   */
  __host__ __device__ constexpr operator T() const noexcept { return value; }

  T value;  ///< Underlying data value
};

}  // namespace cuco::detail
