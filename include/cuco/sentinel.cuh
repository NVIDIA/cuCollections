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

namespace cuco {
inline namespace sentinel {
/**
 * @brief A strong type wrapper used to denote the empty key sentinel.
 *
 * @tparam T Type of the key values
 */
template <typename T>
struct empty_key {
  /**
   * @brief Constructs an empty key sentinel with the given `v`.
   *
   * @param v The empty key sentinel value
   */
  __host__ __device__ explicit constexpr empty_key(T v) : value{v} {}

  /**
   * @brief Implicit conversion operator to the underlying value.
   *
   * @return Sentinel as underlying value type
   */
  __host__ __device__ constexpr operator T() const noexcept { return value; }

  T value;  ///< Empty key sentinel
};

/**
 * @brief A strong type wrapper used to denote the empty value sentinel.
 *
 * @tparam T Type of the mapped values
 */
template <typename T>
struct empty_value {
  /**
   * @brief Constructs an empty value sentinel with the given `v`.
   *
   * @param v The empty value sentinel value
   */
  __host__ __device__ explicit constexpr empty_value(T v) : value{v} {}

  /**
   * @brief Implicit conversion operator to the underlying value.
   *
   * @return Sentinel as underlying value type
   */
  __host__ __device__ constexpr operator T() const noexcept { return value; }

  T value;  ///< Empty value sentinel
};

/**
 * @brief A strong type wrapper used to denote the erased key sentinel.
 *
 * @tparam T Type of the key values
 */
template <typename T>
struct erased_key {
  /**
   * @brief Constructs an erased key sentinel with the given `v`.
   *
   * @param v The erased key sentinel value
   */
  __host__ __device__ explicit constexpr erased_key(T v) : value{v} {}

  /**
   * @brief Implicit conversion operator to the underlying value.
   *
   * @return Sentinel as underlying value type
   */
  __host__ __device__ constexpr operator T() const noexcept { return value; }

  T value;  ///< Erased key sentinel
};

}  // namespace sentinel
}  // namespace cuco
