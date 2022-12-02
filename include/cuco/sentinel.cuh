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

#include <cuco/detail/utils.cuh>

namespace cuco {
inline namespace sentinel {

/**
 * @brief A strong type wrapper used to denote the empty key sentinel.
 *
 * @tparam T Type of the key values
 */
template <typename T>
struct empty_key : public cuco::detail::strong_type<T> {
  /**
   * @brief Constructs an empty key sentinel with the given `v`.
   *
   * @param v The empty key sentinel value
   */
  __host__ __device__ explicit constexpr empty_key(T v) : cuco::detail::strong_type<T>(v) {}
};

/**
 * @brief A strong type wrapper used to denote the empty value sentinel.
 *
 * @tparam T Type of the mapped values
 */
template <typename T>
struct empty_value : public cuco::detail::strong_type<T> {
  /**
   * @brief Constructs an empty value sentinel with the given `v`.
   *
   * @param v The empty value sentinel value
   */
  __host__ __device__ explicit constexpr empty_value(T v) : cuco::detail::strong_type<T>(v) {}
};

/**
 * @brief A strong type wrapper used to denote the erased key sentinel.
 *
 * @tparam T Type of the key values
 */
template <typename T>
struct erased_key : public cuco::detail::strong_type<T> {
  /**
   * @brief Constructs an erased key sentinel with the given `v`.
   *
   * @param v The erased key sentinel value
   */
  __host__ __device__ explicit constexpr erased_key(T v) : cuco::detail::strong_type<T>(v) {}
};

}  // namespace sentinel
}  // namespace cuco
