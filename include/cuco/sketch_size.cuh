/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

/**
 * @brief Strng type for specifying the sketch size of cuco::distinct_count_estimator(_ref) in KB.
 *
 * Values can also be given as literals, e.g., 64.3_KB
 */
class sketch_size_kb {
 public:
  /**
   * @brief Constructs a sketch_size_kb object.
   *
   * @param value The size of a sketch given in KB
   */
  __host__ __device__ explicit constexpr sketch_size_kb(double value) noexcept : value_{value} {}

  /**
   * @brief Conversion to value type.
   *
   * @return Sketch size in KB
   */
  __host__ __device__ constexpr operator double() const noexcept { return this->value_; }

 private:
  double value_;  ///< Sketch size in KB
};
}  // namespace cuco

// User-defined literal operators for sketch_size_KB
__host__ __device__ constexpr cuco::sketch_size_kb operator""_KB(long double value)
{
  return cuco::sketch_size_kb{static_cast<double>(value)};
}

__host__ __device__ constexpr cuco::sketch_size_kb operator""_KB(unsigned long long int value)
{
  return cuco::sketch_size_kb{static_cast<double>(value)};
}