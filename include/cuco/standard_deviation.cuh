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
 * @brief Strong type for specifying the desired standard deviation of
 * cuco::distinct_count_estimator(_ref).
 */
class standard_deviation {
 public:
  /**
   * @brief Constructs a standard_deviation object.
   *
   * @param value The desired standard deviation
   */
  __host__ __device__ explicit constexpr standard_deviation(double value) noexcept : value_{value}
  {
  }

  /**
   * @brief Conversion to value type.
   *
   * @return Standard deviation
   */
  __host__ __device__ constexpr operator double() const noexcept { return this->value_; }

 private:
  double value_;  ///< Sketch size in KB
};
}  // namespace cuco