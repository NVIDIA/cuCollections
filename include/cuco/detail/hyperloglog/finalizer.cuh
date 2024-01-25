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

#include <cuco/detail/hyperloglog/tuning.cuh>

#include <cuda/std/cmath>

namespace cuco::hyperloglog_ns::detail {

/**
 * @brief Estimate correction algorithm based on HyperLogLog++.
 *
 * @note Variable names correspond to the definitions given in the HLL++ paper:
 * https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf
 *
 * @tparam Precision Tuning parameter to trade accuracy for runtime/memory footprint
 */
template <int32_t Precision>
class finalizer {
  // this minimum number of registers is required by HLL++
  static_assert(Precision >= 4, "Precision must be greater or equal to 4");

 public:
  /**
   * @brief Compute the bias-corrected cardinality estimate.
   *
   * @param z Geometric mean of registers
   * @param v Number of 0 registers
   *
   * @return Bias-corrected cardinality estimate
   */
  __host__ __device__ static double constexpr finalize(double z, int v) noexcept
  {
    auto e = alpha_mm() / z;
    // TODO remove test code
    // printf("raw e: %lf\n", e);

    if (v > 0) {
      // Use linear counting for small cardinality estimates.
      double const h = m * log(static_cast<double>(m) / v);
      // HLL++ is defined only when p < 19, otherwise we need to fallback to HLL.
      // The threshold `2.5 * m` is from the original HLL algorithm.
      if ((Precision < 19 and h <= thresholds[Precision - 4]) or e <= 2.5 * m) {
        e = h;
      } else {
        e = bias_corrected_estimate(e);
      }
    } else {
      e = bias_corrected_estimate(e);
    }

    return cuda::std::round(e);
  }

 private:
  static auto constexpr m = (1 << Precision);  ///< Number of registers

  __host__ __device__ static double constexpr alpha_mm() noexcept
  {
    if constexpr (m == 16) {
      return 0.673 * m * m;
    } else if constexpr (m == 32) {
      return 0.697 * m * m;
    } else if constexpr (m == 64) {
      return 0.709 * m * m;
    } else {
      return (0.7213 / (1.0 + 1.079 / m)) * m * m;
    }
  }

  __host__ __device__ static double constexpr bias_corrected_estimate(double e) noexcept
  {
    if constexpr (Precision < 19) {
      if (e < 5.0 * m) { return e - bias(e); }
    }
    return e;
  }

  // TODO implement HLL++ bias correction
  __host__ __device__ static double constexpr bias(double e) noexcept { return e * 0; }
};
}  // namespace cuco::hyperloglog_ns::detail