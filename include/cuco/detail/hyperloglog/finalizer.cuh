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

#include <cstddef>
#include <cuda/std/cmath>
#include <cuda/std/limits>

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
  // Note: Most of the types in this implementation are explicit instead of relying on `auto` to
  // avoid confusion with the reference implementation.

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
  __host__ __device__ static std::size_t constexpr finalize(double z, int v) noexcept
  {
    auto e = alpha_mm() / z;

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
  static auto constexpr k = 6;                 ///< Number of interpolation points to consider

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

  __host__ __device__ static double constexpr bias(double e) noexcept
  {
    auto const anchor_index = interpolation_anchor_index(e);
    int const n             = raw_estimate_data<Precision>().size();

    auto low  = cuda::std::max(anchor_index - k + 1, 0);
    auto high = cuda::std::min(low + k, n);
    // Keep moving bounds as long as the (exclusive) high bound is closer to the estimate than
    // the lower (inclusive) bound.
    while (high < n and distance(e, high) < distance(e, low)) {
      low += 1;
      high += 1;
    }

    auto const& biases = bias_data<Precision>();
    double bias_sum    = 0.0;
    for (int i = low; i < high; ++i) {
      bias_sum += biases[i];
    }

    return bias_sum / (high - low);
  }

  __host__ __device__ static double distance(double e, int i) noexcept
  {
    auto const diff = e - raw_estimate_data<Precision>()[i];
    return diff * diff;
  }

  __host__ __device__ static int interpolation_anchor_index(double e) noexcept
  {
    auto const& estimates = raw_estimate_data<Precision>();
    int left              = 0;
    int right             = static_cast<int>(estimates.size()) - 1;
    int mid;
    int candidate_index = 0;  // Index of the closest element found

    while (left <= right) {
      mid = left + (right - left) / 2;

      if (estimates[mid] < e) {
        left = mid + 1;
      } else if (estimates[mid] > e) {
        right = mid - 1;
      } else {
        // Exact match found, no need to look further
        return mid;
      }
    }

    // At this point, 'left' is the insertion point. We need to compare the elements at 'left' and
    // 'left - 1' to find the closest one, taking care of boundary conditions.

    // Distance from 'e' to the element at 'left', if within bounds
    double const dist_lhs = left < static_cast<int>(estimates.size())
                              ? cuda::std::abs(estimates[left] - e)
                              : cuda::std::numeric_limits<double>::max();
    // Distance from 'e' to the element at 'left - 1', if within bounds
    double const dist_rhs = left - 1 >= 0 ? cuda::std::abs(estimates[left - 1] - e)
                                          : cuda::std::numeric_limits<double>::max();

    candidate_index = (dist_lhs < dist_rhs) ? left : left - 1;

    return candidate_index;
  }
};
}  // namespace cuco::hyperloglog_ns::detail