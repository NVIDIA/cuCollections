/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 * Copyright (c) 2022, Jim Apple.
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

#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include <cstdint>

namespace cuco::detail {

/**
 * @brief Computes the expected false-positive rate of a blocked Bloom filter
 * using the Poisson-based formula (Eq. 3) from Putze et.al. "Cache-, Hash- and Space-Efficient
 * Bloom Filters".
 *
 * Reference implementation:
 * https://github.com/jbapple/libfilter/blob/4ebeaef1215969aee9edb05eb145e94b8dd98e16/c/lib/util.c#L5
 *
 * @param ndv Number of distinct inserted elements
 * @param bytes Filter size in bytes
 * @param word_bits Number of bits in the underlying word type of a filter block
 * @param block_words Number of words in each filter block
 * @param hash_bits Total number of bits in the hash value type
 * @param k Number of pattern bits to set for a key
 * @param max_iters Maximum number of iterations for accuracy refinement
 *
 * @return Approximation of the expected false-positive rate
 */
__host__ inline double blocked_bloom_filter_expected_fpr(double ndv,
                                                         double bytes,
                                                         double word_bits,
                                                         double block_words,
                                                         double hash_bits,
                                                         double k,
                                                         std::uint64_t max_iters = 1000)
{
  if (ndv == 0) return 0.0;
  if (bytes <= 0) return 1.0;
  if (ndv / (bytes * cuda::std::numeric_limits<std::uint8_t>::digits) >= 2.0) return 1.0;

  double result = 0;
  double const lam =
    block_words * word_bits / ((bytes * cuda::std::numeric_limits<std::uint8_t>::digits) / ndv);
  double const loglam      = cuda::std::log(lam);
  double const log1collide = -hash_bits * cuda::std::log(2.0);

  for (std::uint64_t j = 0; j < max_iters; ++j) {
    double const i         = static_cast<double>(max_iters - 1 - j);
    double const logp      = i * loglam - lam - cuda::std::lgamma(i + 1.0);
    double const logfinner = k * cuda::std::log(1.0 - cuda::std::pow(1.0 - 1.0 / word_bits, i * k));
    double const logcollide = cuda::std::log(i) + log1collide;
    result += cuda::std::exp(logp + logfinner) + cuda::std::exp(logp + logcollide);
    // result += exp(logp + logfinner); // alternative approach
  }

  return (result > 1.0) ? 1.0 : result;
}

}  // namespace cuco::detail