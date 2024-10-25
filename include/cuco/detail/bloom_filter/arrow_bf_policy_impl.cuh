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

#include <cuco/detail/error.hpp>
#include <cuco/hash_functions.cuh>

#include <cuda/std/bit>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cstdint>
#include <nv/target>

namespace cuco::detail {

template <class T>
class arrow_bf_policy_impl {
 public:
  using hasher             = cuco::xxhash_64<T>;
  using word_type          = std::uint32_t;
  using hash_argument_type = typename hasher::argument_type;
  using hash_result_type   = decltype(std::declval<hasher>()(std::declval<hash_argument_type>()));

  // Bytes in a tiny BF block.
  static constexpr int bytes_per_filter_block       = 32;  // hardcoded values from Arrow BF
  static constexpr uint32_t bits_set_per_block      = 8;   // hardcoded values from Arrow BF
  static constexpr uint32_t words_per_block         = 8;   // hardcoded values from Arrow BF
  static constexpr std::uint32_t min_arrow_bf_bytes = 32;
  static constexpr std::uint32_t max_arrow_bf_bytes = 128 * 1024 * 1024;

  __host__ __device__ explicit constexpr arrow_bf_policy_impl(std::uint32_t num_blocks, hasher hash)
    : hash_{hash}
  {
    NV_DISPATCH_TARGET(
      NV_IS_HOST,
      (CUCO_EXPECTS(num_blocks >= 1 and num_blocks <= (max_arrow_bf_bytes / bytes_per_filter_block),
                    "`num_blocks` must be in the range of [1, 4194304]");),
      NV_IS_DEVICE,
      (if (num_blocks < 1 or num_blocks > (max_arrow_bf_bytes / bytes_per_filter_block)) {
        __trap();  // TODO this kills the kernel and corrupts the CUDA context. Not ideal.
      }));
  }

  __device__ constexpr hash_result_type hash(hash_argument_type const& key) const
  {
    return hash_(key);
  }

  template <class Extent>
  __device__ constexpr auto block_index(hash_result_type hash, Extent num_blocks) const
  {
    constexpr auto hash_bits = cuda::std::numeric_limits<uint32_t>::digits;
    return static_cast<uint32_t>(((hash >> hash_bits) * num_blocks) >> hash_bits);
  }

  __device__ constexpr word_type word_pattern(hash_result_type hash, std::uint32_t word_index) const
  {
    // The block-based algorithm needs eight odd SALT values to calculate eight indexes of bit to
    // set, one bit in each 32-bit word.

    // @MH: Fix this
    constexpr std::uint32_t SALT[bits_set_per_block] = {0x47b6137bU,
                                                        0x44974d91U,
                                                        0x8824ad5bU,
                                                        0xa2b7289dU,
                                                        0x705495c7U,
                                                        0x2df1424bU,
                                                        0x9efc4947U,
                                                        0x5c6bfb31U};
    word_type const key                              = static_cast<uint32_t>(hash);
    return word_type{1} << ((key * SALT[word_index]) >> 27);
  }

 private:
  hasher hash_;
};

}  // namespace cuco::detail