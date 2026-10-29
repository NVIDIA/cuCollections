/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/__config>

#include <cuda/cmath>
#include <cuda/std/array>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace cuco {
namespace detail {

/**
 * @brief For the `n` least significant bits in the given unsigned 32-bit integer `x`,
 * returns the number of set bits.
 */
__device__ __forceinline__ cuda::std::int32_t count_least_significant_bits(cuda::std::uint32_t x,
                                                                           cuda::std::int32_t n)
{
  return __popc(x & (1 << n) - 1);
}

/**
 * @brief Maps an unsigned hash bit pattern into `[0, upper_bound)` using multiply-high.
 *
 * @note Hash bits should be uniformly distributed across the hash type's width. This is range
 * reduction, not a remainder operation. A 128-bit hash uses only its low 64-bit word (the first
 * element for a two-word hash).
 *
 * @tparam SizeType Integral index type
 * @tparam HashType Integral hash type of at most 128 bits, or a two-word 128-bit hash
 *
 * @param hash The hash value
 * @param upper_bound Positive exclusive upper bound for the returned index
 *
 * @return An index in `[0, upper_bound)`
 */
template <typename SizeType, typename HashType>
__host__ __device__ constexpr SizeType reduce_hash(HashType hash, SizeType upper_bound) noexcept
{
  if constexpr (cuda::std::is_same_v<HashType, cuda::std::array<std::uint64_t, 2>>) {
    return reduce_hash<SizeType>(hash[0], upper_bound);
  } else if constexpr (sizeof(HashType) > sizeof(cuda::std::uint64_t)) {
    static_assert(cuda::std::is_integral_v<HashType> and sizeof(HashType) == 16);
    return reduce_hash<SizeType>(static_cast<cuda::std::uint64_t>(hash), upper_bound);
  } else if constexpr (sizeof(HashType) == 4 and sizeof(SizeType) == 8) {
    auto const bits  = static_cast<cuda::std::uint32_t>(hash);
    auto const bound = static_cast<cuda::std::uint64_t>(upper_bound);
    // Decompose the 96-bit product without introducing a full 64-bit high multiply.
    return static_cast<SizeType>(cuda::mul_hi(bits, static_cast<cuda::std::uint32_t>(bound)) +
                                 static_cast<cuda::std::uint64_t>(bits) *
                                   static_cast<cuda::std::uint32_t>(bound >> 32));
  } else {
    using unsigned_hash_type = cuda::std::make_unsigned_t<HashType>;
    using unsigned_size_type = cuda::std::make_unsigned_t<SizeType>;
    using product_type       = cuda::std::
      conditional_t<(sizeof(HashType) > sizeof(SizeType)), unsigned_hash_type, unsigned_size_type>;
    static_assert(sizeof(product_type) <= sizeof(cuda::std::uint64_t));
    constexpr auto shift = cuda::std::numeric_limits<product_type>::digits -
                           cuda::std::numeric_limits<unsigned_hash_type>::digits;
    // Left-align a narrower hash so widening does not shrink the range of initial indices.
    auto const bits = static_cast<product_type>(static_cast<unsigned_hash_type>(hash)) << shift;
    return static_cast<SizeType>(
      cuda::mul_hi(static_cast<product_type>(bits), static_cast<product_type>(upper_bound)));
  }
}

}  // namespace detail
}  // namespace cuco
