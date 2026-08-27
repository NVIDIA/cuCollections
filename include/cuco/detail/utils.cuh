/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/__config>

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

template <typename SizeType, typename HashType>
__host__ __device__ constexpr cuda::std::make_unsigned_t<SizeType> to_positive(HashType hash)
{
  using unsigned_size_type = cuda::std::make_unsigned_t<SizeType>;
  auto const value         = static_cast<unsigned_size_type>(hash);

  if constexpr (cuda::std::is_signed_v<SizeType>) {
    auto constexpr max =
      static_cast<unsigned_size_type>(cuda::std::numeric_limits<SizeType>::max());
    return value > max ? unsigned_size_type{0} - value : value;
  } else {
    return value;
  }
}

/**
 * @brief Converts a hash value into a valid index for the given modulus.
 *
 * @note Hash values wider than `SizeType` are narrowed before reduction, preserving the existing
 * low-bit mapping policy.
 *
 * @tparam SizeType The target type
 * @tparam HashType The input type
 *
 * @param hash The hash value
 * @param modulus Exclusive upper bound for the returned index
 *
 * @return An index in `[0, modulus)`
 */
template <typename SizeType, typename HashType>
__host__ __device__ constexpr SizeType sanitize_hash(HashType hash, SizeType modulus) noexcept
{
  using unsigned_size_type = cuda::std::make_unsigned_t<SizeType>;

  unsigned_size_type magnitude;
  if constexpr (cuda::std::is_same_v<HashType, cuda::std::array<std::uint64_t, 2>>) {
#if !defined(CUCO_HAS_INT128)
    static_assert(false,
                  "CUCO_HAS_INT128 undefined. Need unsigned __int128 type when sanitizing "
                  "cuda::std::array<std::uint64_t, 2>");
#endif
    unsigned __int128 ret{};
    memcpy(&ret, &hash, sizeof(unsigned __int128));
    magnitude = to_positive<SizeType>(ret);
  } else {
    magnitude = to_positive<SizeType>(hash);
  }

  return static_cast<SizeType>(magnitude % static_cast<unsigned_size_type>(modulus));
}

}  // namespace detail
}  // namespace cuco
