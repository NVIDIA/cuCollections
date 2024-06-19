/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
 */

#pragma once

#include <cuco/detail/bitwise_compare.cuh>

#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <thrust/tuple.h>

#include <cstddef>

namespace cuco {
namespace detail {

/**
 * @brief For the `n` least significant bits in the given unsigned 32-bit integer `x`,
 * returns the number of set bits.
 */
__device__ __forceinline__ int32_t count_least_significant_bits(uint32_t x, int32_t n)
{
  return __popc(x & (1 << n) - 1);
}

/**
 * @brief Converts pair to `thrust::tuple` to allow assigning to a zip iterator.
 *
 * @tparam Key The slot key type
 * @tparam Value The slot value type
 */
template <typename Key, typename Value>
struct slot_to_tuple {
  /**
   * @brief Converts a pair to a `thrust::tuple`.
   *
   * @tparam S The slot type
   *
   * @param s The slot to convert
   * @return A thrust::tuple containing `s.first` and `s.second`
   */
  template <typename S>
  __device__ thrust::tuple<Key, Value> operator()(S const& s)
  {
    return thrust::tuple<Key, Value>(s.first, s.second);
  }
};

/**
 * @brief Device functor returning whether the input slot `s` is filled.
 *
 * @tparam Key The slot key type
 */
template <typename Key>
struct slot_is_filled {
  Key empty_key_sentinel_;  ///< The value of the empty key sentinel

  /**
   * @brief Indicates if the target slot `s` is filled.
   *
   * @tparam S The slot type
   *
   * @param s The slot to query
   * @return `true` if slot `s` is filled
   */
  template <typename S>
  __device__ bool operator()(S const& s)
  {
    return not cuco::detail::bitwise_compare(thrust::get<0>(s), empty_key_sentinel_);
  }
};

template <typename SizeType, typename HashType>
__host__ __device__ constexpr SizeType to_positive(HashType hash)
{
  if constexpr (cuda::std::is_signed_v<SizeType>) {
    return cuda::std::abs(static_cast<SizeType>(hash));
  } else {
    return static_cast<SizeType>(hash);
  }
}

/**
 * @brief Converts a given hash value into a valid (positive) size type.
 *
 * @tparam SizeType The target type
 * @tparam HashType The input type
 *
 * @return Converted hash value
 */
template <typename SizeType, typename HashType>
__host__ __device__ constexpr SizeType sanitize_hash(HashType hash) noexcept
{
  if constexpr (cuda::std::is_same_v<HashType, cuda::std::array<std::uint64_t, 2>>) {
#if !defined(CUCO_HAS_INT128)
    static_assert(false,
                  "CUCO_HAS_INT128 undefined. Need unsigned __int128 type when sanitizing "
                  "cuda::std::array<std::uint64_t, 2>");
#endif
    unsigned __int128 ret{};
    memcpy(&ret, &hash, sizeof(unsigned __int128));
    return to_positive<SizeType>(static_cast<SizeType>(ret));
  } else {
    return to_positive<SizeType>(hash);
  }
}

/**
 * @brief Converts a given hash value and cg_rank, into a valid (positive) size type.
 *
 * @tparam SizeType The target type
 * @tparam CG Cooperative group type
 * @tparam HashType The input type
 *
 * @return Converted hash value
 */
template <typename SizeType, typename CG, typename HashType>
__device__ constexpr SizeType sanitize_hash(CG const& group, HashType hash) noexcept
{
  auto const base_hash = sanitize_hash<SizeType>(hash);
  auto const max_size  = cuda::std::numeric_limits<SizeType>::max();
  auto const cg_rank   = static_cast<SizeType>(group.thread_rank());

  if (base_hash > (max_size - cg_rank)) { return cg_rank - (max_size - base_hash); }
  return base_hash + cg_rank;
}

}  // namespace detail
}  // namespace cuco
