/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/__config>
#include <cuco/detail/pair/helpers.cuh>

#include <cuda/cmath>
#include <cuda/memory>
#include <cuda/ptx>
#include <cuda/std/array>
#include <cuda/std/bit>
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
 * reduction, not a remainder operation. A 128-bit hash contributes its high 64 bits (the second
 * element of a two-word hash), which are then reduced like a 64-bit hash.
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
__host__ __device__ constexpr SizeType hash_to_index(HashType hash, SizeType upper_bound) noexcept
{
  using unsigned_size_type   = cuda::std::make_unsigned_t<SizeType>;
  constexpr auto hash_bytes  = sizeof(HashType);
  constexpr auto index_bytes = sizeof(SizeType);
  if constexpr (cuda::std::is_same_v<HashType, cuda::std::array<std::uint64_t, 2>>) {
    return hash_to_index<SizeType>(hash[1], upper_bound);
  } else if constexpr (hash_bytes > 8) {
    static_assert(cuda::std::is_integral_v<HashType> and hash_bytes == 16);
    using unsigned_hash_type = cuda::std::make_unsigned_t<HashType>;
    return hash_to_index<SizeType>(
      static_cast<std::uint64_t>(static_cast<unsigned_hash_type>(hash) >> 64), upper_bound);
  } else if constexpr (hash_bytes == 4 and index_bytes == 8) {
    auto const bits  = static_cast<cuda::std::uint32_t>(hash);
    auto const bound = static_cast<cuda::std::uint64_t>(upper_bound);
    // Decompose the 96-bit product without introducing a full 64-bit high multiply.
    return static_cast<SizeType>(cuda::mul_hi(bits, static_cast<cuda::std::uint32_t>(bound)) +
                                 static_cast<cuda::std::uint64_t>(bits) *
                                   static_cast<cuda::std::uint32_t>(bound >> 32));
  } else if constexpr (hash_bytes == 8 and index_bytes == 4) {
    auto const bits  = static_cast<cuda::std::uint64_t>(hash);
    auto const bound = static_cast<cuda::std::uint32_t>(upper_bound);
    // Only the upper 32 bits of the 96-bit product are needed.
    auto const middle =
      static_cast<cuda::std::uint64_t>(static_cast<cuda::std::uint32_t>(bits >> 32)) * bound +
      cuda::mul_hi(static_cast<cuda::std::uint32_t>(bits), bound);
    return static_cast<SizeType>(middle >> 32);
  } else {
    using unsigned_hash_type = cuda::std::make_unsigned_t<HashType>;
    using product_type =
      cuda::std::conditional_t<(hash_bytes > index_bytes), unsigned_hash_type, unsigned_size_type>;
    static_assert(sizeof(product_type) <= 8);
    constexpr auto shift = cuda::std::numeric_limits<product_type>::digits -
                           cuda::std::numeric_limits<unsigned_hash_type>::digits;
    // Left-align a narrower hash so widening does not shrink the range of initial indices.
    auto const bits = static_cast<product_type>(static_cast<unsigned_hash_type>(hash)) << shift;
    return static_cast<SizeType>(
      cuda::mul_hi(static_cast<product_type>(bits), static_cast<product_type>(upper_bound)));
  }
}

/**
 * @brief Loads the object at `address` while other threads may be swapping it atomically.
 *
 * A plain load must not race with an atomic operation: the compiler may narrow, widen, merge, or
 * drop it. An object in global memory is read with one weak PTX load of exactly the word the
 * atomic operation swaps instead, which is the object itself or the aligned 32-bit word that holds
 * a 1- or 2-byte object. An atomic load is not used there because it is several times slower when
 * many threads probe the same slots.
 *
 * The value may already be stale when the caller sees it. Re-read it only after an atomic
 * operation: a loop that only waits on this load may never observe a change.
 *
 * @tparam T Type of the object, 1, 2, 4, 8, or 16 bytes in size
 *
 * @param address Address of the object
 *
 * @return The object held by the swapped word
 */
template <typename T>
[[nodiscard]] __device__ T weak_load(T const* address) noexcept
{
  using word_type = packed_t<T>;
  static_assert(not cuda::std::is_void_v<word_type>,
                "A weak load requires a 1, 2, 4, 8, or 16 byte type");

  // `cuda::ptx::ld` reads global memory only, so an object in other memory, such as shared memory,
  // takes a volatile load.
  if (not cuda::device::is_address_from(address, cuda::device::address_space::global)) {
    word_type const word = *reinterpret_cast<word_type const volatile*>(address);
    return cuda::std::bit_cast<T>(word);
  }

  if constexpr (sizeof(T) <= 2) {
    constexpr cuda::std::uintptr_t align_mask = sizeof(cuda::std::uint32_t) - 1;
    auto const address_value                  = reinterpret_cast<cuda::std::uintptr_t>(address);
    auto const* const aligned =
      reinterpret_cast<cuda::std::uint32_t const*>(address_value & ~align_mask);
    auto const offset = static_cast<cuda::std::uint32_t>((address_value & align_mask) * 8);
    auto const word   = cuda::ptx::ld(cuda::ptx::space_global, aligned);
    return cuda::std::bit_cast<T>(static_cast<word_type>(word >> offset));
  } else {
    auto const* const word = reinterpret_cast<word_type const*>(address);
    return cuda::std::bit_cast<T>(cuda::ptx::ld(cuda::ptx::space_global, word));
  }
}

}  // namespace detail
}  // namespace cuco
