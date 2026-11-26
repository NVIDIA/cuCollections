/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/__config>
#include <cuco/detail/pair/helpers.cuh>

#include <cuda/cmath>
#include <cuda/memory>
#include <cuda/std/array>
#include <cuda/std/cstddef>
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
 * @brief Reinterprets a word as an object of the same size.
 *
 * Uses the same reinterpretation as `packed_cas`, so it also accepts bitwise-comparable types that
 * are not trivially copyable or not default constructible.
 *
 * @tparam T Type of the object
 * @tparam Word Type of the word, of the same size as `T`
 *
 * @param word The word that holds the object representation
 *
 * @return A copy of the object
 */
template <typename T, typename Word>
[[nodiscard]] __device__ T object_from_word(Word const& word) noexcept
{
  static_assert(sizeof(T) == sizeof(Word), "The word must have the size of the object");
  return *reinterpret_cast<T const*>(&word);
}

/// Unsigned word of `Unit` bytes (4, 8, or 16)
template <cuda::std::size_t Unit>
using load_word_t = typename packed<Unit>::type;

/**
 * @brief Bucket alignment in bytes that a storage ref guarantees.
 *
 * `StorageRef::alignment` if the storage ref declares it, else the alignment of its slot type.
 *
 * @tparam StorageRef Storage ref type
 */
template <typename StorageRef, typename = void>
struct storage_alignment
  : cuda::std::integral_constant<cuda::std::size_t, alignof(typename StorageRef::value_type)> {};

template <typename StorageRef>
struct storage_alignment<StorageRef, cuda::std::void_t<decltype(StorageRef::alignment)>>
  : cuda::std::integral_constant<cuda::std::size_t, StorageRef::alignment> {};

#define CUCO_DETAIL_WEAK_LD(SPACE, PTR)                                                 \
  if constexpr (Unit == 4 and Bytes == 4) {                                             \
    asm("ld" SPACE ".b32 %0, [%1];" : "=r"(words[0]) : "l"(PTR) : "memory");            \
  } else if constexpr (Unit == 4 and Bytes == 8) {                                      \
    asm("ld" SPACE ".v2.b32 {%0, %1}, [%2];"                                            \
        : "=r"(words[0]), "=r"(words[1])                                                \
        : "l"(PTR)                                                                      \
        : "memory");                                                                    \
  } else if constexpr (Unit == 4 and Bytes == 16) {                                     \
    asm("ld" SPACE ".v4.b32 {%0, %1, %2, %3}, [%4];"                                    \
        : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3])                \
        : "l"(PTR)                                                                      \
        : "memory");                                                                    \
  } else if constexpr (Unit == 8 and Bytes == 8) {                                      \
    asm("ld" SPACE ".b64 %0, [%1];" : "=l"(words[0]) : "l"(PTR) : "memory");            \
  } else if constexpr (Unit == 8 and Bytes == 16) {                                     \
    asm("ld" SPACE ".v2.b64 {%0, %1}, [%2];"                                            \
        : "=l"(words[0]), "=l"(words[1])                                                \
        : "l"(PTR)                                                                      \
        : "memory");                                                                    \
  } else {                                                                              \
    static_assert(Unit == 16 and Bytes == 16, "Unsupported weak load shape");           \
    cuda::std::uint64_t lo;                                                             \
    cuda::std::uint64_t hi;                                                             \
    asm("{\n\t.reg .b128 w;\n\tld" SPACE ".b128 w, [%2];\n\tmov.b128 {%0, %1}, w;\n\t}" \
        : "=l"(lo), "=l"(hi)                                                            \
        : "l"(PTR)                                                                      \
        : "memory");                                                                    \
    words[0] = (static_cast<load_word_t<16>>(hi) << 64) | lo;                           \
  }

/**
 * @brief Loads `Bytes` bytes at `address` with one weak PTX load whose elements are `Unit` bytes.
 *
 * The PTX memory model treats a vector load as one load per element. Each element must be exactly
 * the word that atomic operations on this memory swap or store. Then a race with such an operation
 * is a uniform-size race, which the memory model defines. The compiler cannot split, widen, or
 * merge the asm.
 *
 * @tparam Unit Element width in bytes: 4, 8, or 16
 * @tparam Bytes Number of bytes to load: 4, 8, or 16, a multiple of `Unit`
 *
 * @param address Address of the first byte, aligned to `Bytes`
 * @param global Whether `address` points to global memory; otherwise a generic load is used
 * @param words The loaded elements
 */
template <cuda::std::size_t Unit, cuda::std::size_t Bytes>
__device__ void weak_load_words(void const* address,
                                bool global,
                                load_word_t<Unit> (&words)[Bytes / Unit]) noexcept
{
  static_assert(Bytes % Unit == 0, "Bytes must be a multiple of Unit");
  if constexpr (Bytes == 32) {
    // 256-bit loads exist for global memory on sm_100+ only (PTX ISA 8.8).
    static_assert(Unit == 4 or Unit == 8, "A 32-byte load has 4- or 8-byte elements");
    if (global) {
      auto const ptr = __cvta_generic_to_global(address);
      if constexpr (Unit == 4) {
        asm("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(words[0]),
              "=r"(words[1]),
              "=r"(words[2]),
              "=r"(words[3]),
              "=r"(words[4]),
              "=r"(words[5]),
              "=r"(words[6]),
              "=r"(words[7])
            : "l"(ptr)
            : "memory");
      } else {
        asm("ld.global.v4.b64 {%0, %1, %2, %3}, [%4];"
            : "=l"(words[0]), "=l"(words[1]), "=l"(words[2]), "=l"(words[3])
            : "l"(ptr)
            : "memory");
      }
    } else {
      constexpr auto half     = 16 / Unit;
      auto const* const bytes = static_cast<char const*>(address);
      weak_load_words<Unit, 16>(bytes, false, *reinterpret_cast<load_word_t<Unit>(*)[half]>(words));
      weak_load_words<Unit, 16>(
        bytes + 16, false, *reinterpret_cast<load_word_t<Unit>(*)[half]>(words + half));
    }
  } else if (global) {
    auto const ptr = __cvta_generic_to_global(address);
    CUCO_DETAIL_WEAK_LD(".global", ptr)
  } else {
    auto const ptr = reinterpret_cast<cuda::std::uintptr_t>(address);
    CUCO_DETAIL_WEAK_LD("", ptr)
  }
}

#undef CUCO_DETAIL_WEAK_LD

/**
 * @brief Loads the object at `address` while other threads may be swapping it atomically.
 *
 * A plain load must not race with an atomic operation: the compiler may narrow, widen, merge, or
 * drop it. The object is read with one weak PTX load of exactly the word the atomic operation
 * swaps instead, which is the object itself or the aligned 32-bit word that holds a 1- or 2-byte
 * object. An atomic load is not used because it is several times slower when many threads probe
 * the same slots.
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

  bool const global = cuda::device::is_address_from(address, cuda::device::address_space::global);
  if constexpr (sizeof(T) <= 2) {
    constexpr cuda::std::uintptr_t align_mask = sizeof(cuda::std::uint32_t) - 1;
    auto const address_value                  = reinterpret_cast<cuda::std::uintptr_t>(address);
    auto const offset = static_cast<cuda::std::uint32_t>((address_value & align_mask) * 8);
    load_word_t<4> word[1];
    weak_load_words<4, 4>(reinterpret_cast<void const*>(address_value & ~align_mask), global, word);
    return object_from_word<T>(static_cast<word_type>(word[0] >> offset));
  } else {
    load_word_t<sizeof(T)> word[1];
    weak_load_words<sizeof(T), sizeof(T)>(address, global, word);
    return object_from_word<T>(word[0]);
  }
}

}  // namespace detail
}  // namespace cuco
