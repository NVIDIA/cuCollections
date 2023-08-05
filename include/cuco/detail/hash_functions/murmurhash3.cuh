/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuco/detail/hash_functions/utils.cuh>
#include <cuco/extent.cuh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cuco::detail {

/**
 * @brief The 32bit integer finalizer hash function of `MurmurHash3`.
 *
 * @throw Key type must be 4 bytes in size
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct MurmurHash3_fmix32 {
  static_assert(sizeof(Key) == 4, "Key type must be 4 bytes in size.");

  using argument_type = Key;            ///< The type of the values taken as argument
  using result_type   = std::uint32_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a MurmurHash3_fmix32 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  __host__ __device__ constexpr MurmurHash3_fmix32(std::uint32_t seed = 0) : seed_{seed} {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return A resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    std::uint32_t h = static_cast<std::uint32_t>(key) ^ seed_;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

 private:
  std::uint32_t seed_;
};

/**
 * @brief The 64bit integer finalizer hash function of `MurmurHash3`.
 *
 * @throw Key type must be 8 bytes in size
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct MurmurHash3_fmix64 {
  static_assert(sizeof(Key) == 8, "Key type must be 8 bytes in size.");

  using argument_type = Key;            ///< The type of the values taken as argument
  using result_type   = std::uint64_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a MurmurHash3_fmix64 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  __host__ __device__ constexpr MurmurHash3_fmix64(std::uint64_t seed = 0) : seed_{seed} {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return A resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    std::uint64_t h = static_cast<std::uint64_t>(key) ^ seed_;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;
    return h;
  }

 private:
  std::uint64_t seed_;
};

/**
 * @brief A `MurmurHash3_32` hash function to hash the given argument on host and device.
 *
 * MurmurHash3_32 implementation from
 * https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 * -----------------------------------------------------------------------------
 * MurmurHash3 was written by Austin Appleby, and is placed in the public domain. The author
 * hereby disclaims copyright to this source code.
 *
 * Note - The x86 and x64 versions do _not_ produce the same results, as the algorithms are
 * optimized for their respective platforms. You can still compile and run any of them on any
 * platform, but your performance with the non-native version will be less than optimal.
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct MurmurHash3_32 {
  using argument_type = Key;            ///< The type of the values taken as argument
  using result_type   = std::uint32_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a MurmurHash3_32 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  __host__ __device__ constexpr MurmurHash3_32(std::uint32_t seed = 0) : fmix32_{0}, seed_{seed} {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return The resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    return compute_hash(reinterpret_cast<std::byte const*>(&key),
                        cuco::experimental::extent<std::size_t, sizeof(Key)>{});
  }

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @tparam Extent The extent type
   *
   * @param bytes The input argument to hash
   * @param size The extent of the data in bytes
   * @return The resulting hash value
   */
  template <typename Extent>
  constexpr result_type __host__ __device__ compute_hash(std::byte const* bytes,
                                                         Extent size) const noexcept
  {
    auto const nblocks = size / 4;

    std::uint32_t h1           = seed_;
    constexpr std::uint32_t c1 = 0xcc9e2d51;
    constexpr std::uint32_t c2 = 0x1b873593;
    //----------
    // body
    for (std::remove_const_t<decltype(nblocks)> i = 0; size >= 4 && i < nblocks; i++) {
      std::uint32_t k1 = load_chunk<std::uint32_t>(bytes, i);
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }
    //----------
    // tail
    std::uint32_t k1 = 0;
    switch (size & 3) {
      case 3: k1 ^= std::to_integer<std::uint32_t>(bytes[nblocks * 4 + 2]) << 16; [[fallthrough]];
      case 2: k1 ^= std::to_integer<std::uint32_t>(bytes[nblocks * 4 + 1]) << 8; [[fallthrough]];
      case 1:
        k1 ^= std::to_integer<std::uint32_t>(bytes[nblocks * 4 + 0]);
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
    };
    //----------
    // finalization
    h1 ^= size;
    h1 = fmix32_(h1);
    return h1;
  }

 private:
  constexpr __host__ __device__ std::uint32_t rotl32(std::uint32_t x, std::int8_t r) const noexcept
  {
    return (x << r) | (x >> (32 - r));
  }

  MurmurHash3_fmix32<std::uint32_t> fmix32_;
  std::uint32_t seed_;
};
}  //  namespace cuco::detail