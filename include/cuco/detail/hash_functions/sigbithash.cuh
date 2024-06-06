/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

namespace cuco::detail {

/**
 * @brief An SigBit hash function to hash the given argument on host and device.
 *
 * -----------------------------------------------------------------------------
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct SigBitHash_32 {
  using argument_type = Key;            ///< The type of the values taken as argument
  using result_type   = std::uint32_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a SigBitHash_32 hash function by finding the significant bits of
   * a given span of keys
   *
   * @param start 	A pointer/iterator to the first key in the list
   * @param numKeys 	The number of keys in the span
   */
  __host__ __device__ constexpr SigBitHash_32(Key* start, std::uint32_t numKeys)
    : sigbit_mask_{reduce_and_or(start, numKeys)}
  {
  }

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return The resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    if constexpr (sizeof(Key) <= 16) {
      Key const key_copy = key;
      return compute_hash(reinterpret_cast<std::byte const*>(&key_copy),
                          cuco::extent<std::size_t, sizeof(Key)>{});
    } else {
      return compute_hash(reinterpret_cast<std::byte const*>(&key),
                          cuco::extent<std::size_t, sizeof(Key)>{});
    }
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
    std::uint32_t h1 = load_chunk<std::uint32_t>(bytes, 0) & sigbit_mask_;
    return h1;
  }

 private:
  std::uint32_t sigbit_mask_;
};  // SigBit hash 32

/**
 * @brief An SigBit hash function to hash the given argument on host and device.
 *
 * -----------------------------------------------------------------------------
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct SigBitHash_64 {
  using argument_type = Key;            ///< The type of the values taken as argument
  using result_type   = std::uint64_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a SigBitHash_64 hash function by finding the significant bits of
   * a given span of keys
   *
   * @param start 	A pointer/iterator to the first key in the list
   * @param numKeys 	The number of keys in the span
   */
  __host__ __device__ constexpr SigBitHash_64(Key* start, std::uint64_t numKeys)
    : sigbit_mask_{reduce_and_or(start, numKeys)}
  {
  }

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return The resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    if constexpr (sizeof(Key) <= 16) {
      Key const key_copy = key;
      return compute_hash(reinterpret_cast<std::byte const*>(&key_copy),
                          cuco::extent<std::size_t, sizeof(Key)>{});
    } else {
      return compute_hash(reinterpret_cast<std::byte const*>(&key),
                          cuco::extent<std::size_t, sizeof(Key)>{});
    }
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
    std::uint64_t h1 = load_chunk<std::uint64_t>(bytes, 0) & sigbit_mask_;
    return h1;
  }

 private:
  std::uint64_t sigbit_mask_;
};  // SigBit hash 64

}  //  namespace cuco::detail
