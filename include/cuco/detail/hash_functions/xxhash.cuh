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

#include <cstdint>

namespace cuco::detail {

/**
 * @brief A `XXHash_32` hash function to hash the given argument on host and device.
 *
 * XXHash_32 implementation from
 * https://github.com/Cyan4973/xxHash
 * -----------------------------------------------------------------------------
 * xxHash - Extremely Fast Hash algorithm
 * Header File
 * Copyright (C) 2012-2021 Yann Collet
 *
 * BSD 2-Clause License (https://www.opensource.org/licenses/bsd-license.php)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following disclaimer
 *      in the documentation and/or other materials provided with the
 *      distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct XXHash_32 {
 private:
  static constexpr std::uint32_t prime1 = 0x9e3779b1u;
  static constexpr std::uint32_t prime2 = 0x85ebca77u;
  static constexpr std::uint32_t prime3 = 0xc2b2ae3du;
  static constexpr std::uint32_t prime4 = 0x27d4eb2fu;
  static constexpr std::uint32_t prime5 = 0x165667b1u;

 public:
  using argument_type = Key;            ///< The type of the values taken as argument
  using result_type   = std::uint32_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a XXH32 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  __host__ __device__ constexpr XXHash_32(std::uint32_t seed = 0) : seed_{seed} {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return A resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    // TODO do we need to add checks/hints for alignment?
    constexpr auto nbytes             = sizeof(Key);
    [[maybe_unused]] auto const bytes = reinterpret_cast<char const*>(&key);  ///< per-byte access
    [[maybe_unused]] auto const blocks =
      reinterpret_cast<std::uint32_t const*>(&key);  ///< 4-byte word access

    std::size_t offset = 0;
    std::uint32_t h32;

    // data can be processed in 16-byte chunks
    if constexpr (nbytes >= 16) {
      constexpr auto limit = nbytes - 16;
      std::uint32_t v1     = seed_ + prime1 + prime2;
      std::uint32_t v2     = seed_ + prime2;
      std::uint32_t v3     = seed_;
      std::uint32_t v4     = seed_ - prime1;

      do {
        // pipeline 4*4byte computations
        auto const pipeline_offset = offset / 4;
        v1 += blocks[pipeline_offset] * prime2;
        v1 = rotl(v1, 13);
        v1 *= prime1;
        v2 += blocks[pipeline_offset + 1] * prime2;
        v2 = rotl(v2, 13);
        v2 *= prime1;
        v3 += blocks[pipeline_offset + 2] * prime2;
        v3 = rotl(v3, 13);
        v3 *= prime1;
        v4 += blocks[pipeline_offset + 3] * prime2;
        v4 = rotl(v4, 13);
        v4 *= prime1;
        offset += 16;
      } while (offset <= limit);

      h32 = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    } else {
      h32 = seed_ + prime5;
    }

    h32 += nbytes;

    // remaining data can be processed in 4-byte chunks
    if constexpr ((nbytes % 16) >= 4) {
      for (; offset <= nbytes - 4; offset += 4) {
        h32 += blocks[offset / 4] * prime3;
        h32 = rotl(h32, 17) * prime4;
      }
    }

    // the following loop is only needed if the size of the key is not a multiple of the block size
    if constexpr (nbytes % 4) {
      while (offset < nbytes) {
        h32 += (bytes[offset] & 255) * prime5;
        h32 = rotl(h32, 11) * prime1;
        ++offset;
      }
    }

    return finalize(h32);
  }

 private:
  constexpr __host__ __device__ std::uint32_t rotl(std::uint32_t h, std::int8_t r) const noexcept
  {
    return ((h << r) | (h >> (32 - r)));
  }

  // avalanche helper
  constexpr __host__ __device__ std::uint32_t finalize(std::uint32_t h) const noexcept
  {
    h ^= h >> 15;
    h *= prime2;
    h ^= h >> 13;
    h *= prime3;
    h ^= h >> 16;
    return h;
  }

  std::uint32_t seed_;
};

/**
 * @brief A `XXHash_64` hash function to hash the given argument on host and device.
 *
 * XXHash_64 implementation from
 * https://github.com/Cyan4973/xxHash
 * -----------------------------------------------------------------------------
 * xxHash - Extremely Fast Hash algorithm
 * Header File
 * Copyright (C) 2012-2021 Yann Collet
 *
 * BSD 2-Clause License (https://www.opensource.org/licenses/bsd-license.php)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following disclaimer
 *      in the documentation and/or other materials provided with the
 *      distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * You can contact the author at:
 *   - xxHash homepage: https://www.xxhash.com
 *   - xxHash source repository: https://github.com/Cyan4973/xxHash
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct XXHash_64 {
 private:
  static constexpr std::uint64_t prime1 = 11400714785074694791ull;
  static constexpr std::uint64_t prime2 = 14029467366897019727ull;
  static constexpr std::uint64_t prime3 = 1609587929392839161ull;
  static constexpr std::uint64_t prime4 = 9650029242287828579ull;
  static constexpr std::uint64_t prime5 = 2870177450012600261ull;

 public:
  using argument_type = Key;            ///< The type of the values taken as argument
  using result_type   = std::uint64_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a XXH64 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  __host__ __device__ constexpr XXHash_64(std::uint64_t seed = 0) : seed_{seed} {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return A resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    // TODO do we need to add checks/hints for alignment?
    constexpr auto nbytes             = sizeof(Key);
    [[maybe_unused]] auto const bytes = reinterpret_cast<char const*>(&key);  ///< per-byte access
    [[maybe_unused]] auto const blocks4 =
      reinterpret_cast<std::uint32_t const*>(&key);  ///< 4-byte word access
    [[maybe_unused]] auto const blocks8 =
      reinterpret_cast<std::uint64_t const*>(&key);  ///< 8-byte word access

    std::size_t offset = 0;
    std::uint64_t h64;

    // data can be processed in 32-byte chunks
    if constexpr (nbytes >= 32) {
      constexpr auto limit = nbytes - 32;
      std::uint64_t v1     = seed_ + prime1 + prime2;
      std::uint64_t v2     = seed_ + prime2;
      std::uint64_t v3     = seed_;
      std::uint64_t v4     = seed_ - prime1;

      do {
        // pipeline 4*8byte computations
        auto const pipeline_offset = offset / 8;
        v1 += blocks8[pipeline_offset] * prime2;
        v1 = rotl(v1, 31);
        v1 *= prime1;
        v2 += blocks8[pipeline_offset + 1] * prime2;
        v2 = rotl(v2, 31);
        v2 *= prime1;
        v3 += blocks8[pipeline_offset + 2] * prime2;
        v3 = rotl(v3, 31);
        v3 *= prime1;
        v4 += blocks8[pipeline_offset + 3] * prime2;
        v4 = rotl(v4, 31);
        v4 *= prime1;
        offset += 32;
      } while (offset <= limit);

      h64 = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);

      v1 *= prime2;
      v1 = rotl(v1, 31);
      v1 *= prime1;
      h64 ^= v1;
      h64 = h64 * prime1 + prime4;

      v2 *= prime2;
      v2 = rotl(v2, 31);
      v2 *= prime1;
      h64 ^= v2;
      h64 = h64 * prime1 + prime4;

      v3 *= prime2;
      v3 = rotl(v3, 31);
      v3 *= prime1;
      h64 ^= v3;
      h64 = h64 * prime1 + prime4;

      v4 *= prime2;
      v4 = rotl(v4, 31);
      v4 *= prime1;
      h64 ^= v4;
      h64 = h64 * prime1 + prime4;
    } else {
      h64 = seed_ + prime5;
    }

    h64 += nbytes;

    // remaining data can be processed in 8-byte chunks
    if constexpr ((nbytes % 32) >= 8) {
      for (; offset <= nbytes - 8; offset += 8) {
        std::uint64_t k1 = blocks8[offset / 8] * prime2;
        k1               = rotl(k1, 31) * prime1;
        h64 ^= k1;
        h64 = rotl(h64, 27) * prime1 + prime4;
      }
    }

    // remaining data can be processed in 4-byte chunks
    if constexpr (((nbytes % 32) % 8) >= 4) {
      for (; offset <= nbytes - 4; offset += 4) {
        h64 ^= (blocks4[offset / 4] & 0xffffffffull) * prime1;
        h64 = rotl(h64, 23) * prime2 + prime3;
      }
    }

    // the following loop is only needed if the size of the key is not a multiple of a previous
    // block size
    if constexpr (nbytes % 4) {
      while (offset < nbytes) {
        h64 ^= (bytes[offset] & 0xff) * prime5;
        h64 = rotl(h64, 11) * prime1;
        ++offset;
      }
    }
    return finalize(h64);
  }

 private:
  constexpr __host__ __device__ std::uint64_t rotl(std::uint64_t h, std::int8_t r) const noexcept
  {
    return ((h << r) | (h >> (64 - r)));
  }

  // avalanche helper
  constexpr __host__ __device__ std::uint64_t finalize(std::uint64_t h) const noexcept
  {
    h ^= h >> 33;
    h *= prime2;
    h ^= h >> 29;
    h *= prime3;
    h ^= h >> 32;
    return h;
  }

  std::uint64_t seed_;
};

}  // namespace cuco::detail