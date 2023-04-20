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

#include <utils.hpp>

#include <cuco/hash_functions.cuh>

#include <catch2/catch_test_macros.hpp>

template <int32_t Words>
struct large_key {
  constexpr __host__ __device__ large_key(int32_t value) noexcept
  {
    for (int32_t i = 0; i < Words; ++i) {
      data_[i] = value;
    }
  }

 private:
  int32_t data_[Words];
};

TEST_CASE("Test cuco::xxhash_64", "")
{
  // Reference hash values were computed using https://github.com/Cyan4973/xxHash
  SECTION("Check if host-generated hash values match the reference implementation.")
  {
    int32_t k1  = 0;                  // key
    uint64_t s1 = 0;                  // seed
    cuco::xxhash_64<int32_t> h1(s1);  // hasher
    CHECK(h1(k1) == 4246796580750024372ULL);

    int32_t k2  = 0;                  // key
    uint64_t s2 = 42;                 // seed
    cuco::xxhash_64<int32_t> h2(s2);  // hasher
    CHECK(h2(k2) == 3614696996920510707ULL);

    int32_t k3  = 42;                 // key
    uint64_t s3 = 0;                  // seed
    cuco::xxhash_64<int32_t> h3(s3);  // hasher
    CHECK(h3(k3) == 15516826743637085169ULL);

    int32_t k4  = 123456789;          // key
    uint64_t s4 = 0;                  // seed
    cuco::xxhash_64<int32_t> h4(s4);  // hasher
    CHECK(h4(k4) == 9462334144942111946ULL);

    int64_t k5  = 0;                  // key
    uint64_t s5 = 0;                  // seed
    cuco::xxhash_64<int64_t> h5(s5);  // hasher
    CHECK(h5(k5) == 3803688792395291579ULL);

    int64_t k6  = 0;                  // key
    uint64_t s6 = 42;                 // seed
    cuco::xxhash_64<int64_t> h6(s6);  // hasher
    CHECK(h6(k6) == 13194218611613725804ULL);

    int64_t k7  = 42;                 // key
    uint64_t s7 = 0;                  // seed
    cuco::xxhash_64<int64_t> h7(s7);  // hasher
    CHECK(h7(k7) == 13066772586158965587ULL);

    int64_t k8  = 123456789;          // key
    uint64_t s8 = 0;                  // seed
    cuco::xxhash_64<int64_t> h8(s8);  // hasher
    CHECK(h8(k8) == 14662639848940634189ULL);

    // TODO check if __int128 is available
    __int128 k9 = 123456789;           // key
    uint64_t s9 = 0;                   // seed
    cuco::xxhash_64<__int128> h9(s9);  // hasher
    CHECK(h9(k9) == 7986913354431084250ULL);

    // 32*4=128-byte key to test the pipelined outermost hashing loop
    large_key<32> k10(123456789);             // key
    uint64_t s10 = 0;                         // seed
    cuco::xxhash_64<large_key<32>> h10(s10);  // hasher
    CHECK(h10(k10) == 2031761887105658523ULL);
  }

  // TODO SECTION("Check if device-generated hash values match the reference implementation.")
}

TEST_CASE("Test cuco::xxhash_32", "")
{
  // Reference hash values were computed using https://github.com/Cyan4973/xxHash
  SECTION("Check if host-generated hash values match the reference implementation.")
  {
    int32_t k1  = 0;                  // key
    uint32_t s1 = 0;                  // seed
    cuco::xxhash_32<int32_t> h1(s1);  // hasher
    CHECK(h1(k1) == 148298089);

    int32_t k2  = 0;                  // key
    uint32_t s2 = 42;                 // seed
    cuco::xxhash_32<int32_t> h2(s2);  // hasher
    CHECK(h2(k2) == 2132181312);

    int32_t k3  = 42;                 // key
    uint32_t s3 = 0;                  // seed
    cuco::xxhash_32<int32_t> h3(s3);  // hasher
    CHECK(h3(k3) == 1161967057);

    int32_t k4  = 123456789;          // key
    uint32_t s4 = 0;                  // seed
    cuco::xxhash_32<int32_t> h4(s4);  // hasher
    CHECK(h4(k4) == 2987034094);

    int64_t k5  = 0;                  // key
    uint32_t s5 = 0;                  // seed
    cuco::xxhash_32<int64_t> h5(s5);  // hasher
    CHECK(h5(k5) == 3736311059);

    int64_t k6  = 0;                  // key
    uint32_t s6 = 42;                 // seed
    cuco::xxhash_32<int64_t> h6(s6);  // hasher
    CHECK(h6(k6) == 1076387279);

    int64_t k7  = 42;                 // key
    uint32_t s7 = 0;                  // seed
    cuco::xxhash_32<int64_t> h7(s7);  // hasher
    CHECK(h7(k7) == 2332451213);

    int64_t k8  = 123456789;          // key
    uint32_t s8 = 0;                  // seed
    cuco::xxhash_32<int64_t> h8(s8);  // hasher
    CHECK(h8(k8) == 1561711919);

    // TODO check if __int128 is available
    __int128 k9 = 123456789;           // key
    uint32_t s9 = 0;                   // seed
    cuco::xxhash_32<__int128> h9(s9);  // hasher
    CHECK(h9(k9) == 1846633701);

    // 32*4=128-byte key to test the pipelined outermost hashing loop
    large_key<32> k10(123456789);             // key
    uint64_t s10 = 0;                         // seed
    cuco::xxhash_32<large_key<32>> h10(s10);  // hasher
    CHECK(h10(k10) == 3715432378);
  }

  // TODO SECTION("Check if device-generated hash values match the reference implementation.")
}