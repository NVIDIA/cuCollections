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

#include <test_utils.hpp>

#include <cuco/detail/__config>
#include <cuco/hash_functions.cuh>

#include <thrust/device_vector.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>

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

template <typename Hash>
__host__ __device__ bool check_hash_result(typename Hash::argument_type const& key,
                                           typename Hash::result_type seed,
                                           typename Hash::result_type expected) noexcept
{
  Hash h(seed);
  return (h(key) == expected);
}

template <typename OutputIter>
__global__ void check_hash_result_kernel_64(OutputIter result)
{
  int i = 0;

  result[i++] = check_hash_result<cuco::xxhash_64<char>>(0, 0, 16804241149081757544);
  result[i++] = check_hash_result<cuco::xxhash_64<char>>(42, 0, 765293966243412708);
  result[i++] = check_hash_result<cuco::xxhash_64<char>>(0, 42, 9486749600008296231);

  result[i++] = check_hash_result<cuco::xxhash_64<int32_t>>(0, 0, 4246796580750024372);
  result[i++] = check_hash_result<cuco::xxhash_64<int32_t>>(0, 42, 3614696996920510707);
  result[i++] = check_hash_result<cuco::xxhash_64<int32_t>>(42, 0, 15516826743637085169);
  result[i++] = check_hash_result<cuco::xxhash_64<int32_t>>(123456789, 0, 9462334144942111946);

  result[i++] = check_hash_result<cuco::xxhash_64<int64_t>>(0, 0, 3803688792395291579);
  result[i++] = check_hash_result<cuco::xxhash_64<int64_t>>(0, 42, 13194218611613725804);
  result[i++] = check_hash_result<cuco::xxhash_64<int64_t>>(42, 0, 13066772586158965587);
  result[i++] = check_hash_result<cuco::xxhash_64<int64_t>>(123456789, 0, 14662639848940634189);

#if defined(CUCO_HAS_INT128)
  result[i++] = check_hash_result<cuco::xxhash_64<__int128>>(123456789, 0, 7986913354431084250);
#endif

  result[i++] =
    check_hash_result<cuco::xxhash_64<large_key<32>>>(123456789, 0, 2031761887105658523);
}

TEST_CASE("Test cuco::xxhash_64", "")
{
  // Reference hash values were computed using https://github.com/Cyan4973/xxHash
  SECTION("Check if host-generated hash values match the reference implementation.")
  {
    CHECK(check_hash_result<cuco::xxhash_64<char>>(0, 0, 16804241149081757544));
    CHECK(check_hash_result<cuco::xxhash_64<char>>(42, 0, 765293966243412708));
    CHECK(check_hash_result<cuco::xxhash_64<char>>(0, 42, 9486749600008296231));

    CHECK(check_hash_result<cuco::xxhash_64<int32_t>>(0, 0, 4246796580750024372));
    CHECK(check_hash_result<cuco::xxhash_64<int32_t>>(0, 42, 3614696996920510707));
    CHECK(check_hash_result<cuco::xxhash_64<int32_t>>(42, 0, 15516826743637085169));
    CHECK(check_hash_result<cuco::xxhash_64<int32_t>>(123456789, 0, 9462334144942111946));

    CHECK(check_hash_result<cuco::xxhash_64<int64_t>>(0, 0, 3803688792395291579));
    CHECK(check_hash_result<cuco::xxhash_64<int64_t>>(0, 42, 13194218611613725804));
    CHECK(check_hash_result<cuco::xxhash_64<int64_t>>(42, 0, 13066772586158965587));
    CHECK(check_hash_result<cuco::xxhash_64<int64_t>>(123456789, 0, 14662639848940634189));

#if defined(CUCO_HAS_INT128)
    CHECK(check_hash_result<cuco::xxhash_64<__int128>>(123456789, 0, 7986913354431084250));
#endif

    // 32*4=128-byte key to test the pipelined outermost hashing loop
    CHECK(check_hash_result<cuco::xxhash_64<large_key<32>>>(123456789, 0, 2031761887105658523));
  }

  SECTION("Check if device-generated hash values match the reference implementation.")
  {
    thrust::device_vector<bool> result(10);

    check_hash_result_kernel_64<<<1, 1>>>(result.begin());

    CHECK(cuco::test::all_of(result.begin(), result.end(), [] __device__(bool v) { return v; }));
  }
}

template <typename OutputIter>
__global__ void check_hash_result_kernel_32(OutputIter result)
{
  int i = 0;

  result[i++] = check_hash_result<cuco::xxhash_32<char>>(0, 0, 3479547966);
  result[i++] = check_hash_result<cuco::xxhash_32<char>>(42, 0, 3774771295);
  result[i++] = check_hash_result<cuco::xxhash_32<char>>(0, 42, 2099223482);

  result[i++] = check_hash_result<cuco::xxhash_32<int32_t>>(0, 0, 148298089);
  result[i++] = check_hash_result<cuco::xxhash_32<int32_t>>(0, 42, 2132181312);
  result[i++] = check_hash_result<cuco::xxhash_32<int32_t>>(42, 0, 1161967057);
  result[i++] = check_hash_result<cuco::xxhash_32<int32_t>>(123456789, 0, 2987034094);

  result[i++] = check_hash_result<cuco::xxhash_32<int64_t>>(0, 0, 3736311059);
  result[i++] = check_hash_result<cuco::xxhash_32<int64_t>>(0, 42, 1076387279);
  result[i++] = check_hash_result<cuco::xxhash_32<int64_t>>(42, 0, 2332451213);
  result[i++] = check_hash_result<cuco::xxhash_32<int64_t>>(123456789, 0, 1561711919);

#if defined(CUCO_HAS_INT128)
  result[i++] = check_hash_result<cuco::xxhash_32<__int128>>(123456789, 0, 1846633701);
#endif

  result[i++] = check_hash_result<cuco::xxhash_32<large_key<32>>>(123456789, 0, 3715432378);
}

TEST_CASE("Test cuco::xxhash_32", "")
{
  // Reference hash values were computed using https://github.com/Cyan4973/xxHash
  SECTION("Check if host-generated hash values match the reference implementation.")
  {
    CHECK(check_hash_result<cuco::xxhash_32<char>>(0, 0, 3479547966));
    CHECK(check_hash_result<cuco::xxhash_32<char>>(42, 0, 3774771295));
    CHECK(check_hash_result<cuco::xxhash_32<char>>(0, 42, 2099223482));

    CHECK(check_hash_result<cuco::xxhash_32<int32_t>>(0, 0, 148298089));
    CHECK(check_hash_result<cuco::xxhash_32<int32_t>>(0, 42, 2132181312));
    CHECK(check_hash_result<cuco::xxhash_32<int32_t>>(42, 0, 1161967057));
    CHECK(check_hash_result<cuco::xxhash_32<int32_t>>(123456789, 0, 2987034094));

    CHECK(check_hash_result<cuco::xxhash_32<int64_t>>(0, 0, 3736311059));
    CHECK(check_hash_result<cuco::xxhash_32<int64_t>>(0, 42, 1076387279));
    CHECK(check_hash_result<cuco::xxhash_32<int64_t>>(42, 0, 2332451213));
    CHECK(check_hash_result<cuco::xxhash_32<int64_t>>(123456789, 0, 1561711919));

#if defined(CUCO_HAS_INT128)
    CHECK(check_hash_result<cuco::xxhash_32<__int128>>(123456789, 0, 1846633701));
#endif

    // 32*4=128-byte key to test the pipelined outermost hashing loop
    CHECK(check_hash_result<cuco::xxhash_32<large_key<32>>>(123456789, 0, 3715432378));
  }

  SECTION("Check if device-generated hash values match the reference implementation.")
  {
    thrust::device_vector<bool> result(20, true);

    check_hash_result_kernel_32<<<1, 1>>>(result.begin());

    CHECK(cuco::test::all_of(result.begin(), result.end(), [] __device__(bool v) { return v; }));
  }
}

TEMPLATE_TEST_CASE_SIG("Static vs. dynamic key hash test",
                       "",
                       ((typename Hash), Hash),
                       (cuco::murmurhash3_32<char>),
                       (cuco::murmurhash3_32<int32_t>),
                       (cuco::xxhash_32<char>),
                       (cuco::xxhash_32<int32_t>),
                       (cuco::xxhash_64<char>),
                       (cuco::xxhash_64<int32_t>))
{
  using key_type = typename Hash::argument_type;

  Hash hash;
  key_type key = 42;

  SECTION("Identical keys with static and dynamic key size should have the same hash value.")
  {
    CHECK(hash(key) ==
          hash.compute_hash(reinterpret_cast<std::byte const*>(&key), sizeof(key_type)));
  }
}

template <typename Hash>
__host__ __device__ bool check_murmurhash3_128_result(typename Hash::argument_type const& key,
                                                      std::uint64_t seed,
                                                      typename Hash::result_type expected) noexcept
{
  Hash h(seed);
  return (h(key) == expected);
}

template <typename OutputIter>
__global__ void check_murmurhash3_128_result_kernel(OutputIter result)
{
  int i = 0;

  result[i++] = check_murmurhash3_128_result<cuco::murmurhash3_128<int>>(
    0, 0, {14961230494313510588, 6383328099726337777});
  result[i++] = check_murmurhash3_128_result<cuco::murmurhash3_128<int>>(
    9, 0, {1779292183511753683, 16298496441448380334});
  result[i++] = check_murmurhash3_128_result<cuco::murmurhash3_128<int>>(
    42, 0, {2913627637088662735, 16344193523890567190});
  result[i++] = check_murmurhash3_128_result<cuco::murmurhash3_128<int>>(
    42, 42, {2248879576374326886, 18006515275339376488});
  result[i++] = check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<int, 2>>>(
    {2, 2}, 0, {12221386834995143465, 6690950894782946573});
  result[i++] = check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<int, 3>>>(
    {1, 4, 9}, 42, {299140022350411792, 9891903873182035274});
  result[i++] = check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<int, 4>>>(
    {42, 64, 108, 1024}, 63, {4333511168876981289, 4659486988434316416});
  result[i++] = check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<int, 16>>>(
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    1024,
    {3302412811061286680, 7070355726356610672});
  result[i++] = check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<long, 2>>>(
    {2, 2}, 0, {8554944597931919519, 14938998000509429729});
  result[i++] = check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<long, 3>>>(
    {1, 4, 9}, 42, {13442629947720186435, 7061727494178573325});
  result[i++] = check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<long, 4>>>(
    {42, 64, 108, 1024}, 63, {8786399719555989948, 14954183901757012458});
  result[i++] = check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<long, 16>>>(
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    1024,
    {15409921801541329777, 10546487400963404004});
}

TEST_CASE("Test cuco::murmurhash3_128", "")
{
  // Reference hash values were computed using
  // https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp

  SECTION("Check if host-generated hash values match the reference implementation.")
  {
    CHECK(check_murmurhash3_128_result<cuco::murmurhash3_128<int>>(
      0, 0, {14961230494313510588, 6383328099726337777}));
    CHECK(check_murmurhash3_128_result<cuco::murmurhash3_128<int>>(
      9, 0, {1779292183511753683, 16298496441448380334}));
    CHECK(check_murmurhash3_128_result<cuco::murmurhash3_128<int>>(
      42, 0, {2913627637088662735, 16344193523890567190}));
    CHECK(check_murmurhash3_128_result<cuco::murmurhash3_128<int>>(
      42, 42, {2248879576374326886, 18006515275339376488}));
    CHECK(check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<int, 2>>>(
      {2, 2}, 0, {12221386834995143465, 6690950894782946573}));
    CHECK(check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<int, 3>>>(
      {1, 4, 9}, 42, {299140022350411792, 9891903873182035274}));
    CHECK(check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<int, 4>>>(
      {42, 64, 108, 1024}, 63, {4333511168876981289, 4659486988434316416}));
    CHECK(check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<int, 16>>>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      1024,
      {3302412811061286680, 7070355726356610672}));
    CHECK(check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<long, 2>>>(
      {2, 2}, 0, {8554944597931919519, 14938998000509429729}));
    CHECK(check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<long, 3>>>(
      {1, 4, 9}, 42, {13442629947720186435, 7061727494178573325}));
    CHECK(check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<long, 4>>>(
      {42, 64, 108, 1024}, 63, {8786399719555989948, 14954183901757012458}));
    CHECK(check_murmurhash3_128_result<cuco::murmurhash3_128<cuda::std::array<long, 16>>>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      1024,
      {15409921801541329777, 10546487400963404004}));
  }

  SECTION("Check if device-generated hash values match the reference implementation.")
  {
    thrust::device_vector<bool> result(12, true);

    check_murmurhash3_128_result_kernel<<<1, 1>>>(result.begin());

    CHECK(cuco::test::all_of(result.begin(), result.end(), [] __device__(bool v) { return v; }));
  }
}