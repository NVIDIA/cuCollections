/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <cuco/detail/prime.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

TEST_CASE("detail::is_prime", "")
{
  using cuco::detail::is_prime;

  SECTION("Values below 2 are not prime")
  {
    STATIC_REQUIRE(not is_prime(0));
    STATIC_REQUIRE(not is_prime(1));
  }

  SECTION("Small primes and composites")
  {
    STATIC_REQUIRE(is_prime(2));
    STATIC_REQUIRE(is_prime(3));
    STATIC_REQUIRE(not is_prime(4));
    STATIC_REQUIRE(is_prime(5));
    STATIC_REQUIRE(not is_prime(9));
    STATIC_REQUIRE(is_prime(11));
    STATIC_REQUIRE(is_prime(97));
    STATIC_REQUIRE(not is_prime(100));
  }

  SECTION("Carmichael numbers are correctly rejected")
  {
    // Strong pseudoprime candidates that fool weak primality tests
    REQUIRE(not is_prime(561));     // 3 * 11 * 17
    REQUIRE(not is_prime(1105));    // 5 * 13 * 17
    REQUIRE(not is_prime(1729));    // 7 * 13 * 19
    REQUIRE(not is_prime(2465));    // 5 * 17 * 29
    REQUIRE(not is_prime(41041));   // 7 * 11 * 13 * 41
    REQUIRE(not is_prime(825265));  // 5 * 7 * 17 * 19 * 73
  }

  SECTION("Large primes")
  {
    // Mersenne prime 2^31 - 1
    REQUIRE(is_prime(2147483647ull));
    // Near uint32 max
    REQUIRE(is_prime(4294967291ull));
    // Large 64-bit prime
    REQUIRE(is_prime(18446744073709551557ull));
    // Adjacent composite
    REQUIRE(not is_prime(18446744073709551556ull));
  }
}

TEST_CASE("detail::next_prime", "")
{
  using cuco::detail::next_prime;

  SECTION("Values at or below 2 map to 2")
  {
    STATIC_REQUIRE(next_prime(0) == 2ull);
    STATIC_REQUIRE(next_prime(1) == 2ull);
    STATIC_REQUIRE(next_prime(2) == 2ull);
  }

  SECTION("Already-prime inputs are returned unchanged")
  {
    STATIC_REQUIRE(next_prime(3) == 3ull);
    STATIC_REQUIRE(next_prime(13) == 13ull);
    STATIC_REQUIRE(next_prime(101) == 101ull);
  }

  SECTION("Composite inputs advance to the next prime")
  {
    STATIC_REQUIRE(next_prime(4) == 5ull);
    STATIC_REQUIRE(next_prime(14) == 17ull);
    STATIC_REQUIRE(next_prime(100) == 101ull);
    STATIC_REQUIRE(next_prime(155) == 157ull);  // used by extent_test
  }

  SECTION("Large composite inputs")
  {
    REQUIRE(next_prime(1ull << 20) == 1048583ull);
    REQUIRE(next_prime(1ull << 32) == 4294967311ull);
  }

  SECTION("Result is always >= input and prime")
  {
    using cuco::detail::is_prime;
    for (std::uint64_t n : {0ull, 1ull, 42ull, 1000ull, 999983ull, 1ull << 40}) {
      auto const p = next_prime(n);
      REQUIRE(p >= n);
      REQUIRE(is_prime(p));
    }
  }
}
