/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

namespace cuco {
namespace detail {

/**
 * @brief Modular multiplication: (n1 * n2) % m using 128-bit intermediate.
 */
constexpr std::uint64_t mod_mul(std::uint64_t n1, std::uint64_t n2, std::uint64_t m)
{
  auto r = static_cast<unsigned __int128>(n1) * n2;
  return static_cast<std::uint64_t>(r % m);
}

/**
 * @brief Modular exponentiation: (b ^ e) % m via binary exponentiation.
 */
constexpr std::uint64_t mod_pow(std::uint64_t b, std::uint64_t e, std::uint64_t m)
{
  std::uint64_t r = 1;
  b %= m;
  while (e > 0) {
    if (e & 1) { r = mod_mul(r, b, m); }
    b = mod_mul(b, b, m);
    e >>= 1;
  }
  return r;
}

/**
 * @brief Single Miller-Rabin witness test.
 *
 * Given n - 1 == 2^s * d, checks whether a^d == 1 (mod n) or
 * a^(2^r * d) == n - 1 (mod n) for some 0 <= r < s.
 */
constexpr bool miller_rabin_test(std::uint64_t n, std::uint64_t a, std::uint64_t d, std::uint32_t s)
{
  std::uint64_t x             = mod_pow(a % n, d, n);
  std::uint64_t const neg_one = n - 1;
  if (x == 1 || x == neg_one) { return true; }

  for (std::uint32_t i = 1; i < s; ++i) {
    x = mod_mul(x, x, n);
    if (x == neg_one) { return true; }
  }
  return false;
}

/**
 * @brief Deterministic primality test for all uint64_t values.
 *
 * Uses trial division for small factors followed by Miller-Rabin with
 * a fixed set of bases that make the test deterministic for all 64-bit
 * integers. Bases from https://cp-algorithms.com/algebra/primality_tests.html
 */
constexpr bool is_prime(std::uint64_t n)
{
  if (n < 2) { return false; }

  // Trial division by small primes
  for (std::uint64_t p :
       {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
    if (n % p == 0) { return n == p; }
  }

  // Decompose n - 1 == 2^s * d
  std::uint64_t d = n - 1;
  std::uint32_t s = 0;
  while ((d & 1) == 0) {
    d >>= 1;
    ++s;
  }

  // Deterministic bases for all uint64_t values
  for (std::uint64_t a : {2ull, 325ull, 9375ull, 28178ull, 450775ull, 9780504ull, 1795265022ull}) {
    if (!miller_rabin_test(n, a, d, s)) { return false; }
  }

  return true;
}

/**
 * @brief Returns the smallest prime >= n.
 *
 * For n <= 2, returns 2. Otherwise searches odd numbers starting
 * from n (or n+1 if n is even).
 */
constexpr std::uint64_t next_prime(std::uint64_t n)
{
  if (n <= 2ull) { return 2ull; }

  n |= 1;  // make odd

  while (!is_prime(n)) {
    n += 2ull;
  }

  return n;
}

}  // namespace detail
}  // namespace cuco
