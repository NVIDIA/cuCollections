/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/__config>

#include <cstdint>
#include <limits>

namespace cuco {
namespace detail {

/**
 * @brief Modular multiplication: (n1 * n2) % m without overflow.
 */
constexpr std::uint64_t mod_mul(std::uint64_t n1, std::uint64_t n2, std::uint64_t m)
{
#if defined(CUCO_HAS_INT128)
  auto r = static_cast<unsigned __int128>(n1) * n2;
  return static_cast<std::uint64_t>(r % m);
#else
  // Fallback: Russian peasant multiplication in modular arithmetic.
  std::uint64_t r = 0;
  n1 %= m;
  n2 %= m;
  while (n2 > 0) {
    if (n2 & 1) { r = (r >= m - n1) ? r - (m - n1) : r + n1; }
    n1 = (n1 >= m - n1) ? n1 - (m - n1) : n1 + n1;
    n2 >>= 1;
  }
  return r;
#endif
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
 * @brief Returns the smallest prime in `[n, upper_bound]`.
 *
 * @param n Lower bound of the search range
 * @param upper_bound Upper bound of the search range
 *
 * @return The smallest prime in `[n, upper_bound]`, or zero if none exists
 */
constexpr std::uint64_t next_prime(std::uint64_t n, std::uint64_t upper_bound)
{
  if (upper_bound < 2ull || n > upper_bound) { return 0ull; }
  if (n <= 2ull) { return 2ull; }

  if ((n & 1ull) == 0) { ++n; }

  while (n <= upper_bound) {
    if (is_prime(n)) { return n; }
    if (upper_bound - n < 2ull) { break; }
    n += 2ull;
  }

  return 0ull;
}

/**
 * @brief Returns the smallest representable prime greater than or equal to `n`.
 *
 * @param n Lower bound of the search range
 *
 * @return The smallest representable prime greater than or equal to `n`, or zero if none exists
 */
constexpr std::uint64_t next_prime(std::uint64_t n)
{
  return next_prime(n, std::numeric_limits<std::uint64_t>::max());
}

}  // namespace detail
}  // namespace cuco
