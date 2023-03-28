/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cuco/detail/prime.hpp>

#include <cstddef>
#include <cstdint>

namespace cuco {
namespace experimental {
static constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);

/**
 * @brief Extent class.
 *
 * @tparam SizeType Size type
 * @tparam N Extent
 */
template <typename SizeType, std::size_t N = dynamic_extent>
struct extent {
  using value_type = SizeType;  ///< Extent value type

  constexpr extent() = default;

  /// Constructs from `SizeType`
  __host__ __device__ constexpr explicit extent(SizeType) noexcept {}

  /**
   * @brief Conversion to value_type.
   *
   * @return Extent size
   */
  __host__ __device__ constexpr operator value_type() const noexcept { return N; }

  /**
   * @brief Computes valid extent based on given parameters.
   *
   * @tparam CGSize Number of elements handled per CG
   * @tparam WindowSize Number of elements handled per Window
   *
   * @return Resulting valid static extent
   */
  template <int32_t CGSize, int32_t WindowSize>
  [[nodiscard]] constexpr auto valid_extent() const
  {
    auto constexpr max_prime = cuco::detail::primes.back();
    auto constexpr max_value =
      (static_cast<uint64_t>(std::numeric_limits<value_type>::max()) < max_prime)
        ? std::numeric_limits<value_type>::max()
        : static_cast<value_type>(max_prime);
    auto constexpr size = SDIV(N, CGSize * WindowSize);
    if (size <= 0 or size > max_value) { CUCO_FAIL("Invalid input extent"); }
    return extent<value_type,
                  static_cast<value_type>(*cuco::detail::lower_bound(cuco::detail::primes.begin(),
                                                                     cuco::detail::primes.end(),
                                                                     static_cast<uint64_t>(size)) *
                                          CGSize)>{};
  }

  /**
   * @brief Multiplies the current extent with the given `Value`.
   *
   * @tparam Value The input value to multiply with
   *
   * @return Resulting static extent
   */
  template <int32_t Value>
  __host__ __device__ constexpr auto multiply() const noexcept
  {
    return extent<value_type, N * Value>{};
  }
};

/**
 * @brief Extent class.
 *
 * @tparam SizeType Size type
 */
template <typename SizeType>
struct extent<SizeType, dynamic_extent> {
  using value_type = SizeType;  ///< Extent value type

  /**
   * @brief Constructs extent from a given `size`.
   *
   * @param size The extent size
   */
  __host__ __device__ constexpr extent(SizeType size) noexcept : value_{size} {}

  /**
   * @brief Conversion to value_type.
   *
   * @return Extent size
   */
  __host__ __device__ constexpr operator value_type() const noexcept { return value_; }

  /**
   * @brief Computes valid extent based on given parameters.
   *
   * @tparam CGSize Number of elements handled per CG
   * @tparam WindowSize Number of elements handled per Window
   *
   * @return Resulting valid dynamic extent
   */
  template <int32_t CGSize, int32_t WindowSize>
  [[nodiscard]] constexpr auto valid_extent() const noexcept
  {
    auto constexpr max_prime = cuco::detail::primes.back();
    auto constexpr max_value =
      (static_cast<uint64_t>(std::numeric_limits<value_type>::max()) < max_prime)
        ? std::numeric_limits<value_type>::max()
        : static_cast<value_type>(max_prime);
    auto const size = SDIV(value_, CGSize * WindowSize);
    if (size <= 0 or size > max_value) { return extent<value_type>{0}; }
    return extent<value_type>{static_cast<value_type>(
      *cuco::detail::lower_bound(
        cuco::detail::primes.begin(), cuco::detail::primes.end(), static_cast<uint64_t>(size)) *
      CGSize)};
  }

  /**
   * @brief Multiplies the current extent with the given `Value`.
   *
   * @tparam Value The input value to multiply with
   *
   * @return Resulting extent
   */
  template <int32_t Value>
  __host__ __device__ constexpr auto multiply() const noexcept
  {
    return extent<value_type>{Value * value_};
  }

 private:
  value_type value_;  ///< Size of extent
};
}  // namespace experimental
}  // namespace cuco
