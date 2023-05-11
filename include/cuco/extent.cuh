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
#include <type_traits>

namespace cuco {
namespace experimental {

// TODO docs
enum class extent_kind : std::int32_t { PLAIN, PRIME, POW2 };

static constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);

/**
 * @brief Static extent class.
 *
 * @tparam SizeType Size type
 * @tparam N Extent
 * @tparam Kind Extent kind
 */
template <typename SizeType, std::size_t N = dynamic_extent, extent_kind Kind = extent_kind::PRIME>
struct extent {
  static_assert(std::is_integral_v<SizeType>, "SizeType must be an integer type");

  using value_type = SizeType;  ///< Extent value type

  static constexpr extent_kind kind = Kind;  ///< Extent kind

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

  /**
   * @brief Optimized modulo operator for `lhs % *this`.
   *
   * @param lhs Left hand side operand
   *
   * @return Resulting remainder
   */
  [[nodiscard]] constexpr value_type mod(value_type lhs) const noexcept { return lhs % N; }
};

/**
 * @brief Dynamic extent class.
 *
 * @tparam SizeType Size type
 * @tparam Kind Extent kind
 */
template <typename SizeType, extent_kind Kind>
struct extent<SizeType, dynamic_extent, Kind> {
  static_assert(std::is_integral_v<SizeType>, "SizeType must be an integer type");

  using value_type = SizeType;  ///< Extent value type

  static constexpr extent_kind kind = Kind;  ///< Extent kind

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

  /**
   * @brief Optimized modulo operator for `lhs % *this`.
   *
   * @param lhs Left hand side operand
   *
   * @return Resulting remainder
   */
  [[nodiscard]] constexpr value_type mod(value_type lhs) const noexcept
  {
    if constexpr (Kind == extent_kind::POW2) {
      return lhs & (value_ - 1);
    } else {
      return lhs % value_;  // unoptimized code path
    }
  }

 private:
  value_type value_;  ///< Extent value
};

/**
 * @brief Computes valid extent based on given parameters.
 *
 * @note The actual capacity of a container (map/set) should be exclusively determined by the return
 * value of this utility since the output depends on the requested low-bound size, the probing
 * scheme, and the storage. This utility is used internally during container constructions while for
 * container ref constructions, it would be users' responsibility to use this function to determine
 * the input size of the ref.
 *
 * @tparam CGSize Number of elements handled per CG
 * @tparam WindowSize Number of elements handled per Window
 * @tparam SizeType Size type
 * @tparam N Extent
 * @tparam Kind Extent kind
 *
 * @throw If the input extent is invalid
 *
 * @return Resulting valid extent
 */
template <int32_t CGSize, int32_t WindowSize, typename SizeType, std::size_t N, extent_kind Kind>
[[nodiscard]] auto constexpr make_valid_extent(extent<SizeType, N, Kind> ext)
{
  if (ext <= 0) { CUCO_FAIL("Extent must be greater than 0"); }

  auto const div = SDIV(static_cast<SizeType>(ext), WindowSize * CGSize);

  if constexpr (Kind == extent_kind::PLAIN) {
    if constexpr (N == dynamic_extent) {
      return extent<SizeType, dynamic_extent, Kind>{div * CGSize};
    } else {
      return extent<SizeType, div * CGSize, Kind>{};
    }
  }

  if constexpr (Kind == extent_kind::POW2) {
    auto next_pow2 = [](SizeType v) constexpr->SizeType
    {
      auto constexpr max_pow2 = 1ULL << (sizeof(SizeType) * CHAR_BIT - 1);
      static_assert(sizeof(SizeType) == 4 or sizeof(SizeType) == 8, "Invalid SizeType");
      if (v > max_pow2) { CUCO_FAIL("Extent out of range for SizeType"); }
      v--;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
      if constexpr (sizeof(SizeType) == 8) { v |= v >> 32; }
      v++;
      return v;
    };

    if constexpr (N == dynamic_extent) {
      return extent<SizeType, dynamic_extent, Kind>{next_pow2(div) * CGSize};
    } else {
      return extent<SizeType, next_pow2(div) * CGSize, Kind>{};
    }
  }

  if constexpr (Kind == extent_kind::PRIME) {
    auto next_prime = [](SizeType v) constexpr->SizeType
    {
      auto constexpr max_prime = cuco::detail::primes.back();
      auto constexpr max_value =
        (static_cast<uint64_t>(std::numeric_limits<SizeType>::max()) < max_prime)
          ? std::numeric_limits<SizeType>::max()
          : static_cast<SizeType>(max_prime);
      if (v > max_value) { CUCO_FAIL("Extent out of range for SizeType"); }

      return *cuco::detail::lower_bound(
        cuco::detail::primes.begin(), cuco::detail::primes.end(), static_cast<uint64_t>(v));
    };

    if constexpr (N == dynamic_extent) {
      return extent<SizeType, dynamic_extent, Kind>{next_prime(div) * CGSize};
    } else {
      return extent<SizeType, next_prime(div) * CGSize, Kind>{};
    }
  }
}

}  // namespace experimental
}  // namespace cuco
