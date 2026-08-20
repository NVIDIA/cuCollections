/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/error.hpp>
#include <cuco/detail/prime.hpp>
#include <cuco/detail/utility/math.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/storage.cuh>
#include <cuco/utility/fast_int.cuh>

#include <cuda/std/type_traits>

#include <cmath>
#include <cstdint>

namespace cuco {
namespace detail {

constexpr std::uint64_t extent_div_ceil(std::uint64_t dividend, std::uint64_t divisor)
{
  return dividend / divisor + static_cast<std::uint64_t>(dividend % divisor != 0);
}

template <typename SizeType>
constexpr std::uint64_t max_extent_value()
{
  static_assert(cuda::std::is_integral_v<SizeType>);
  static_assert(sizeof(SizeType) <= sizeof(std::uint64_t));
  return static_cast<std::uint64_t>(cuda::std::numeric_limits<SizeType>::max());
}

template <typename SizeType, std::size_t N>
constexpr bool is_static_extent_representable()
{
  if constexpr (sizeof(SizeType) > sizeof(std::size_t) ||
                (sizeof(SizeType) == sizeof(std::size_t) && cuda::std::is_unsigned_v<SizeType>)) {
    return true;
  } else {
    return N <= static_cast<std::size_t>(cuda::std::numeric_limits<SizeType>::max());
  }
}

template <typename SizeType>
constexpr std::uint64_t normalize_extent(SizeType size)
{
  if constexpr (cuda::std::is_signed_v<SizeType>) {
    return size > 0 ? static_cast<std::uint64_t>(size) : 1ull;
  } else {
    return size == 0 ? 1ull : static_cast<std::uint64_t>(size);
  }
}

}  // namespace detail

template <typename SizeType, std::size_t N>
struct valid_extent {
  using value_type = SizeType;  ///< Extent value type

  __host__ __device__ constexpr value_type value() const noexcept { return N; }
  __host__ __device__ explicit constexpr operator value_type() const noexcept { return value(); }

 private:
  __host__ __device__ explicit constexpr valid_extent() noexcept {}
  __host__ __device__ explicit constexpr valid_extent(SizeType) noexcept {}

  // Friend declarations for all make_valid_extent overloads
  template <int32_t CGSize_, int32_t BucketSize_, typename SizeType_, std::size_t N_>
  friend auto constexpr make_valid_extent(extent<SizeType_, N_> ext);

  template <typename ProbingScheme, typename Storage, typename SizeType_, std::size_t N_>
  friend auto constexpr make_valid_extent(extent<SizeType_, N_> ext);

  template <template <typename> class ProbingScheme,
            typename Storage,
            typename SizeType_,
            std::size_t N_>
  friend auto constexpr make_valid_extent(extent<SizeType_, N_> ext);

  template <template <typename, typename> class ProbingScheme,
            typename Storage,
            typename SizeType_,
            std::size_t N_>
  friend auto constexpr make_valid_extent(extent<SizeType_, N_> ext);

  // Operator overloads
  template <typename Rhs>
  friend __host__ __device__ constexpr value_type operator-(valid_extent const& lhs,
                                                            Rhs rhs) noexcept
  {
    return lhs.value() - rhs;
  }

  template <typename Rhs>
  friend __host__ __device__ constexpr value_type operator/(valid_extent const& lhs,
                                                            Rhs rhs) noexcept
  {
    return lhs.value() / rhs;
  }

  template <typename Lhs>
  friend __host__ __device__ constexpr value_type operator%(Lhs lhs,
                                                            valid_extent const& rhs) noexcept
  {
    return lhs % rhs.value();
  }
};

template <typename SizeType>
struct valid_extent<SizeType, dynamic_extent> : cuco::utility::fast_int<SizeType> {
  using value_type =
    typename cuco::utility::fast_int<SizeType>::fast_int::value_type;  ///< Extent value type

 private:
  using cuco::utility::fast_int<SizeType>::fast_int;

  // Friend declarations for all make_valid_extent overloads
  template <int32_t CGSize_, int32_t BucketSize_, typename SizeType_, std::size_t N_>
  friend auto constexpr make_valid_extent(extent<SizeType_, N_> ext);

  template <typename ProbingScheme, typename Storage, typename SizeType_, std::size_t N_>
  friend auto constexpr make_valid_extent(extent<SizeType_, N_> ext);

  template <template <typename> class ProbingScheme,
            typename Storage,
            typename SizeType_,
            std::size_t N_>
  friend auto constexpr make_valid_extent(extent<SizeType_, N_> ext);

  template <template <typename, typename> class ProbingScheme,
            typename Storage,
            typename SizeType_,
            std::size_t N_>
  friend auto constexpr make_valid_extent(extent<SizeType_, N_> ext);
};

// Primary implementation for fixed CGSize and BucketSize
template <int32_t CGSize, int32_t BucketSize, typename SizeType, std::size_t N>
[[nodiscard]] auto constexpr make_valid_extent(extent<SizeType, N> ext)
{
  static_assert(CGSize > 0);
  static_assert(BucketSize > 0);

  constexpr auto stride     = static_cast<std::uint64_t>(CGSize) * BucketSize;
  constexpr auto max_groups = detail::max_extent_value<SizeType>() / stride;

  if constexpr (N == dynamic_extent) {
    auto const requested = detail::normalize_extent(static_cast<SizeType>(ext));
    auto const groups    = detail::extent_div_ceil(requested, stride);
    auto const prime     = cuco::detail::next_prime(groups, max_groups);
    if (prime == 0) { CUCO_FAIL("Requested extent exceeds the representable capacity"); }
    return valid_extent<SizeType, dynamic_extent>{static_cast<SizeType>(prime * stride)};
  } else {
    static_assert(detail::is_static_extent_representable<SizeType, N>(),
                  "Static extent must be representable by its size type");
    constexpr auto requested = N == 0 ? 1 : N;
    constexpr auto groups    = detail::extent_div_ceil(requested, stride);
    constexpr auto prime     = cuco::detail::next_prime(groups, max_groups);
    static_assert(prime != 0, "Requested extent exceeds the representable capacity");
    return valid_extent<SizeType, static_cast<std::size_t>(prime * stride)>{};
  }
}

// Overload for SizeType without extent
template <int32_t CGSize, int32_t BucketSize, typename SizeType>
[[nodiscard]] auto constexpr make_valid_extent(SizeType size)
{
  return make_valid_extent<CGSize, BucketSize, SizeType, dynamic_extent>(extent<SizeType>{size});
}

// Implementation for ProbingScheme and Storage types
template <typename ProbingScheme, typename Storage, typename SizeType, std::size_t N>
[[nodiscard]] auto constexpr make_valid_extent(extent<SizeType, N> ext)
{
  if constexpr (cuco::is_double_hashing<ProbingScheme>::value) {
    return make_valid_extent<ProbingScheme::cg_size, Storage::bucket_size, SizeType, N>(ext);
  } else {
    static_assert(ProbingScheme::cg_size > 0);
    static_assert(Storage::bucket_size > 0);

    constexpr auto stride =
      static_cast<std::uint64_t>(ProbingScheme::cg_size) * Storage::bucket_size;
    constexpr auto max_groups = detail::max_extent_value<SizeType>() / stride;

    if constexpr (N == dynamic_extent) {
      auto const value = static_cast<SizeType>(ext);
      auto groups      = detail::extent_div_ceil(detail::normalize_extent(value), stride);
      if (value == 0) { ++groups; }
      if (groups > max_groups) { CUCO_FAIL("Requested extent exceeds the representable capacity"); }
      return valid_extent<SizeType, dynamic_extent>{static_cast<SizeType>(groups * stride)};
    } else {
      static_assert(detail::is_static_extent_representable<SizeType, N>(),
                    "Static extent must be representable by its size type");
      constexpr auto requested = N == 0 ? 1 : N;
      constexpr auto groups =
        detail::extent_div_ceil(requested, stride) + static_cast<std::uint64_t>(N == 0);
      static_assert(groups <= max_groups, "Requested extent exceeds the representable capacity");
      return valid_extent<SizeType, static_cast<std::size_t>(groups * stride)>{};
    }
  }
}

// Overload for ProbingScheme and Storage with SizeType
template <typename ProbingScheme, typename Storage, typename SizeType>
[[nodiscard]] auto constexpr make_valid_extent(extent<SizeType> ext, double desired_load_factor)
{
  CUCO_EXPECTS(desired_load_factor > 0., "Desired occupancy must be larger than zero");
  CUCO_EXPECTS(desired_load_factor <= 1., "Desired occupancy must be no larger than one");

  auto const value = static_cast<SizeType>(ext);
  if constexpr (cuda::std::is_signed_v<SizeType>) {
    if (value <= 0) { return make_valid_extent<ProbingScheme, Storage>(ext); }
  } else {
    if (value == 0) { return make_valid_extent<ProbingScheme, Storage>(ext); }
  }

  auto const temp =
    std::ceil(static_cast<long double>(value) / static_cast<long double>(desired_load_factor));
  if (temp > static_cast<long double>(cuda::std::numeric_limits<SizeType>::max())) {
    CUCO_FAIL(
      "Invalid load factor: requested extent divided by load factor exceeds maximum representable "
      "value");
  }
  return make_valid_extent<ProbingScheme, Storage>(
    cuco::extent<SizeType>{static_cast<SizeType>(temp)});
}

template <typename ProbingScheme, typename Storage, typename SizeType>
[[nodiscard]] auto constexpr make_valid_extent(SizeType size, double desired_load_factor)
{
  return make_valid_extent<ProbingScheme, Storage>(cuco::extent<SizeType>{size},
                                                   desired_load_factor);
}

template <typename ProbingScheme, typename Storage, typename SizeType>
[[nodiscard]] auto constexpr make_valid_extent(SizeType size)
{
  return make_valid_extent<ProbingScheme, Storage, SizeType, dynamic_extent>(
    cuco::extent<SizeType>{size});
}

// Template template parameter overloads for single-type ProbingScheme
template <template <typename> class ProbingScheme,
          typename Storage,
          typename SizeType,
          std::size_t N>
[[nodiscard]] auto constexpr make_valid_extent(extent<SizeType, N> ext)
{
  using ProbeType = ProbingScheme<int>;
  return make_valid_extent<ProbeType, Storage, SizeType, N>(ext);
}

template <template <typename> class ProbingScheme, typename Storage, typename SizeType>
[[nodiscard]] auto constexpr make_valid_extent(SizeType size)
{
  using ProbeType = ProbingScheme<int>;
  return make_valid_extent<ProbeType, Storage, SizeType>(size);
}

// Template template parameter overloads for two-type ProbingScheme
template <template <typename, typename> class ProbingScheme,
          typename Storage,
          typename SizeType,
          std::size_t N>
[[nodiscard]] auto constexpr make_valid_extent(extent<SizeType, N> ext)
{
  using ProbeType = ProbingScheme<int, int>;
  return make_valid_extent<ProbeType, Storage, SizeType, N>(ext);
}

template <template <typename, typename> class ProbingScheme, typename Storage, typename SizeType>
[[nodiscard]] auto constexpr make_valid_extent(SizeType size)
{
  using ProbeType = ProbingScheme<int, int>;
  return make_valid_extent<ProbeType, Storage, SizeType>(size);
}

}  // namespace cuco
