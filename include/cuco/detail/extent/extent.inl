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

#include <cuco/detail/error.hpp>
#include <cuco/detail/prime.hpp>  // TODO move to detail/extent/
#include <cuco/detail/utility/math.hpp>
#include <cuco/detail/utils.hpp>
#include <cuco/utility/fast_int.cuh>

#include <type_traits>

namespace cuco {
namespace experimental {

template <int32_t CGSize, int32_t WindowSize, typename SizeType, std::size_t N = dynamic_extent>
struct window_extent {
  using value_type = SizeType;  ///< Extent value type

  static auto constexpr cg_size     = CGSize;
  static auto constexpr window_size = WindowSize;

  __host__ __device__ constexpr value_type value() const noexcept { return N; }
  __host__ __device__ explicit constexpr operator value_type() const noexcept { return value(); }

 private:
  __host__ __device__ explicit constexpr window_extent() noexcept {}
  __host__ __device__ explicit constexpr window_extent(SizeType) noexcept {}

  template <int32_t CGSize_, int32_t WindowSize_, typename SizeType_, std::size_t N_>
  friend auto constexpr make_window_extent(extent<SizeType_, N_> ext);
};

template <int32_t CGSize, int32_t WindowSize, typename SizeType>
struct window_extent<CGSize, WindowSize, SizeType, dynamic_extent>
  : cuco::utility::fast_int<SizeType> {
  using value_type =
    typename cuco::utility::fast_int<SizeType>::fast_int::value_type;  ///< Extent value type

  static auto constexpr cg_size     = CGSize;
  static auto constexpr window_size = WindowSize;

 private:
  using cuco::utility::fast_int<SizeType>::fast_int;

  template <int32_t CGSize_, int32_t WindowSize_, typename SizeType_, std::size_t N_>
  friend auto constexpr make_window_extent(extent<SizeType_, N_> ext);
};

template <typename Container, typename SizeType, std::size_t N>
[[nodiscard]] auto constexpr make_window_extent(extent<SizeType, N> ext)
{
  return make_window_extent<Container::cg_size, Container::window_size>(ext);
}

template <typename Container>
[[nodiscard]] std::size_t constexpr make_window_extent(std::size_t size)
{
  return make_window_extent<Container::cg_size, Container::window_size>(size);
}

template <int32_t CGSize, int32_t WindowSize, typename SizeType, std::size_t N>
[[nodiscard]] auto constexpr make_window_extent(extent<SizeType, N> ext)
{
  auto constexpr max_prime = cuco::detail::primes.back();
  auto constexpr max_value =
    (static_cast<uint64_t>(std::numeric_limits<SizeType>::max()) < max_prime)
      ? std::numeric_limits<SizeType>::max()
      : static_cast<SizeType>(max_prime);
  auto const size = cuco::detail::int_div_ceil(
    std::max(static_cast<SizeType>(ext), static_cast<SizeType>(1)), CGSize * WindowSize);
  if (size > max_value) { CUCO_FAIL("Invalid input extent"); }

  if constexpr (N == dynamic_extent) {
    return window_extent<CGSize, WindowSize, SizeType>{static_cast<SizeType>(
      *cuco::detail::lower_bound(
        cuco::detail::primes.begin(), cuco::detail::primes.end(), static_cast<uint64_t>(size)) *
      CGSize)};
  }
  if constexpr (N != dynamic_extent) {
    return window_extent<CGSize,
                         WindowSize,
                         SizeType,
                         static_cast<std::size_t>(
                           *cuco::detail::lower_bound(cuco::detail::primes.begin(),
                                                      cuco::detail::primes.end(),
                                                      static_cast<uint64_t>(size)) *
                           CGSize)>{};
  }
}

template <int32_t CGSize, int32_t WindowSize>
[[nodiscard]] std::size_t constexpr make_window_extent(std::size_t size)
{
  return static_cast<std::size_t>(make_window_extent<CGSize, WindowSize>(extent{size}));
}

namespace detail {

template <typename...>
struct is_window_extent : std::false_type {
};

template <int32_t CGSize, int32_t WindowSize, typename SizeType, std::size_t N>
struct is_window_extent<window_extent<CGSize, WindowSize, SizeType, N>> : std::true_type {
};

template <typename T>
inline constexpr bool is_window_extent_v = is_window_extent<T>::value;

}  // namespace detail

}  // namespace experimental
}  // namespace cuco
