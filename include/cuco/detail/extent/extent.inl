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
#include <cuco/detail/utils.hpp>
#include <cuco/utility/fast_int.cuh>

namespace cuco {
namespace experimental {

template <typename SizeType, std::size_t N = dynamic_extent>
struct valid_extent {
  using value_type = SizeType;  ///< Extent value type

  __host__ __device__ constexpr value_type value() const noexcept { return N; }
  __host__ __device__ explicit constexpr operator value_type() const noexcept { return value(); }

 private:
  __host__ __device__ explicit constexpr valid_extent() noexcept {}
  __host__ __device__ explicit constexpr valid_extent(SizeType) noexcept {}

  template <int32_t CGSize, int32_t WindowSize, typename SizeType_, std::size_t N_>
  friend auto constexpr make_valid_extent(extent<SizeType_, N_> ext);
};

template <typename SizeType>
struct valid_extent<SizeType, dynamic_extent> : cuco::utility::fast_int<SizeType> {
  using value_type =
    typename cuco::utility::fast_int<SizeType>::fast_int::value_type;  ///< Extent value type

 private:
  using cuco::utility::fast_int<SizeType>::fast_int;

  template <int32_t CGSize, int32_t WindowSize, typename SizeType_, std::size_t N_>
  friend auto constexpr make_valid_extent(extent<SizeType_, N_> ext);
};

template <int32_t CGSize, int32_t WindowSize, typename SizeType, std::size_t N>
[[nodiscard]] auto constexpr make_valid_extent(extent<SizeType, N> ext)
{
  auto constexpr max_prime = cuco::detail::primes.back();
  auto constexpr max_value =
    (static_cast<uint64_t>(std::numeric_limits<SizeType>::max()) < max_prime)
      ? std::numeric_limits<SizeType>::max()
      : static_cast<SizeType>(max_prime);
  auto const size = SDIV(ext, CGSize * WindowSize);
  if (size <= 0 or size > max_value) { CUCO_FAIL("Invalid input extent"); }

  if constexpr (N == dynamic_extent) {
    return valid_extent<SizeType>{static_cast<SizeType>(
      *cuco::detail::lower_bound(
        cuco::detail::primes.begin(), cuco::detail::primes.end(), static_cast<uint64_t>(size)) *
      CGSize)};
  }
  if constexpr (N != dynamic_extent) {
    return valid_extent<SizeType,
                        static_cast<std::size_t>(
                          *cuco::detail::lower_bound(cuco::detail::primes.begin(),
                                                     cuco::detail::primes.end(),
                                                     static_cast<uint64_t>(size)) *
                          CGSize)>{};
  }
}

}  // namespace experimental
}  // namespace cuco