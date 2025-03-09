/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include <cuco/detail/utility/math.cuh>
#include <cuco/detail/utils.hpp>
#include <cuco/probing_scheme.cuh>
#include <cuco/storage.cuh>
#include <cuco/utility/fast_int.cuh>

#include <cuda/std/type_traits>

namespace cuco {

template <typename SizeType, std::size_t N>
struct bucket_extent {
  using value_type = SizeType;  ///< Extent value type

  __host__ __device__ constexpr value_type value() const noexcept { return N; }
  __host__ __device__ explicit constexpr operator value_type() const noexcept { return value(); }

 private:
  __host__ __device__ explicit constexpr bucket_extent() noexcept {}
  __host__ __device__ explicit constexpr bucket_extent(SizeType) noexcept {}

  template <int32_t CGSize_, int32_t BucketSize_, typename SizeType_, std::size_t N_>
  friend auto constexpr make_bucket_extent(extent<SizeType_, N_> ext);

  template <typename ProbingScheme, typename Storage, typename SizeType_, std::size_t N_>
  friend auto constexpr make_bucket_extent(extent<SizeType_, N_> ext);

  template <typename Rhs>
  friend __host__ __device__ constexpr value_type operator-(bucket_extent const& lhs,
                                                            Rhs rhs) noexcept
  {
    return lhs.value() - rhs;
  }

  template <typename Rhs>
  friend __host__ __device__ constexpr value_type operator/(bucket_extent const& lhs,
                                                            Rhs rhs) noexcept
  {
    return lhs.value() / rhs;
  }

  template <typename Lhs>
  friend __host__ __device__ constexpr value_type operator%(Lhs lhs,
                                                            bucket_extent const& rhs) noexcept
  {
    return lhs % rhs.value();
    ;
  }
};

template <typename SizeType>
struct bucket_extent<SizeType, dynamic_extent> : cuco::utility::fast_int<SizeType> {
  using value_type =
    typename cuco::utility::fast_int<SizeType>::fast_int::value_type;  ///< Extent value type

 private:
  using cuco::utility::fast_int<SizeType>::fast_int;

  template <int32_t CGSize_, int32_t BucketSize_, typename SizeType_, std::size_t N_>
  friend auto constexpr make_bucket_extent(extent<SizeType_, N_> ext);

  template <typename ProbingScheme, typename Storage, typename SizeType_, std::size_t N_>
  friend auto constexpr make_bucket_extent(extent<SizeType_, N_> ext);
};

template <int32_t CGSize, int32_t BucketSize, typename SizeType, std::size_t N>
[[nodiscard]] auto constexpr make_bucket_extent(extent<SizeType, N> ext)
{
  auto constexpr max_prime = cuco::detail::primes.back();
  auto constexpr max_value =
    (static_cast<uint64_t>(std::numeric_limits<SizeType>::max()) < max_prime)
      ? std::numeric_limits<SizeType>::max()
      : static_cast<SizeType>(max_prime);
  auto const size = cuco::detail::int_div_ceil(
    std::max(static_cast<SizeType>(ext), static_cast<SizeType>(1)), CGSize * BucketSize);
  if (size > max_value) { CUCO_FAIL("Invalid input extent"); }

  if constexpr (N == dynamic_extent) {
    return bucket_extent<SizeType>{static_cast<SizeType>(
      *cuco::detail::lower_bound(
        cuco::detail::primes.begin(), cuco::detail::primes.end(), static_cast<uint64_t>(size)) *
      CGSize)};
  }
  if constexpr (N != dynamic_extent) {
    return bucket_extent<SizeType,
                         static_cast<std::size_t>(
                           *cuco::detail::lower_bound(cuco::detail::primes.begin(),
                                                      cuco::detail::primes.end(),
                                                      static_cast<uint64_t>(size)) *
                           CGSize)>{};
  }
}

template <int32_t CGSize, int32_t BucketSize, typename SizeType>
[[nodiscard]] auto constexpr make_bucket_extent(SizeType size)
{
  return make_bucket_extent<CGSize, BucketSize, SizeType, dynamic_extent>(extent<SizeType>{size});
}

template <typename ProbingScheme, typename Storage, typename SizeType, std::size_t N>
[[nodiscard]] auto constexpr make_bucket_extent(extent<SizeType, N> ext)
{
  if constexpr (cuco::is_double_hashing<ProbingScheme>::value) {
    return make_bucket_extent<ProbingScheme::cg_size, Storage::bucket_size, SizeType, N>(ext);
  } else {
    auto const size =
      cuco::detail::int_div_ceil(std::max(static_cast<SizeType>(ext), static_cast<SizeType>(1)),
                                 ProbingScheme::cg_size * Storage::bucket_size) +
      static_cast<SizeType>(ext == 0);

    if constexpr (N == dynamic_extent) {
      return bucket_extent<SizeType>{size * ProbingScheme::cg_size};
    } else {
      return bucket_extent<SizeType, size * ProbingScheme::cg_size>{};
    }
  }
}

template <typename ProbingScheme, typename Storage, typename SizeType>
[[nodiscard]] auto constexpr make_bucket_extent(SizeType size)
{
  return make_bucket_extent<ProbingScheme, Storage, SizeType, dynamic_extent>(
    cuco::extent<SizeType>{size});
}

template <typename Container, typename SizeType, std::size_t N>
[[nodiscard]] auto constexpr make_bucket_extent(extent<SizeType, N> ext)
{
  return make_bucket_extent<typename Container::probing_scheme_type,
                            typename Container::storage_ref_type,
                            SizeType,
                            N>(ext);
}

template <typename Container, typename SizeType>
[[nodiscard]] auto constexpr make_bucket_extent(SizeType size)
{
  return make_bucket_extent<typename Container::probing_scheme_type,
                            typename Container::storage_ref_type,
                            SizeType,
                            dynamic_extent>(extent<SizeType>{size});
}

namespace detail {

template <typename...>
struct is_bucket_extent : cuda::std::false_type {};

template <typename SizeType, std::size_t N>
struct is_bucket_extent<bucket_extent<SizeType, N>> : cuda::std::true_type {};

template <typename T>
inline constexpr bool is_bucket_extent_v = is_bucket_extent<T>::value;

}  // namespace detail
}  // namespace cuco
