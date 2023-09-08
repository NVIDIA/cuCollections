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

#include <utils.hpp>

#include <cuco/detail/__config>
#include <cuco/detail/utils.hpp>
#include <cuco/extent.cuh>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cuda/std/array>
#include <limits>

using T                        = uint64_t;
__device__ auto constexpr data = cuda::std::array<T, 6>{1, 2, 3, 5, 6, 7};

constexpr T none = std::numeric_limits<T>::max();  // denotes a missing value

template <size_t N>
constexpr T compute_host_infimum(cuco::experimental::extent<T, N> extent)
{
  auto const res = cuco::detail::infimum(data.begin(), data.end(), extent);

  return (res != data.end()) ? *res : none;
}

template <size_t N>
constexpr T compute_host_supremum(cuco::experimental::extent<T, N> extent)
{
  auto const res = cuco::detail::supremum(data.begin(), data.end(), extent);

  return (res != data.end()) ? *res : none;
}

TEST_CASE("Infimum computation", "")
{
  SECTION("Check if host-generated infimum is correct.")
  {
    CHECK(compute_host_infimum(cuco::experimental::extent<T>{0}) == none);
    CHECK(compute_host_infimum(cuco::experimental::extent<T>{1}) == 1);
    CHECK(compute_host_infimum(cuco::experimental::extent<T>{4}) == 3);
    CHECK(compute_host_infimum(cuco::experimental::extent<T>{7}) == 7);
    CHECK(compute_host_infimum(cuco::experimental::extent<T>{8}) == 7);
  }

  SECTION("Check if constexpr infimum is correct.")
  {
    STATIC_REQUIRE(compute_host_infimum(cuco::experimental::extent<T, 0>{}) == none);
    STATIC_REQUIRE(compute_host_infimum(cuco::experimental::extent<T, 1>{}) == 1);
    STATIC_REQUIRE(compute_host_infimum(cuco::experimental::extent<T, 4>{}) == 3);
    STATIC_REQUIRE(compute_host_infimum(cuco::experimental::extent<T, 7>{}) == 7);
    STATIC_REQUIRE(compute_host_infimum(cuco::experimental::extent<T, 8>{}) == 7);
  }

  // TODO device test
}

TEST_CASE("Supremum computation", "")
{
  SECTION("Check if host-generated supremum is correct.")
  {
    CHECK(compute_host_supremum(cuco::experimental::extent<T>{0}) == 1);
    CHECK(compute_host_supremum(cuco::experimental::extent<T>{1}) == 1);
    CHECK(compute_host_supremum(cuco::experimental::extent<T>{4}) == 5);
    CHECK(compute_host_supremum(cuco::experimental::extent<T>{7}) == 7);
    CHECK(compute_host_supremum(cuco::experimental::extent<T>{8}) == none);
  }

  SECTION("Check if constexpr supremum is correct.")
  {
    STATIC_REQUIRE(compute_host_supremum(cuco::experimental::extent<T, 0>{}) == 1);
    STATIC_REQUIRE(compute_host_supremum(cuco::experimental::extent<T, 1>{}) == 1);
    STATIC_REQUIRE(compute_host_supremum(cuco::experimental::extent<T, 4>{}) == 5);
    STATIC_REQUIRE(compute_host_supremum(cuco::experimental::extent<T, 7>{}) == 7);
    STATIC_REQUIRE(compute_host_supremum(cuco::experimental::extent<T, 8>{}) == none);
  }

  // TODO device test
}
