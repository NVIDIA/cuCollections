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

#include <test_utils.hpp>

#include <cuco/extent.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/storage.cuh>

#include <catch2/catch_template_test_macros.hpp>

auto constexpr cg_size     = 2;
auto constexpr bucket_size = 4;

using storage_t = cuco::storage<bucket_size>;
template <typename H1, typename H2>
using probing_t = cuco::double_hashing<cg_size, H1, H2>;

TEMPLATE_TEST_CASE_SIG(
  "utility extent tests", "", ((typename SizeType), SizeType), (int32_t), (int64_t), (std::size_t))
{
  SizeType constexpr num            = 1234;
  SizeType constexpr gold_reference = 314;  // 157 x 2

  SECTION("Static extent must be evaluated at compile time.")
  {
    auto const size = cuco::extent<SizeType, num>{};
    STATIC_REQUIRE(num == size);
  }

  SECTION("Dynamic extent is evaluated at run time.")
  {
    auto const size = cuco::extent(num);
    REQUIRE(size == num);
  }

  SECTION("Compute static valid extent at compile time.")
  {
    auto constexpr size = cuco::extent<SizeType, num>{};
    auto constexpr res  = cuco::make_bucket_extent<probing_t, storage_t>(size);
    STATIC_REQUIRE(gold_reference == res.value());
  }

  SECTION("Compute dynamic valid extent at run time.")
  {
    auto const size = cuco::extent<SizeType>{num};
    auto const res  = cuco::make_bucket_extent<probing_t, storage_t>(size);
    REQUIRE(gold_reference == res.value());
  }
}
