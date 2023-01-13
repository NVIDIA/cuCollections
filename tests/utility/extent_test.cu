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

#include <cuco/extent.cuh>

#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE_SIG(
  "Extent tests", "", ((typename SizeType), SizeType), (int32_t), (int64_t), (std::size_t))
{
  SizeType constexpr num = 1234;

  SECTION("Static extent must be evaluated at compile time.")
  {
    auto const size = cuco::experimental::extent<SizeType, num>{};
    static_assert(num == size);
  }

  SECTION("Dynamic extent is evaluated at run time.")
  {
    auto const size = cuco::experimental::extent(num);
    REQUIRE(size == num);
  }
}
