/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuco/detail/prime.hpp>

#include <catch2/catch_test_macros.hpp>

#pragma nv_diag_suppress 177  // WAR for weird catch2 warning
TEST_CASE("Prime number computation", "")
{
  REQUIRE(cuco::detail::next_prime(0) == 0);
  REQUIRE(cuco::detail::next_prime(6) == 7);
  REQUIRE(cuco::detail::next_prime(13) == 13);
  REQUIRE(cuco::detail::next_prime(17177758132) == 17177758133);

  // make sure constexpr evaluation is possible
  STATIC_REQUIRE(cuco::detail::next_prime(6) == 7);
}
#pragma nv_diag_default 177
