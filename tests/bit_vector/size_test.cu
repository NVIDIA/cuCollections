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

#include <cuco/detail/trie/bit_vector/bit_vector.cuh>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Size computation", "")
{
  cuco::experimental::detail::bit_vector bv;
  using size_type = std::size_t;
  constexpr size_type num_elements{400};

  for (size_type i = 0; i < num_elements; i++) {
    bv.append(i % 2 == 0);  // Alternate 0s and 1s pattern
  }
  bv.build();

  auto size = bv.size();
  REQUIRE(size == num_elements);
}
