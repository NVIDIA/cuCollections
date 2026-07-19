/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/detail/trie/dynamic_bitset/dynamic_bitset.cuh>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("dynamic_bitset size computation test", "")
{
  cuco::experimental::detail::dynamic_bitset bv;
  using size_type = std::size_t;
  constexpr size_type num_elements{400};

  for (size_type i = 0; i < num_elements; i++) {
    bv.push_back(i % 2 == 0);  // Alternate 0s and 1s pattern
  }

  auto size = bv.size();
  REQUIRE(size == num_elements);
}
