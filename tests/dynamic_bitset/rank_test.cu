/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/trie/dynamic_bitset/dynamic_bitset.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <catch2/catch_test_macros.hpp>

using cuco::test::modulo_bitgen;

TEST_CASE("dynamic_bitset rank test", "")
{
  cuco::experimental::detail::dynamic_bitset bv;

  using size_type = std::size_t;
  constexpr size_type num_elements{4000};

  for (size_type i = 0; i < num_elements; i++) {
    bv.push_back(modulo_bitgen(i));
  }

  thrust::device_vector<size_type> keys(num_elements);
  thrust::sequence(keys.begin(), keys.end(), 0);

  thrust::device_vector<size_type> d_ranks(num_elements);

  bv.rank(keys.begin(), keys.end(), d_ranks.begin());

  thrust::host_vector<size_type> h_ranks = d_ranks;

  size_type cur_rank    = 0;
  size_type num_matches = 0;
  for (size_type i = 0; i < num_elements; i++) {
    num_matches += cur_rank == h_ranks[i];
    if (modulo_bitgen(i)) { cur_rank++; }
  }
  REQUIRE(num_matches == num_elements);
}
