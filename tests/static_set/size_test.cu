/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/static_set.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("static_set size test", "")
{
  constexpr std::size_t num_keys{400};

  cuco::static_set<int> set{cuco::extent<std::size_t>{400}, cuco::empty_key{-1}};

  thrust::device_vector<int> d_keys(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());

  auto const num_successes = set.insert(d_keys.begin(), d_keys.end());

  REQUIRE(set.size() == num_keys);
  REQUIRE(num_successes == num_keys);

  set.clear();

  REQUIRE(set.size() == 0);
}
