/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/static_set.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("static_set rehash test", "")
{
  using key_type = int;

  constexpr std::size_t num_keys{400};
  constexpr std::size_t num_erased_keys{100};

  cuco::static_set set{num_keys, cuco::empty_key<key_type>{-1}, cuco::erased_key<key_type>{-2}};

  thrust::device_vector<key_type> d_keys(num_keys);

  thrust::sequence(d_keys.begin(), d_keys.end());

  set.insert(d_keys.begin(), d_keys.end());

  set.rehash();
  REQUIRE(set.size() == num_keys);

  set.rehash(num_keys * 2);
  REQUIRE(set.size() == num_keys);

  set.erase(d_keys.begin(), d_keys.begin() + num_erased_keys);
  set.rehash();
  REQUIRE(set.size() == num_keys - num_erased_keys);
}
