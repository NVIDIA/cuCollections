/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/static_multiset.cuh>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("static_multiset rehash test", "")
{
  using key_type = int;

  constexpr std::size_t num_keys{400};

  // `static_multiset` defaults to two slots per bucket.
  cuco::static_multiset set{num_keys * 2, cuco::empty_key<key_type>{-1}};
  REQUIRE(decltype(set)::bucket_size == 2);

  thrust::device_vector<key_type> d_keys(num_keys);
  thrust::sequence(d_keys.begin(), d_keys.end());

  set.insert(d_keys.begin(), d_keys.end());
  REQUIRE(set.size() == num_keys);

  thrust::device_vector<bool> d_contained(num_keys);
  auto const check_contents = [&]() {
    REQUIRE(set.size() == num_keys);
    set.contains(d_keys.begin(), d_keys.end(), d_contained.begin());
    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  };

  set.rehash();
  check_contents();

  set.rehash(num_keys * 4);
  check_contents();

  set.rehash_async(num_keys * 8);
  check_contents();
}
