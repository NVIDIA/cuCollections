/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/static_multimap.cuh>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("static_multimap rehash test", "")
{
  using key_type   = int;
  using value_type = int;
  using pair_type  = cuco::pair<key_type, value_type>;

  constexpr std::size_t num_unique_keys{400};
  constexpr std::size_t multiplicity{2};
  constexpr std::size_t num_keys{num_unique_keys * multiplicity};

  // `static_multimap` defaults to two slots per bucket.
  cuco::static_multimap map{
    num_keys * 2, cuco::empty_key<key_type>{-1}, cuco::empty_value<value_type>{-1}};
  REQUIRE(decltype(map)::bucket_size == 2);

  thrust::device_vector<key_type> d_keys(num_unique_keys);
  thrust::sequence(d_keys.begin(), d_keys.end());

  thrust::device_vector<pair_type> d_pairs(num_unique_keys);
  thrust::transform(d_keys.begin(), d_keys.end(), d_pairs.begin(), [] __device__(key_type key) {
    return pair_type{key, key};
  });

  for (std::size_t i = 0; i < multiplicity; ++i) {
    map.insert(d_pairs.begin(), d_pairs.end());
  }
  REQUIRE(map.size() == num_keys);

  thrust::device_vector<bool> d_contained(num_unique_keys);
  auto const check_contents = [&]() {
    REQUIRE(map.size() == num_keys);
    REQUIRE(map.count(d_keys.begin(), d_keys.end()) == num_keys);

    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());
    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  };

  map.rehash();
  check_contents();

  map.rehash(num_keys * 4);
  check_contents();

  map.rehash_async(num_keys * 8);
  check_contents();
}
