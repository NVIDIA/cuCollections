/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/static_map.cuh>

#include <cuda/functional>
#include <cuda/iterator>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("static_map rehash test", "")
{
  using key_type    = int;
  using mapped_type = long;

  constexpr std::size_t num_keys{400};
  constexpr std::size_t num_erased_keys{100};

  cuco::static_map map{num_keys,
                       cuco::empty_key<key_type>{-1},
                       cuco::empty_value<mapped_type>{-1},
                       cuco::erased_key<key_type>{-2}};

  auto keys_begin = cuda::counting_iterator<key_type>(1);

  auto pairs_begin = cuda::make_transform_iterator(
    keys_begin,
    cuda::proclaim_return_type<cuco::pair<key_type, mapped_type>>([] __device__(key_type const& x) {
      return cuco::pair<key_type, mapped_type>(x, static_cast<mapped_type>(x));
    }));

  map.insert(pairs_begin, pairs_begin + num_keys);

  map.rehash();
  REQUIRE(map.size() == num_keys);

  map.rehash(num_keys * 2);
  REQUIRE(map.size() == num_keys);

  map.erase(keys_begin, keys_begin + num_erased_keys);
  map.rehash();
  REQUIRE(map.size() == num_keys - num_erased_keys);
}
