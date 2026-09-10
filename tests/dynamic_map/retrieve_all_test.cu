/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/__config>
#include <cuco/dynamic_map.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE_SIG("dynamic_map retrieve_all tests",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       // Robin Hood (hard-wired into static_map, which dynamic_map's submaps are)
                       // needs a single-CAS slot: padded int32/int64 and int64/int32 (12B) are
                       // unsupported; int64/int64 (16B) needs 128-bit atomics.
                       (int32_t, int32_t)
// (int32_t, int64_t),
// (int64_t, int32_t),
#if defined(CUCO_HAS_128BIT_ATOMICS)
                         ,
                       (int64_t, int64_t)
#endif
)
{
  constexpr std::size_t num_keys = 1'000'000;
  cuco::dynamic_map<Key, Value> map{num_keys * 2,
                                    cuco::empty_key<Key>{-1},
                                    cuco::empty_value<Value>{-1},
                                    cuco::erased_key<Key>{-2}};

  SECTION("retrieve_all after insert")
  {
    thrust::device_vector<Key> d_keys(num_keys);
    thrust::device_vector<Value> d_values(num_keys);

    thrust::sequence(d_keys.begin(), d_keys.end(), 0);
    thrust::sequence(d_values.begin(), d_values.end(), 0);

    auto pairs = cuda::make_transform_iterator(
      cuda::counting_iterator<std::size_t>{0},
      cuda::proclaim_return_type<cuco::pair<Key, Value>>(
        [keys = d_keys.begin(), values = d_values.begin()] __device__(auto i) {
          return cuco::pair<Key, Value>{keys[i], values[i]};
        }));

    map.insert(pairs, pairs + num_keys);

    REQUIRE(map.size() == num_keys);

    thrust::device_vector<Key> retrieved_keys(num_keys);
    thrust::device_vector<Value> retrieved_values(num_keys);

    auto [keys_out, values_out] =
      map.retrieve_all(retrieved_keys.begin(), retrieved_values.begin());

    REQUIRE(keys_out == retrieved_keys.end());
    REQUIRE(values_out == retrieved_values.end());

    thrust::sort(retrieved_keys.begin(), retrieved_keys.end());
    thrust::sort(retrieved_values.begin(), retrieved_values.end());

    REQUIRE(thrust::equal(d_keys.begin(), d_keys.end(), retrieved_keys.begin()));
    REQUIRE(thrust::equal(d_values.begin(), d_values.end(), retrieved_values.begin()));
  }

  SECTION("retrieve_all after partial erase")
  {
    thrust::device_vector<Key> d_keys(num_keys);
    thrust::device_vector<Value> d_values(num_keys);
    thrust::sequence(d_keys.begin(), d_keys.end(), 0);
    thrust::sequence(d_values.begin(), d_values.end(), 0);

    auto pairs = cuda::make_transform_iterator(
      cuda::counting_iterator<std::size_t>{0},
      cuda::proclaim_return_type<cuco::pair<Key, Value>>(
        [keys = d_keys.begin(), values = d_values.begin()] __device__(auto i) {
          return cuco::pair<Key, Value>{keys[i], values[i]};
        }));

    map.insert(pairs, pairs + num_keys);

    map.erase(d_keys.begin(), d_keys.begin() + num_keys / 2);

    REQUIRE(map.size() == num_keys / 2);

    thrust::device_vector<Key> retrieved_keys(num_keys / 2);
    thrust::device_vector<Value> retrieved_values(num_keys / 2);

    auto [keys_out, values_out] =
      map.retrieve_all(retrieved_keys.begin(), retrieved_values.begin());

    REQUIRE(std::distance(retrieved_keys.begin(), keys_out) == num_keys / 2);
    REQUIRE(std::distance(retrieved_values.begin(), values_out) == num_keys / 2);

    thrust::sort(retrieved_keys.begin(), retrieved_keys.end());
    thrust::sort(retrieved_values.begin(), retrieved_values.end());

    REQUIRE(thrust::equal(d_keys.begin() + num_keys / 2, d_keys.end(), retrieved_keys.begin()));
    REQUIRE(
      thrust::equal(d_values.begin() + num_keys / 2, d_values.end(), retrieved_values.begin()));
  }

  SECTION("retrieve_all on empty map")
  {
    thrust::device_vector<Key> retrieved_keys(0);
    thrust::device_vector<Value> retrieved_values(0);

    auto [keys_out, values_out] =
      map.retrieve_all(retrieved_keys.begin(), retrieved_values.begin());

    REQUIRE(keys_out == retrieved_keys.begin());
    REQUIRE(values_out == retrieved_values.begin());
  }
}
