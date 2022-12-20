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

#include <cuco/dynamic_map.cuh>
#include <utils.hpp>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE_SIG("erase key",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t),
                       (int32_t, int64_t),
                       (int64_t, int32_t),
                       (int64_t, int64_t))
{
  unsigned long num_keys = 1'000'000;
  cuco::dynamic_map<Key, Value> map{num_keys * 2,
                                    cuco::empty_key<Key>{-1},
                                    cuco::empty_value<Value>{-1},
                                    cuco::erased_key<Key>{-2}};

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<Value> d_values(num_keys);
  thrust::device_vector<bool> d_keys_exist(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end(), 1);
  thrust::sequence(thrust::device, d_values.begin(), d_values.end(), 1);

  auto pairs_begin =
    thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_values.begin()));

  SECTION("Check basic insert/erase")
  {
    // *****************************************
    // first, check single submap works properly
    // *****************************************

    map.insert(pairs_begin, pairs_begin + num_keys);

    REQUIRE(map.get_size() == num_keys);

    map.erase(d_keys.begin(), d_keys.end());

    // delete decreases count correctly
    REQUIRE(map.get_size() == 0);

    map.contains(d_keys.begin(), d_keys.end(), d_keys_exist.begin());

    // keys were actaully deleted
    REQUIRE(cuco::test::none_of(d_keys_exist.begin(),
                                d_keys_exist.end(),
                                [] __device__(const bool key_found) { return key_found; }));

    // ensures that map is reusing deleted slots
    map.insert(pairs_begin, pairs_begin + num_keys);

    REQUIRE(map.get_size() == num_keys);

    map.contains(d_keys.begin(), d_keys.end(), d_keys_exist.begin());

    REQUIRE(cuco::test::all_of(d_keys_exist.begin(),
                               d_keys_exist.end(),
                               [] __device__(const bool key_found) { return key_found; }));

    // erase can act selectively
    map.erase(d_keys.begin(), d_keys.begin() + num_keys / 2);
    map.contains(d_keys.begin(), d_keys.end(), d_keys_exist.begin());

    REQUIRE(cuco::test::none_of(d_keys_exist.begin(),
                                d_keys_exist.begin() + num_keys / 2,
                                [] __device__(const bool key_found) { return key_found; }));

    REQUIRE(cuco::test::all_of(d_keys_exist.begin() + num_keys / 2,
                               d_keys_exist.end(),
                               [] __device__(const bool key_found) { return key_found; }));

    // clear map
    map.erase(d_keys.begin() + num_keys / 2, d_keys.end());

    // *************************************************
    // second, check multiple submaps case works properly
    // *************************************************

    thrust::device_vector<Key> d_keys2(4 * num_keys);
    thrust::device_vector<Value> d_values2(4 * num_keys);
    thrust::device_vector<bool> d_keys_exist2(4 * num_keys);

    thrust::sequence(thrust::device, d_keys2.begin(), d_keys2.end(), 1);
    thrust::sequence(thrust::device, d_values2.begin(), d_values2.end(), 1);

    auto pairs_begin2 =
      thrust::make_zip_iterator(thrust::make_tuple(d_keys2.begin(), d_values2.begin()));

    map.insert(pairs_begin2, pairs_begin2 + 4 * num_keys);

    // map should resize twice if the erased slots are successfully reused
    REQUIRE(map.get_capacity() == 8 * num_keys);
    // check that keys can be successfully deleted from only the first and second submaps
    map.erase(d_keys2.begin(), d_keys2.begin() + 2 * num_keys);
    map.contains(d_keys2.begin(), d_keys2.end(), d_keys_exist2.begin());

    REQUIRE(cuco::test::none_of(d_keys_exist2.begin(),
                                d_keys_exist2.begin() + 2 * num_keys,
                                [] __device__(const bool key_found) { return key_found; }));

    REQUIRE(cuco::test::all_of(d_keys_exist2.begin() + 2 * num_keys,
                               d_keys_exist2.end(),
                               [] __device__(const bool key_found) { return key_found; }));

    REQUIRE(map.get_size() == 2 * num_keys);
    // check that keys can be successfully deleted from all submaps (some will be unsuccessful
    // erases)
    map.erase(d_keys2.begin(), d_keys2.end());

    map.contains(d_keys2.begin(), d_keys2.end(), d_keys_exist2.begin());

    REQUIRE(cuco::test::none_of(d_keys_exist2.begin(),
                                d_keys_exist2.end(),
                                [] __device__(const bool key_found) { return key_found; }));

    REQUIRE(map.get_size() == 0);
  }
}
