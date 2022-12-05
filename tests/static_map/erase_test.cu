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

#include <catch2/catch.hpp>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <cuco/static_map.cuh>

#include <utils.hpp>

TEMPLATE_TEST_CASE_SIG("erase key", "", ((typename T), T), (int32_t), (int64_t))
{
  using Key   = T;
  using Value = T;

  constexpr std::size_t num_keys = 1'000'000;
  constexpr std::size_t capacity = 1'100'000;

  cuco::static_map<Key, Value> map{
    capacity, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}, cuco::erased_key<Key>{-2}};

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<Value> d_values(num_keys);
  thrust::device_vector<bool> d_keys_exist(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end(), 1);
  thrust::sequence(thrust::device, d_values.begin(), d_values.end(), 1);

  auto pairs_begin =
    thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_values.begin()));

  SECTION("Check basic insert/erase")
  {
    map.insert(pairs_begin, pairs_begin + num_keys);

    REQUIRE(map.get_size() == num_keys);

    map.erase(d_keys.begin(), d_keys.end());

    REQUIRE(map.get_size() == 0);

    map.contains(d_keys.begin(), d_keys.end(), d_keys_exist.begin());

    REQUIRE(cuco::test::none_of(d_keys_exist.begin(),
                                d_keys_exist.end(),
                                [] __device__(const bool key_found) { return key_found; }));

    map.insert(pairs_begin, pairs_begin + num_keys);

    REQUIRE(map.get_size() == num_keys);

    map.contains(d_keys.begin(), d_keys.end(), d_keys_exist.begin());

    REQUIRE(cuco::test::all_of(d_keys_exist.begin(),
                               d_keys_exist.end(),
                               [] __device__(const bool key_found) { return key_found; }));

    map.erase(d_keys.begin(), d_keys.begin() + num_keys / 2);
    map.contains(d_keys.begin(), d_keys.end(), d_keys_exist.begin());

    REQUIRE(cuco::test::none_of(d_keys_exist.begin(),
                                d_keys_exist.begin() + num_keys / 2,
                                [] __device__(const bool key_found) { return key_found; }));

    REQUIRE(cuco::test::all_of(d_keys_exist.begin() + num_keys / 2,
                               d_keys_exist.end(),
                               [] __device__(const bool key_found) { return key_found; }));

    map.erase(d_keys.begin() + num_keys / 2, d_keys.end());
    REQUIRE(map.get_size() == 0);
  }
}
