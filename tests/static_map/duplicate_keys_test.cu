/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <utils.hpp>

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE_SIG("Duplicate keys",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t),
                       (int32_t, int64_t),
                       (int64_t, int32_t),
                       (int64_t, int64_t))
{
  constexpr std::size_t num_keys{500'000};
  cuco::static_map<Key, Value> map{
    num_keys * 2, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<Value> d_values(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  thrust::sequence(thrust::device, d_values.begin(), d_values.end());

  auto pairs_begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator<int>(0),
    [] __device__(auto i) { return cuco::pair_type<Key, Value>(i / 2, i / 2); });

  thrust::device_vector<Value> d_results(num_keys);
  thrust::device_vector<bool> d_contained(num_keys);

  SECTION("Retrieve all entries")
  {
    auto constexpr gold = num_keys / 2;
    thrust::device_vector<Key> unique_keys(gold);
    thrust::device_vector<Key> unique_values(gold);

    // Retrieve all from an empty map
    auto [empty_key_end, empty_value_end] =
      map.retrieve_all(unique_keys.begin(), unique_values.begin());
    REQUIRE(std::distance(unique_keys.begin(), empty_key_end) == 0);
    REQUIRE(std::distance(unique_values.begin(), empty_value_end) == 0);

    map.insert(pairs_begin, pairs_begin + num_keys);

    auto const num_entries = map.get_size();
    REQUIRE(num_entries == gold);

    auto [key_out_end, value_out_end] =
      map.retrieve_all(unique_keys.begin(), unique_values.begin());
    REQUIRE(std::distance(unique_keys.begin(), key_out_end) == gold);
    REQUIRE(std::distance(unique_values.begin(), value_out_end) == gold);

    thrust::sort(thrust::device, unique_keys.begin(), unique_keys.end());
    REQUIRE(cuco::test::equal(unique_keys.begin(),
                              unique_keys.end(),
                              thrust::make_counting_iterator<Key>(0),
                              thrust::equal_to<Key>{}));
  }

  SECTION("Tests of contains")
  {
    map.insert(pairs_begin, pairs_begin + num_keys);
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());

    REQUIRE(cuco::test::all_of(
      d_contained.begin(), d_contained.begin() + num_keys / 2, thrust::identity{}));

    REQUIRE(cuco::test::none_of(
      d_contained.begin() + num_keys / 2, d_contained.end(), thrust::identity{}));
  }
}
