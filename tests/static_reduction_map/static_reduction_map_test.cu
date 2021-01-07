/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cuco/static_reduction_map.cuh>
#include <limits>

namespace {
// Thrust logical algorithms (any_of/all_of/none_of) don't work with device
// lambdas: See https://github.com/thrust/thrust/issues/1062
template <typename Iterator, typename Predicate>
bool all_of(Iterator begin, Iterator end, Predicate p)
{
  auto size = thrust::distance(begin, end);
  return size == thrust::count_if(begin, end, p);
}

template <typename Iterator, typename Predicate>
bool any_of(Iterator begin, Iterator end, Predicate p)
{
  return thrust::count_if(begin, end, p) > 0;
}

template <typename Iterator, typename Predicate>
bool none_of(Iterator begin, Iterator end, Predicate p)
{
  return not all_of(begin, end, p);
}
}  // namespace

TEMPLATE_TEST_CASE_SIG("Insert all identical keys",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t))
{
  thrust::device_vector<Key> keys(100, 42);
  thrust::device_vector<Value> values(keys.size(), 1);

  auto const num_slots{keys.size() * 2};
  cuco::static_reduction_map<cuco::reduce_add<Value>, Key, Value> map{num_slots, -1};

  auto zip     = thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), values.begin()));
  auto zip_end = zip + keys.size();
  map.insert(zip, zip_end);

  SECTION("There should only be one key in the map") { REQUIRE(map.get_size() == 1); }

  SECTION("Map should contain the inserted key")
  {
    thrust::device_vector<bool> contained(keys.size());
    map.contains(keys.begin(), keys.end(), contained.begin());
    REQUIRE(all_of(contained.begin(), contained.end(), [] __device__(bool c) { return c; }));
  }

  SECTION("Found value should equal aggregate of inserted values")
  {
    thrust::device_vector<Value> found(keys.size());
    map.find(keys.begin(), keys.end(), found.begin());
    auto const expected_aggregate = keys.size();  // All keys inserted "1", so the
                                                  // sum aggregate should be
                                                  // equal to the number of keys inserted
    REQUIRE(all_of(found.begin(), found.end(), [expected_aggregate] __device__(Value v) {
      return v == expected_aggregate;
    }));
  }
}

TEMPLATE_TEST_CASE_SIG("Insert all unique keys",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t))
{
  constexpr std::size_t num_keys = 10000;
  constexpr std::size_t num_slots{num_keys * 2};
  cuco::static_reduction_map<cuco::reduce_add<Value>, Key, Value> map{num_slots, -1};

  auto keys_begin   = thrust::make_counting_iterator<Key>(0);
  auto values_begin = thrust::make_counting_iterator<Value>(0);
  auto zip          = thrust::make_zip_iterator(thrust::make_tuple(keys_begin, values_begin));
  auto zip_end      = zip + num_keys;
  map.insert(zip, zip_end);

  SECTION("Size of map should equal number of inserted keys")
  {
    REQUIRE(map.get_size() == num_keys);
  }

  SECTION("Map should contain the inserted keys")
  {
    thrust::device_vector<bool> contained(num_keys);
    map.contains(keys_begin, keys_begin + num_keys, contained.begin());
    REQUIRE(all_of(contained.begin(), contained.end(), [] __device__(bool c) { return c; }));
  }

  SECTION("Found value should equal inserted value")
  {
    thrust::device_vector<Value> found(num_keys);
    map.find(keys_begin, keys_begin + num_keys, found.begin());
    REQUIRE(thrust::equal(thrust::device, values_begin, values_begin + num_keys, found.begin()));
  }
}
