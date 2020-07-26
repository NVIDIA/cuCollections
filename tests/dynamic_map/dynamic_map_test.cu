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

#include "catch.hpp"

#include <cuco/dynamic_map.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <algorithm>



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


TEMPLATE_TEST_CASE("Unique sequence of keys", "", int32_t, int64_t)
{
  using Key   = TestType;
  using Value = TestType;

  constexpr std::size_t num_keys{50'000'000};
  cuco::dynamic_map<Key, Value> map{30'000'000, -1, -1};

  std::vector<Key> h_keys( num_keys );
  std::vector<Value> h_values( num_keys );
  std::vector<Value> h_results( num_keys );
  std::vector<bool> h_contained( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  
  for(auto i = 0; i < num_keys; ++i) {
    h_keys[i] = (Key)(i + 1);
    h_values[i] = (Value)(i + 1);
    h_pairs[i] = cuco::make_pair((Key)(i + 1), (Value)(i + 1));
  }

  thrust::device_vector<Key> d_keys( h_keys );
  thrust::device_vector<Value> d_results( num_keys );
  thrust::device_vector<bool> d_contained( num_keys );
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  
  // bulk function test cases
  SECTION("All inserted keys-value pairs should be correctly recovered during find")
  {
    map.insert(d_pairs.begin(), d_pairs.end());
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    thrust::copy(d_results.begin(), d_results.end(), h_results.begin());

    auto all_match = true;
    for(auto i = 18'000'000; i < 18'000'000 + 100; ++i) {
      if(h_results[i] != h_values[i]) {
        all_match = false;

        std::cout << "i " << i << std::endl;
        std::cout << "h_res " << h_results[i] << std::endl;
        //break;
      }
    }

    REQUIRE(all_match);
  }

  SECTION("All inserted keys-value pairs should be contained")
  {
    map.insert(d_pairs.begin(), d_pairs.end());
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());
    thrust::copy(d_contained.begin(), d_contained.end(), h_contained.begin());

    auto all_contained = true;
    for(auto i = 0; i < num_keys; ++i) {
      if(!h_contained[i]) {
        all_contained = false;
        break;
      }
    }

    REQUIRE(all_contained);
  }
  
  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());
    thrust::copy(d_contained.begin(), d_contained.end(), h_contained.begin());

    auto none_contained = true;
    for(auto i = 0; i < num_keys; ++i) {
      if(h_contained[i]) {
        none_contained = false;
        break;
      }
    }

    REQUIRE(none_contained);
  }
}

TEMPLATE_TEST_CASE("Uniformly random sequence of keys", "", int32_t, int64_t)
{
  using Key   = TestType;
  using Value = TestType;

  constexpr std::size_t num_keys{50'000'000};
  cuco::dynamic_map<Key, Value> map{30'000'000, -1, -1};

  std::vector<Key> h_keys( num_keys );
  std::vector<Value> h_values( num_keys );
  std::vector<Value> h_results( num_keys );
  std::vector<bool> h_contained( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  
  std::mt19937 rng{12};
  for(auto i = 0; i < num_keys; ++i) {
    Key key = std::abs(static_cast<Key>(rng()));
    Value val = ~key;
    h_keys[i] = key;
    h_values[i] = val;
    h_pairs[i].first = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<Key> d_keys( h_keys ); 
  thrust::device_vector<Value> d_results( num_keys);
  thrust::device_vector<bool> d_contained( num_keys );
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  
  // bulk function test cases
  SECTION("All inserted keys-value pairs should be correctly recovered during find")
  {
    map.insert(d_pairs.begin(), d_pairs.end());
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    thrust::copy(d_results.begin(), d_results.end(), h_results.begin());

    auto all_match = true;
    for(auto i = 0; i < num_keys; ++i) {
      if(h_results[i] != h_values[i]) {
        all_match = false;
        break;
      }
    }

    REQUIRE(all_match);
  }

  SECTION("All inserted keys-value pairs should be contained")
  {
    map.insert(d_pairs.begin(), d_pairs.end());
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());
    thrust::copy(d_contained.begin(), d_contained.end(), h_contained.begin());

    auto all_contained = true;
    for(auto i = 0; i < num_keys; ++i) {
      if(!h_contained[i]) {
        all_contained = false;
        break;
      }
    }

    REQUIRE(all_contained);
  }
  
  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());
    thrust::copy(d_contained.begin(), d_contained.end(), h_contained.begin());

    auto none_contained = true;
    for(auto i = 0; i < num_keys; ++i) {
      if(h_contained[i]) {
        none_contained = false;
        break;
      }
    }

    REQUIRE(none_contained);
  }
}

TEMPLATE_TEST_CASE("Normally distributed random sequence of keys", "", int32_t, int64_t)
{
  using Key   = TestType;
  using Value = TestType;

  constexpr std::size_t num_keys{50'000'000};
  cuco::dynamic_map<Key, Value> map{30'000'000, -1, -1};

  std::vector<Key> h_keys( num_keys );
  std::vector<Value> h_values( num_keys );
  std::vector<Value> h_results( num_keys );
  std::vector<bool> h_contained( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  
  std::mt19937 rng{12};
  std::normal_distribution<> d{1e9, 1e7};

  for(auto i = 0; i < num_keys; ++i) {
    Key key = abs(std::round(d(rng)));
    Value val = ~key;
    h_keys[i] = key;
    h_values[i] = val;
    h_pairs[i].first = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<Key> d_keys( h_keys ); 
  thrust::device_vector<Value> d_results( num_keys);
  thrust::device_vector<bool> d_contained( num_keys );
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );
  
  // bulk function test cases
  SECTION("All inserted keys-value pairs should be correctly recovered during find")
  {
    map.insert(d_pairs.begin(), d_pairs.end());
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    thrust::copy(d_results.begin(), d_results.end(), h_results.begin());

    auto all_match = true;
    for(auto i = 0; i < num_keys; ++i) {
      if(h_results[i] != h_values[i]) {
        all_match = false;
        break;
      }
    }

    REQUIRE(all_match);
  }

  SECTION("All inserted keys-value pairs should be contained")
  {
    map.insert(d_pairs.begin(), d_pairs.end());
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());
    thrust::copy(d_contained.begin(), d_contained.end(), h_contained.begin());

    auto all_contained = true;
    for(auto i = 0; i < num_keys; ++i) {
      if(!h_contained[i]) {
        all_contained = false;
        break;
      }
    }

    REQUIRE(all_contained);
  }
  
  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());
    thrust::copy(d_contained.begin(), d_contained.end(), h_contained.begin());

    auto none_contained = true;
    for(auto i = 0; i < num_keys; ++i) {
      if(h_contained[i]) {
        none_contained = false;
        break;
      }
    }

    REQUIRE(none_contained);
  }
}