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
#include <random>

enum class dist_type {
  UNIQUE,
  UNIFORM,
  GAUSSIAN,
  SUM_TEST
};

template<dist_type Dist, typename Key, std::size_t num_sum_duplicates = 1, typename OutputIt>
static void generate_keys(OutputIt output_begin, OutputIt output_end) {
  auto num_keys = std::distance(output_begin, output_end);
  
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::normal_distribution<> dg{1e9, 1e7};

  switch(Dist) {
    case dist_type::UNIQUE:
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = i;
      }
      break;
    case dist_type::UNIFORM:
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = std::abs(static_cast<Key>(gen()));
      }
      break;
    case dist_type::GAUSSIAN:
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = std::abs(static_cast<Key>(dg(gen)));
      }
      break;
    case dist_type::SUM_TEST:
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = i / num_sum_duplicates;
      }
      break;
  }
}

template<dist_type Dist, typename Value, typename InputIt, typename OutputIt>
static void generate_values(InputIt first, InputIt last, OutputIt output_begin) {
  auto num_keys = std::distance(first, last);

  switch(Dist) {
    case dist_type::SUM_TEST:
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = 1;
      }
      break;
    case dist_type::UNIQUE:
    case dist_type::UNIFORM:
    case dist_type::GAUSSIAN:
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = first[i];
      }
      break;
  }
}

template <typename KeyIt, typename ValueIt, typename OutputIt>
static void generate_pairs(KeyIt keys_begin, KeyIt keys_end, ValueIt values_begin, OutputIt output_begin) {
  auto num_keys = std::distance(keys_begin, keys_end);

  for(auto i = 0; i < num_keys; ++i) {
    output_begin[i].first = keys_begin[i];
    output_begin[i].second = values_begin[i];
  }
}

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


TEMPLATE_TEST_CASE_SIG("Unique sequence of keys", "", 
  ((typename T, dist_type Dist), T, Dist), 
   (int32_t, dist_type::UNIQUE), (int64_t, dist_type::UNIQUE),
   (int32_t, dist_type::UNIFORM), (int64_t, dist_type::UNIFORM),
   (int32_t, dist_type::GAUSSIAN), (int64_t, dist_type::GAUSSIAN),
   (int32_t, dist_type::SUM_TEST), (int64_t, dist_type::SUM_TEST))
{
  using Key   = T;
  using Value = T;

  constexpr std::size_t num_keys{50'000'000};
  constexpr std::size_t batch_size{1'000'000};
  constexpr std::size_t num_sum_duplicates{16}; /* should divide num_keys */
  cuco::dynamic_map<Key, Value> map{30'000'000, -1, -1};

  std::vector<Key> h_keys( num_keys );
  std::vector<Value> h_values( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  
  generate_keys<Dist, Key, num_sum_duplicates>(h_keys.begin(), h_keys.end());
  generate_values<Dist, Value>(h_keys.begin(), h_keys.end(), h_values.begin());
  generate_pairs(h_keys.begin(), h_keys.end(), h_values.begin(), h_pairs.begin());
  
  thrust::device_vector<Key> d_keys( h_keys );
  thrust::device_vector<Value> d_values( h_values );
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );
  thrust::device_vector<Value> d_results( num_keys );
  thrust::device_vector<bool> d_contained( num_keys );

  // bulk function test cases
  SECTION("All inserted keys-value pairs should be correctly recovered during find")
  {
    map.insert(d_pairs.begin(), d_pairs.end());
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(d_results.begin(), d_values.begin()));
   
    REQUIRE(all_of(zip, zip + num_keys, 
      [] __device__(auto const& p) {
        return thrust::get<0>(p) == thrust::get<1>(p);
      }));
  }
  
  SECTION("All inserted keys-value pairs should be contained")
  {
    map.insert(d_pairs.begin(), d_pairs.end());
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());
    
    REQUIRE(all_of(d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b; }));
  }
  
  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());
    
    REQUIRE(none_of(d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b; }));
  }
  
  SECTION("Bulk insert sum reduce should work")
  {
    if(Dist == dist_type::SUM_TEST) {
      map.insertSumReduce(d_pairs.begin(), d_pairs.end());
      map.find(d_keys.begin(), d_keys.end(), d_results.begin());

      REQUIRE(all_of(d_results.begin(), d_results.end(), [] __device__(auto const& p) { return p == num_sum_duplicates; }));
    }
  }
  
  SECTION("Batched insert sum reduce should work")
  {
    if(Dist == dist_type::SUM_TEST) {
      for(auto i = 0; i < num_keys; i += batch_size) {
        map.insertSumReduce(d_pairs.begin() + i, d_pairs.begin() + i + batch_size);
      }
      map.find(d_keys.begin(), d_keys.end(), d_results.begin());

      REQUIRE(all_of(d_results.begin(), d_results.end(), [] __device__(auto const& p) { return p == num_sum_duplicates; }));
    }
  }

}