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
#include <cuco/static_map.cuh>
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

enum class dist_type {
  UNIQUE,
  UNIFORM,
  GAUSSIAN
};

template<dist_type Dist, typename Key, typename OutputIt>
static void generate_keys(OutputIt output_begin, OutputIt output_end) {
  auto num_keys = std::distance(output_begin, output_end);
  
  std::random_device rd;
  std::mt19937 gen{rd()};

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
      std::normal_distribution<> dg{1e9, 1e7};
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = std::abs(static_cast<Key>(dg(gen)));
      }
      break;
  }
}

TEMPLATE_TEST_CASE_SIG("Unique sequence of keys", "", 
  ((typename T, dist_type Dist), T, Dist), 
   (int32_t, dist_type::UNIQUE), (int64_t, dist_type::UNIQUE),
   (int32_t, dist_type::UNIFORM), (int64_t, dist_type::UNIFORM),
   (int32_t, dist_type::GAUSSIAN), (int64_t, dist_type::GAUSSIAN))
{
  using Key   = T;
  using Value = T;

  constexpr std::size_t num_keys{50'000'000};
  cuco::static_map<Key, Value> map{100'000'000, -1, -1};

  auto m_view = map.get_device_mutable_view();
  auto view = map.get_device_view();

  std::vector<Key> h_keys( num_keys );
  std::vector<Value> h_values( num_keys );
  std::vector<Value> h_results( num_keys );
  std::vector<bool> h_contained( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for(auto i = 0; i < num_keys; ++i) {
    Key key = h_keys[i];
    Value val = h_keys[i];
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

    REQUIRE(all_of(d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b;}));
  }
  
  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());

    REQUIRE(none_of(d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b;}));
  }
  
  SECTION("Inserting keys should return valid iterator and insert success.")
  {
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [m_view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
        auto const result = m_view.insert(pair);
        return (result.first != m_view.end()) and (result.second == true);
      }));
  }
  
  SECTION("Dereferenced iterator should equal inserted pair.")
  {
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [m_view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
        auto const result = m_view.insert(pair);
        auto const iter   = result.first;
        return iter->first.load() == pair.first and iter->second.load() == pair.second;
      }));
  }
  
  SECTION("Pair isn't changed after inserting twice.")
  {
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [m_view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
        auto const first_insert  = m_view.insert(pair);
        auto const second_insert = m_view.insert(pair);
        auto const iter          = second_insert.first;
        return iter->first.load() == pair.first and iter->second.load() == pair.second;
      }));
  }

  SECTION("Cannot find any key in an empty hash map")
  {
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
        return view.find(pair.first) == view.end();
      }));
  }

  SECTION("Keys are all found after inserting many keys.")
  {
    // Bulk insert keys
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [m_view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
        return m_view.insert(pair).second;
      }));

    // All keys should be found
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
        auto const found = view.find(pair.first);
        return (found != view.end()) and
               (found->first.load() == pair.first and found->second.load() == pair.second);
      }));
  }

  SECTION("Fetch add on value works")
  {
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [m_view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
        auto result = m_view.insert(pair);
        auto& v     = result.first->second;
        v.fetch_add(42);
        return v.load() == (pair.second + 42);
      }));
  }
}