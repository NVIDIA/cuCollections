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


TEMPLATE_TEST_CASE("Unique sequence of keys", "", int32_t, int64_t)
{
  using Key   = TestType;
  using Value = TestType;

  constexpr std::size_t num_keys{50'000'000};
  cuco::static_map<Key, Value> map{100'000'000, -1, -1};

  auto m_view = map.get_device_mutable_view();
  auto view = map.get_device_view();

  std::vector<Key> h_keys( num_keys );
  std::vector<Value> h_values( num_keys );
  std::vector<Value> h_results( num_keys );
  std::vector<bool> h_contained( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  
  for(auto i = 0; i < num_keys; ++i) {
    h_keys[i] = (Key)i;
    h_values[i] = (Value)i;
    h_pairs[i] = cuco::make_pair((Key)i, (Value)i);
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
  
  SECTION("Inserting keys should return valid iterator and insert success.")
  {
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [m_view] __device__(cuco::pair<Key, Value> const& pair) mutable {
        auto const result = m_view.insert(pair);
        return (result.first != m_view.end()) and (result.second == true);
      }));
  }
  
  SECTION("Dereferenced iterator should equal inserted pair.")
  {
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [m_view] __device__(cuco::pair<Key, Value> const& pair) mutable {
        auto const result = m_view.insert(pair);
        auto const iter   = result.first;
        return iter->first.load() == pair.first and iter->second.load() == pair.second;
      }));
  }
  
  SECTION("Key is found immediately after insertion.")
  {
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [m_view, view] __device__(cuco::pair<Key, Value> const& pair) mutable {
        auto const insert_result = m_view.insert(pair);
        auto const find_result   = view.find(pair.first);
        bool const same_iterator = (insert_result.first == find_result);
        return same_iterator;
      }));
  }
  
  SECTION("Inserting same key twice.")
  {
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [m_view, view] __device__(cuco::pair<Key, Value> const& pair) mutable {
        auto const first_insert  = m_view.insert(pair);
        auto const second_insert = m_view.insert(pair);
        auto const find_result   = view.find(pair.first);
        bool const same_iterator = (first_insert.first == second_insert.first);

        // Inserting the same key twice should return the same
        // iterator and false for the insert result
        return same_iterator and (second_insert.second == false);
      }));
  }
  
  SECTION("Pair isn't changed after inserting twice.")
  {
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [m_view] __device__(cuco::pair<Key, Value> const& pair) mutable {
        auto const first_insert  = m_view.insert(pair);
        auto const second_insert = m_view.insert(pair);
        auto const iter          = second_insert.first;
        return iter->first.load() == pair.first and iter->second.load() == pair.second;
      }));
  }

  SECTION("Cannot find any key in an empty hash map")
  {
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [view] __device__(cuco::pair<Key, Value> const& pair) mutable {
        return view.find(pair.first) == view.end();
      }));
  }

  SECTION("Keys are all found after inserting many keys.")
  {
    // Bulk insert keys
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [m_view] __device__(cuco::pair<Key, Value> const& pair) mutable {
        return m_view.insert(pair).second;
      }));

    // All keys should be found
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [view] __device__(cuco::pair<Key, Value> const& pair) mutable {
        auto const found = view.find(pair.first);
        return (found != view.end()) and
               (found->first.load() == pair.first and found->second.load() == pair.second);
      }));
  }

  SECTION("Fetch add on value works")
  {
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [m_view] __device__(cuco::pair<Key, Value> const& pair) mutable {
        auto result = m_view.insert(pair);
        auto& v     = result.first->second;
        v.fetch_add(42);
        return v.load() == (pair.second + 42);
      }));
  }
}