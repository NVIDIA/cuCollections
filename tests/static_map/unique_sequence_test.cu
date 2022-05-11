/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE_SIG("Unique sequence of keys",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t),
                       (int32_t, int64_t),
                       (int64_t, int32_t),
                       (int64_t, int64_t))
{
  constexpr std::size_t num_keys{500'000};
  cuco::static_map<Key, Value> map{
    1'000'000, cuco::sentinel::empty_key<Key>{-1}, cuco::sentinel::empty_value<Value>{-1}};

  auto m_view = map.get_device_mutable_view();
  auto view   = map.get_device_view();

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<Value> d_values(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  thrust::sequence(thrust::device, d_values.begin(), d_values.end());

  auto pairs_begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator<int>(0),
    [] __device__(auto i) { return cuco::pair_type<Key, Value>(i, i); });

  thrust::device_vector<Value> d_results(num_keys);
  thrust::device_vector<bool> d_contained(num_keys);

  // bulk function test cases
  SECTION("All inserted keys-value pairs should be correctly recovered during find")
  {
    map.insert(pairs_begin, pairs_begin + num_keys);
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(d_results.begin(), d_values.begin()));

    REQUIRE(cuco::test::all_of(zip, zip + num_keys, [] __device__(auto const& p) {
      return thrust::get<0>(p) == thrust::get<1>(p);
    }));
  }

  SECTION("All inserted keys-value pairs should be contained")
  {
    map.insert(pairs_begin, pairs_begin + num_keys);
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());

    REQUIRE(cuco::test::all_of(
      d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b; }));
  }

  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());

    REQUIRE(cuco::test::none_of(
      d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b; }));
  }

  SECTION("Inserting unique keys should return insert success.")
  {
    REQUIRE(
      cuco::test::all_of(pairs_begin,
                         pairs_begin + num_keys,
                         [m_view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
                           return m_view.insert(pair);
                         }));
  }

  SECTION("Cannot find any key in an empty hash map with non-const view")
  {
    SECTION("non-const view")
    {
      REQUIRE(
        cuco::test::all_of(pairs_begin,
                           pairs_begin + num_keys,
                           [view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
                             return view.find(pair.first) == view.end();
                           }));
    }
    SECTION("const view")
    {
      REQUIRE(cuco::test::all_of(pairs_begin,
                                 pairs_begin + num_keys,
                                 [view] __device__(cuco::pair_type<Key, Value> const& pair) {
                                   return view.find(pair.first) == view.end();
                                 }));
    }
  }

  SECTION("Keys are all found after inserting many keys.")
  {
    // Bulk insert keys
    thrust::for_each(thrust::device,
                     pairs_begin,
                     pairs_begin + num_keys,
                     [m_view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
                       m_view.insert(pair);
                     });

    SECTION("non-const view")
    {
      // All keys should be found
      REQUIRE(cuco::test::all_of(
        pairs_begin,
        pairs_begin + num_keys,
        [view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
          auto const found = view.find(pair.first);
          return (found != view.end()) and
                 (found->first.load() == pair.first and found->second.load() == pair.second);
        }));
    }
    SECTION("const view")
    {
      // All keys should be found
      REQUIRE(cuco::test::all_of(pairs_begin,
                                 pairs_begin + num_keys,
                                 [view] __device__(cuco::pair_type<Key, Value> const& pair) {
                                   auto const found = view.find(pair.first);
                                   return (found != view.end()) and
                                          (found->first.load() == pair.first and
                                           found->second.load() == pair.second);
                                 }));
    }
  }
}
