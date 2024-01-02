/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cuco/static_multimap.cuh>

#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <catch2/catch_template_test_macros.hpp>

template <typename Map>
__inline__ void test_multiplicity_two(Map& map, std::size_t num_items)
{
  using Key   = typename Map::key_type;
  using Value = typename Map::mapped_type;

  thrust::device_vector<Key> d_keys(num_items / 2);
  thrust::device_vector<cuco::pair<Key, Value>> d_pairs(num_items);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  // multiplicity = 2
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_items),
                    d_pairs.begin(),
                    [] __device__(auto i) {
                      return cuco::pair<Key, Value>{i / 2, i};
                    });

  thrust::device_vector<cuco::pair<Key, Value>> d_results(num_items);

  auto key_begin    = d_keys.begin();
  auto pair_begin   = d_pairs.begin();
  auto result_begin = d_results.begin();
  auto num_keys     = num_items / 2;
  thrust::device_vector<bool> d_contained(num_keys);

  SECTION("Non-inserted key/value pairs should not be contained.")
  {
    auto size = map.get_size();
    REQUIRE(size == 0);

    map.contains(key_begin, key_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::none_of(d_contained.begin(), d_contained.end(), thrust::identity{}));
  }

  map.insert(pair_begin, pair_begin + num_items);

  SECTION("All inserted key/value pairs should be contained.")
  {
    auto size = map.get_size();
    REQUIRE(size == num_items);

    map.contains(key_begin, key_begin + num_keys, d_contained.begin());

    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), thrust::identity{}));
  }

  SECTION("Total count should be equal to the number of inserted pairs.")
  {
    // Count matching keys
    auto num = map.count(key_begin, key_begin + num_keys);

    REQUIRE(num == num_items);

    auto output_begin      = result_begin;
    auto output_end        = map.retrieve(key_begin, key_begin + num_keys, output_begin);
    std::size_t const size = thrust::distance(output_begin, output_end);

    REQUIRE(size == num_items);

    // sort before compare
    thrust::sort(
      thrust::device,
      d_results.begin(),
      d_results.end(),
      [] __device__(const cuco::pair<Key, Value>& lhs, const cuco::pair<Key, Value>& rhs) {
        if (lhs.first != rhs.first) { return lhs.first < rhs.first; }
        return lhs.second < rhs.second;
      });

    REQUIRE(
      cuco::test::equal(pair_begin,
                        pair_begin + num_items,
                        output_begin,
                        [] __device__(cuco::pair<Key, Value> lhs, cuco::pair<Key, Value> rhs) {
                          return lhs.first == rhs.first and lhs.second == rhs.second;
                        }));
  }

  SECTION("count and count_outer should return the same value.")
  {
    auto num       = map.count(key_begin, key_begin + num_keys);
    auto num_outer = map.count_outer(key_begin, key_begin + num_keys);

    REQUIRE(num == num_outer);
  }

  SECTION("Output of retrieve and retrieve_outer should be the same.")
  {
    auto output_begin = result_begin;
    auto output_end   = map.retrieve(key_begin, key_begin + num_keys, output_begin);
    auto size         = thrust::distance(output_begin, output_end);

    output_end      = map.retrieve_outer(key_begin, key_begin + num_keys, output_begin);
    auto size_outer = thrust::distance(output_begin, output_end);

    REQUIRE(size == size_outer);

    // sort before compare
    thrust::sort(
      thrust::device,
      d_results.begin(),
      d_results.end(),
      [] __device__(const cuco::pair<Key, Value>& lhs, const cuco::pair<Key, Value>& rhs) {
        if (lhs.first != rhs.first) { return lhs.first < rhs.first; }
        return lhs.second < rhs.second;
      });

    REQUIRE(
      cuco::test::equal(pair_begin,
                        pair_begin + num_items,
                        output_begin,
                        [] __device__(cuco::pair<Key, Value> lhs, cuco::pair<Key, Value> rhs) {
                          return lhs.first == rhs.first and lhs.second == rhs.second;
                        }));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Multiplicity equals two",
  "",
  ((typename Key, typename Value, cuco::test::probe_sequence Probe), Key, Value, Probe),
  (int32_t, int32_t, cuco::test::probe_sequence::linear_probing),
  (int32_t, int64_t, cuco::test::probe_sequence::linear_probing),
  (int64_t, int64_t, cuco::test::probe_sequence::linear_probing),
  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing),
  (int32_t, int64_t, cuco::test::probe_sequence::double_hashing),
  (int64_t, int64_t, cuco::test::probe_sequence::double_hashing))
{
  constexpr std::size_t num_items{4};

  using probe =
    std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                       cuco::legacy::linear_probing<1, cuco::default_hash_function<Key>>,
                       cuco::legacy::double_hashing<8, cuco::default_hash_function<Key>>>;

  cuco::static_multimap<Key, Value, cuda::thread_scope_device, cuco::cuda_allocator<char>, probe>
    map{5, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};
  test_multiplicity_two(map, num_items);
}
