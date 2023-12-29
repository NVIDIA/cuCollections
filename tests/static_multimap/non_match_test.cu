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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <catch2/catch_template_test_macros.hpp>

template <typename Key, typename Value, typename Map, typename PairIt, typename KeyIt>
__inline__ void test_non_matches(Map& map, PairIt pair_begin, KeyIt key_begin, std::size_t num_keys)
{
  map.insert(pair_begin, pair_begin + num_keys);

  auto res = map.get_size();
  REQUIRE(res == num_keys);

  SECTION("Output of count and retrieve should be coherent.")
  {
    auto num = map.count(key_begin, key_begin + num_keys);
    thrust::device_vector<cuco::pair<Key, Value>> d_results(num);

    REQUIRE(num == num_keys);

    auto output_begin      = d_results.begin();
    auto output_end        = map.retrieve(key_begin, key_begin + num_keys, output_begin);
    std::size_t const size = thrust::distance(output_begin, output_end);

    REQUIRE(size == num_keys);

    // sort before compare
    thrust::sort(
      thrust::device,
      output_begin,
      output_end,
      [] __device__(const cuco::pair<Key, Value>& lhs, const cuco::pair<Key, Value>& rhs) {
        if (lhs.first != rhs.first) { return lhs.first < rhs.first; }
        return lhs.second < rhs.second;
      });

    REQUIRE(
      cuco::test::equal(pair_begin,
                        pair_begin + num_keys,
                        output_begin,
                        [] __device__(cuco::pair<Key, Value> lhs, cuco::pair<Key, Value> rhs) {
                          return lhs.first == rhs.first and lhs.second == rhs.second;
                        }));
  }

  SECTION("Output of count_outer and retrieve_outer should be coherent.")
  {
    auto num = map.count_outer(key_begin, key_begin + num_keys);
    thrust::device_vector<cuco::pair<Key, Value>> d_results(num);

    REQUIRE(num == (num_keys + num_keys / 2));

    auto output_begin      = d_results.begin();
    auto output_end        = map.retrieve_outer(key_begin, key_begin + num_keys, output_begin);
    std::size_t const size = thrust::distance(output_begin, output_end);

    REQUIRE(size == (num_keys + num_keys / 2));

    // sort before compare
    thrust::sort(
      thrust::device,
      output_begin,
      output_end,
      [] __device__(const cuco::pair<Key, Value>& lhs, const cuco::pair<Key, Value>& rhs) {
        if (lhs.first != rhs.first) { return lhs.first < rhs.first; }
        return lhs.second < rhs.second;
      });

    // create gold reference
    thrust::device_vector<cuco::pair<Key, Value>> gold(size);
    auto gold_begin = gold.begin();
    thrust::transform(thrust::device,
                      thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(size),
                      gold_begin,
                      [num_keys] __device__(auto i) {
                        if (i < num_keys) { return cuco::pair<Key, Value>{i / 2, i}; }
                        return cuco::pair<Key, Value>{i - num_keys / 2, -1};
                      });

    REQUIRE(
      cuco::test::equal(gold_begin,
                        gold_begin + size,
                        output_begin,
                        [] __device__(cuco::pair<Key, Value> lhs, cuco::pair<Key, Value> rhs) {
                          return lhs.first == rhs.first and lhs.second == rhs.second;
                        }));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Tests of non-matches",
  "",
  ((typename Key, typename Value, cuco::test::probe_sequence Probe), Key, Value, Probe),
  (int32_t, int32_t, cuco::test::probe_sequence::linear_probing),
  (int32_t, int64_t, cuco::test::probe_sequence::linear_probing),
  (int64_t, int64_t, cuco::test::probe_sequence::linear_probing),
  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing),
  (int32_t, int64_t, cuco::test::probe_sequence::double_hashing),
  (int64_t, int64_t, cuco::test::probe_sequence::double_hashing))
{
  constexpr std::size_t num_keys{1'000};

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<cuco::pair<Key, Value>> d_pairs(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  // multiplicity = 2
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_keys),
                    d_pairs.begin(),
                    [] __device__(auto i) {
                      return cuco::pair<Key, Value>{i / 2, i};
                    });

  using probe =
    std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                       cuco::legacy::linear_probing<1, cuco::default_hash_function<Key>>,
                       cuco::legacy::double_hashing<8, cuco::default_hash_function<Key>>>;

  cuco::static_multimap<Key,
                        Value,
                        cuda::thread_scope_device,
                        cuco::cuda_allocator<char>,
                        cuco::legacy::linear_probing<1, cuco::default_hash_function<Key>>>
    map{num_keys * 2, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};
  test_non_matches<Key, Value>(map, d_pairs.begin(), d_keys.begin(), num_keys);
}
