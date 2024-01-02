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
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <catch2/catch_template_test_macros.hpp>

template <typename Key, typename Map, typename PairIt, typename KeyIt>
__inline__ void test_insert_if(Map& map, PairIt pair_begin, KeyIt key_begin, std::size_t size)
{
  // 50% insertion
  auto pred_lambda = [] __device__(Key k) { return k % 2 == 0; };

  map.insert_if(pair_begin, pair_begin + size, key_begin, pred_lambda);

  auto res = map.get_size();
  REQUIRE(res * 2 == size);

  auto num = map.count(key_begin, key_begin + size);
  REQUIRE(num * 2 == size);
}

TEMPLATE_TEST_CASE_SIG(
  "Tests of insert_if",
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
  // multiplicity = 1
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_keys),
                    d_pairs.begin(),
                    [] __device__(auto i) {
                      return cuco::pair<Key, Value>{i, i};
                    });

  using probe =
    std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                       cuco::legacy::linear_probing<1, cuco::default_hash_function<Key>>,
                       cuco::legacy::double_hashing<8, cuco::default_hash_function<Key>>>;

  cuco::static_multimap<Key, Value, cuda::thread_scope_device, cuco::cuda_allocator<char>, probe>
    map{num_keys * 2, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};
  test_insert_if<Key>(map, d_pairs.begin(), d_keys.begin(), num_keys);
}
