/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <test_utils.hpp>

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>
#include <iostream>

using size_type = std::size_t;

template <typename Map>
__inline__ void test_insert_or_apply(Map& map, size_type num_keys, size_type num_unique_keys)
{
  REQUIRE((num_keys % num_unique_keys) == 0);

  using Key   = typename Map::key_type;
  using Value = typename Map::mapped_type;

  // Insert pairs
  auto pairs_begin = thrust::make_transform_iterator(
    thrust::counting_iterator<size_type>(0),
    cuda::proclaim_return_type<cuco::pair<Key, Value>>([num_unique_keys] __device__(auto i) {
      return cuco::pair<Key, Value>{i % num_unique_keys, 1};
    }));

  map.insert_or_apply(pairs_begin, pairs_begin + num_keys, cuco::op::reduce::sum);

  REQUIRE(map.size() == num_unique_keys);

  thrust::device_vector<Key> d_keys(num_unique_keys);
  thrust::device_vector<Value> d_values(num_unique_keys);
  map.retrieve_all(d_keys.begin(), d_values.begin());

  REQUIRE(cuco::test::equal(d_values.begin(),
                            d_values.end(),
                            thrust::make_constant_iterator<Value>(num_keys / num_unique_keys),
                            thrust::equal_to<Value>{}));
}

TEMPLATE_TEST_CASE_SIG(
  "Insert or apply",
  "",
  ((typename Key, typename Value, cuco::test::probe_sequence Probe, int CGSize),
   Key,
   Value,
   Probe,
   CGSize),
  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int32_t, int64_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, int64_t, cuco::test::probe_sequence::linear_probing, 2))
{
  constexpr size_type num_keys{400};
  constexpr size_type num_unique_keys{100};

  using probe = std::conditional_t<
    Probe == cuco::test::probe_sequence::linear_probing,
    cuco::linear_probing<CGSize, cuco::murmurhash3_32<Key>>,
    cuco::double_hashing<CGSize, cuco::murmurhash3_32<Key>, cuco::murmurhash3_32<Key>>>;

  auto map = cuco::static_map<Key,
                              Value,
                              cuco::extent<size_type>,
                              cuda::thread_scope_device,
                              thrust::equal_to<Key>,
                              probe,
                              cuco::cuda_allocator<std::byte>,
                              cuco::storage<2>>{
    num_keys, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{0}};

  test_insert_or_apply(map, num_keys, num_unique_keys);
}