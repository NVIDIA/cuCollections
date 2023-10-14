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

#include <utils.hpp>

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cstdint>

#include <iostream>

#include <catch2/catch_template_test_macros.hpp>

using size_type = std::size_t;

template <typename Map>
__inline__ void test_insert_or_apply(Map& map, size_type num_keys, size_type num_unique_keys)
{
  REQUIRE((num_keys % num_unique_keys) == 0);

  using key_type    = typename Map::key_type;
  using mapped_type = typename Map::mapped_type;

  auto keys_begin = thrust::make_transform_iterator(
    thrust::counting_iterator<key_type>(0),
    [num_unique_keys] __host__ __device__(key_type const& x) -> key_type {
      return x % num_unique_keys;
    });

  auto values_begin = thrust::make_constant_iterator<mapped_type>(1);

  auto pairs_begin = thrust::make_zip_iterator(thrust::make_tuple(keys_begin, values_begin));

  map.insert_or_apply(pairs_begin, pairs_begin + num_keys, cuco::experimental::op::reduce::sum);

  REQUIRE(map.size() == num_unique_keys);

  thrust::device_vector<key_type> d_keys(num_unique_keys);
  thrust::device_vector<mapped_type> d_values(num_unique_keys);
  map.retrieve_all(d_keys.begin(), d_values.begin());

  // TODO remove
  for (int i = 0; i < num_unique_keys; ++i) {
    std::cout << d_keys[i] << " " << d_values[i] << std::endl;
  }

  REQUIRE(cuco::test::equal(d_values.begin(),
                            d_values.end(),
                            thrust::make_constant_iterator<mapped_type>(num_keys / num_unique_keys),
                            thrust::equal_to<mapped_type>{}));
}

TEMPLATE_TEST_CASE_SIG(
  "Insert or apply",
  "",
  ((typename Key, typename Value, cuco::test::probe_sequence Probe, int CGSize),
   Key,
   Value,
   Probe,
   CGSize),
  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing, 1))
{
  constexpr size_type num_keys{10};
  constexpr size_type num_unique_keys{10};

  using probe =
    std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                       cuco::experimental::linear_probing<CGSize, cuco::murmurhash3_32<Key>>,
                       cuco::experimental::double_hashing<CGSize,
                                                          cuco::murmurhash3_32<Key>,
                                                          cuco::murmurhash3_32<Key>>>;

  auto map = cuco::experimental::static_map<Key,
                                            Value,
                                            cuco::experimental::extent<size_type>,
                                            cuda::thread_scope_device,
                                            thrust::equal_to<Key>,
                                            probe,
                                            cuco::cuda_allocator<std::byte>,
                                            cuco::experimental::storage<2>>{
    num_keys, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};

  test_insert_or_apply(map, num_keys, num_unique_keys);
}