/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuco/static_set.cuh>

#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <catch2/catch.hpp>

template <typename Set>
__inline__ void test_unique_sequence(Set& set, std::size_t num_keys)
{
  using Key = typename Set::key_type;

  thrust::device_vector<Key> d_keys(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());

  auto key_begin = d_keys.begin();
  thrust::device_vector<bool> d_contained(num_keys);

  SECTION("Non-inserted keys should not be contained.")
  {
    set.contains(key_begin, key_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::none_of(d_contained.begin(), d_contained.end(), thrust::identity{}));
  }

  set.insert(key_begin, key_begin + num_keys);

  SECTION("All inserted key/value pairs should be contained.")
  {
    set.contains(key_begin, key_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), thrust::identity{}));
  }
}

TEMPLATE_TEST_CASE_SIG("Unique sequence",
                       "",
                       ((typename Key, cuco::test::probe_sequence Probe), Key, Probe),
                       (int32_t, cuco::test::probe_sequence::double_hashing),
                       (int64_t, cuco::test::probe_sequence::double_hashing))
{
  constexpr std::size_t num_keys{400};

  cuco::experimental::static_set<Key> set{cuco::experimental::extent<std::size_t>{400},
                                          cuco::sentinel::empty_key<Key>{-1}};

  auto constexpr gold_capacity = 422;  // 211 x 2
  REQUIRE(set.capacity() == gold_capacity);

  test_unique_sequence(set, num_keys);
}
