/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuco/static_multiset.cuh>

#include <cuda/functional>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <catch2/catch_template_test_macros.hpp>

using size_type = int32_t;

template <typename Set>
void test_unique_sequence(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  thrust::device_vector<Key> d_keys(num_keys);
  auto const keys_begin = d_keys.begin();

  SECTION("Count of empty set should be zero.")
  {
    auto const count = set.count(keys_begin, keys_begin + num_keys);
    REQUIRE(count == 0);
  }

  thrust::sequence(keys_begin, keys_begin + num_keys);
  set.insert(keys_begin, keys_begin + num_keys);

  SECTION("Count of n unique keys should be n.")
  {
    auto const count = set.count(keys_begin, keys_begin + num_keys);
    REQUIRE(count == num_keys);
  }

  auto constexpr multiplicity = 3;
  auto query_begin            = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0),
    cuda::proclaim_return_type<Key>([] __device__(auto i) { return Key{i / multiplicity}; }));

  SECTION("Count of 3n unique keys should be 3n.")
  {
    auto const count = set.count(query_begin, query_begin + num_keys * multiplicity);
    REQUIRE(count == num_keys * multiplicity);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_multiset count tests",
  "",
  ((typename Key, cuco::test::probe_sequence Probe, int CGSize), Key, Probe, CGSize),
  (int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, cuco::test::probe_sequence::linear_probing, 2))
{
  constexpr size_type num_keys{555};

  using probe = std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                                   cuco::linear_probing<CGSize, cuco::default_hash_function<Key>>,
                                   cuco::double_hashing<CGSize, cuco::default_hash_function<Key>>>;

  auto set =
    cuco::static_multiset{num_keys, cuco::empty_key<Key>{-1}, {}, probe{}, {}, cuco::storage<2>{}};

  test_unique_sequence(set, num_keys);
}
