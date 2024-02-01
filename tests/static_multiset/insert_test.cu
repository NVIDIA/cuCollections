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

#include <utils.hpp>

#include <cuco/static_multiset.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <catch2/catch_template_test_macros.hpp>

using size_type = int32_t;

template <typename Set>
__inline__ void test_insert(Set& set)
{
  using Key = typename Set::key_type;

  auto constexpr num          = 300;
  auto constexpr multiplicity = 3;

  SECTION("Inserting 300 unique keys should get 300 entries in the multiset")
  {
    auto const keys           = thrust::counting_iterator<Key>{0};
    auto const num_insertions = set.insert(keys, keys + num);

    REQUIRE(num_insertions == num);
  }

  SECTION("Inserting one key for 300 times should get 300 entries in the multiset")
  {
    auto const keys           = thrust::constant_iterator<Key>{0};
    auto const num_insertions = set.insert(keys, keys + num);

    REQUIRE(num_insertions == num);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_multiset insert tests",
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
  constexpr size_type num_keys{400};
  constexpr size_type gold_capacity = CGSize == 1 ? 422   // 211 x 1 x 2
                                                  : 412;  // 103 x 2 x 2

  using probe = std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                                   cuco::linear_probing<CGSize, cuco::default_hash_function<Key>>,
                                   cuco::double_hashing<CGSize, cuco::default_hash_function<Key>>>;

  auto set =
    cuco::static_multiset{num_keys, cuco::empty_key<Key>{-1}, {}, probe{}, {}, cuco::storage<2>{}};

  REQUIRE(set.capacity() == gold_capacity);

  test_insert(set);
}
