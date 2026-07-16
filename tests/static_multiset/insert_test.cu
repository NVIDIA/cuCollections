/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/__config>
#include <cuco/static_multiset.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <catch2/catch_template_test_macros.hpp>

using size_type = int32_t;

template <typename Set>
void test_insert(Set& set)
{
  using Key = typename Set::key_type;

  auto constexpr num = 300;

  SECTION("Inserting 300 unique keys should get 300 entries in the multiset")
  {
    auto const keys = cuda::counting_iterator<Key>{0};
    set.insert(keys, keys + num);
    auto const num_insertions = set.size();

    REQUIRE(num_insertions == num);
  }

  SECTION("Inserting one key for 300 times should get 300 entries in the multiset")
  {
    auto const keys = cuda::constant_iterator<Key>{0};
    set.insert(keys, keys + num);
    auto const num_insertions = set.size();

    REQUIRE(num_insertions == num);
  }

  auto const is_even =
    cuda::proclaim_return_type<bool>([] __device__(size_type const& i) { return i % 2 == 0; });

  SECTION("Inserting all even values between [0, 300) should get 150 entries in the multiset")
  {
    auto const keys = cuda::counting_iterator<Key>{0};
    set.insert_if(keys, keys + num, keys, is_even);
    auto const num_insertions = set.size();

    REQUIRE(num_insertions == num / 2);
  }

  SECTION("Conditionally inserting one key for 150 times should get 150 entries in the multiset")
  {
    auto const keys = cuda::constant_iterator<Key>{0};
    set.insert_if(keys, keys + num, cuda::counting_iterator<size_type>{0}, is_even);
    auto const num_insertions = set.size();

    REQUIRE(num_insertions == num / 2);
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
  (int64_t, cuco::test::probe_sequence::linear_probing, 2)
#if defined(CUCO_HAS_128BIT_ATOMICS)
    ,
  (__int128_t, cuco::test::probe_sequence::double_hashing, 1),
  (__int128_t, cuco::test::probe_sequence::double_hashing, 2),
  (__int128_t, cuco::test::probe_sequence::linear_probing, 1),
  (__int128_t, cuco::test::probe_sequence::linear_probing, 2)
#endif
)
{
  constexpr size_type num_keys{400};

  using probe = std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                                   cuco::linear_probing<CGSize, cuco::default_hash_function<Key>>,
                                   cuco::double_hashing<CGSize, cuco::default_hash_function<Key>>>;

  constexpr size_type gold_capacity = [&]() {
    if constexpr (cuco::is_double_hashing<probe>::value) {
      return (CGSize == 1) ? 422   // 211 x 1 x 2
                           : 404;  // 101 x 2 x 2
    } else {
      return 400;
    }
  }();

  auto set =
    cuco::static_multiset{num_keys, cuco::empty_key<Key>{-1}, {}, probe{}, {}, cuco::storage<2>{}};

  REQUIRE(set.capacity() == gold_capacity);

  test_insert(set);
}
