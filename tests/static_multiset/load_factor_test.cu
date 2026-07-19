/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/__config>
#include <cuco/static_multiset.cuh>

#include <catch2/catch_template_test_macros.hpp>

using size_type = int32_t;

TEMPLATE_TEST_CASE_SIG(
  "static_multiset load factor tests",
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
  constexpr size_type num_keys{10};

  using probe = std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                                   cuco::linear_probing<CGSize, cuco::default_hash_function<Key>>,
                                   cuco::double_hashing<CGSize, cuco::default_hash_function<Key>>>;

  SECTION("Negative load factor will throw exception")
  {
    REQUIRE_THROWS(cuco::static_multiset{
      num_keys, -0.1, cuco::empty_key<Key>{-1}, {}, probe{}, {}, cuco::storage<2>{}});
  }

  SECTION("Zero load factor will throw exception")
  {
    REQUIRE_THROWS(cuco::static_multiset{
      num_keys, 0.0, cuco::empty_key<Key>{-1}, {}, probe{}, {}, cuco::storage<2>{}});
  }

  SECTION("Load factor larger than one will throw exception")
  {
    REQUIRE_THROWS(cuco::static_multiset{
      num_keys, 1.1, cuco::empty_key<Key>{-1}, {}, probe{}, {}, cuco::storage<2>{}});
  }
}
