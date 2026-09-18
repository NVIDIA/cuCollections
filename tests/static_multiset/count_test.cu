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

#include <catch2/catch_template_test_macros.hpp>

using size_type = int32_t;

template <typename Set>
void test_unique_sequence(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  auto keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>{0},
    cuda::proclaim_return_type<Key>([] __device__(auto i) { return Key{i}; }));

  SECTION("Count of empty set should be zero.")
  {
    auto const count = set.count(keys_begin, keys_begin + num_keys);
    REQUIRE(count == 0);
  }

  set.insert(keys_begin, keys_begin + num_keys);

  SECTION("Count of n unique keys should be n.")
  {
    auto const count = set.count(keys_begin, keys_begin + num_keys);
    REQUIRE(count == num_keys);
  }

  auto constexpr multiplicity = 3;
  auto query_begin            = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>{0},
    cuda::proclaim_return_type<Key>([] __device__(auto i) { return Key{i / multiplicity}; }));

  SECTION("Count of 3n unique keys should be 3n.")
  {
    auto const count = set.count(query_begin, query_begin + num_keys * multiplicity);
    REQUIRE(count == num_keys * multiplicity);
  }
}

template <typename Set>
void test_count_each(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  thrust::device_vector<size_type> d_counts(num_keys);
  auto const counts_begin = d_counts.begin();

  auto keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>{0},
    cuda::proclaim_return_type<Key>([] __device__(auto i) { return Key{i}; }));

  set.clear();

  SECTION("Count_each of empty set should be all zeros.")
  {
    set.count_each(
      keys_begin, keys_begin + num_keys, set.key_eq(), set.hash_function(), counts_begin);
    REQUIRE(cuco::test::all_of(
      d_counts.begin(),
      d_counts.end(),
      cuda::proclaim_return_type<bool>([] __device__(size_type count) { return count == 0; })));
  }

  set.insert(keys_begin, keys_begin + num_keys);

  SECTION("Count_each of n unique keys should be all ones.")
  {
    set.count_each(
      keys_begin, keys_begin + num_keys, set.key_eq(), set.hash_function(), counts_begin);
    REQUIRE(cuco::test::all_of(
      d_counts.begin(),
      d_counts.end(),
      cuda::proclaim_return_type<bool>([] __device__(size_type count) { return count == 1; })));
  }

  set.clear();

  auto constexpr multiplicity = 3;
  auto duplicate_keys_begin   = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>{0},
    cuda::proclaim_return_type<Key>([] __device__(auto i) { return Key{i / multiplicity}; }));
  set.insert(duplicate_keys_begin, duplicate_keys_begin + num_keys);

  auto const query_begin = cuda::counting_iterator<size_type>{0};
  auto const query_size  = num_keys / multiplicity;
  SECTION("Count_each with duplicates should return correct counts.")
  {
    set.count_each(
      query_begin, query_begin + query_size, set.key_eq(), set.hash_function(), counts_begin);
    REQUIRE(cuco::test::all_of(d_counts.begin(),
                               d_counts.begin() + query_size,
                               cuda::proclaim_return_type<bool>([] __device__(size_type count) {
                                 return count == multiplicity;
                               })));
  }
}

template <typename Set>
void test_count_each_outer(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  thrust::device_vector<size_type> d_counts(num_keys);
  auto const counts_begin = d_counts.begin();

  auto keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>{0},
    cuda::proclaim_return_type<Key>([] __device__(auto i) { return Key{i}; }));

  set.clear();

  SECTION("Count_each_outer of empty set should be all ones.")
  {
    set.count_each_outer(
      keys_begin, keys_begin + num_keys, set.key_eq(), set.hash_function(), counts_begin);
    REQUIRE(cuco::test::all_of(
      d_counts.begin(),
      d_counts.end(),
      cuda::proclaim_return_type<bool>([] __device__(size_type count) { return count == 1; })));
  }

  set.insert(keys_begin, keys_begin + num_keys);

  SECTION("Count_each_outer of n unique keys should be all ones.")
  {
    set.count_each_outer(
      keys_begin, keys_begin + num_keys, set.key_eq(), set.hash_function(), counts_begin);
    REQUIRE(cuco::test::all_of(
      d_counts.begin(),
      d_counts.end(),
      cuda::proclaim_return_type<bool>([] __device__(size_type count) { return count == 1; })));
  }

  set.clear();

  auto constexpr multiplicity = 3;
  auto duplicate_keys_begin   = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>{0},
    cuda::proclaim_return_type<Key>([] __device__(auto i) { return Key{i / multiplicity}; }));
  set.insert(duplicate_keys_begin, duplicate_keys_begin + num_keys);

  auto const query_size  = num_keys / multiplicity;
  auto const query_begin = cuda::counting_iterator<size_type>{0};

  SECTION("Count_each_outer with duplicates should return correct counts.")
  {
    set.count_each_outer(
      query_begin, query_begin + query_size, set.key_eq(), set.hash_function(), counts_begin);
    REQUIRE(cuco::test::all_of(d_counts.begin(),
                               d_counts.begin() + query_size,
                               cuda::proclaim_return_type<bool>([] __device__(size_type count) {
                                 return count == multiplicity;
                               })));
  }
}

template <typename Set>
void test_count_if(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  auto keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>{0},
    cuda::proclaim_return_type<Key>([] __device__(auto i) { return Key{i}; }));

  auto stencil_begin = cuda::counting_iterator<size_type>{0};

  set.clear();

  set.insert(keys_begin, keys_begin + num_keys);

  SECTION("Count_if with all elements selected should match count.")
  {
    auto const count =
      set.count_if(keys_begin,
                   keys_begin + num_keys,
                   stencil_begin,
                   cuda::proclaim_return_type<bool>([] __device__(size_type) { return true; }));

    REQUIRE(count == num_keys);
  }

  SECTION("Count_if with no elements selected should return zero.")
  {
    auto const count =
      set.count_if(keys_begin,
                   keys_begin + num_keys,
                   stencil_begin,
                   cuda::proclaim_return_type<bool>([] __device__(size_type) { return false; }));

    REQUIRE(count == 0);
  }

  SECTION("Count_if with alternating predicate should count selected keys.")
  {
    auto const count = set.count_if(
      keys_begin,
      keys_begin + num_keys,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type i) { return (i % 2) == 0; }));

    REQUIRE(count == (num_keys + 1) / 2);
  }
}

template <typename Set>
void test_count_outer_if(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  auto keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>{0},
    cuda::proclaim_return_type<Key>([] __device__(auto i) { return Key{i}; }));

  auto stencil_begin = cuda::counting_iterator<size_type>{0};

  set.clear();

  set.insert(keys_begin, keys_begin + num_keys);
  SECTION("Count_outer_if with all elements selected should match count_outer.")
  {
    auto const count = set.count_outer_if(
      keys_begin,
      keys_begin + num_keys,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type) { return true; }));

    REQUIRE(count == num_keys);
  }

  SECTION("Count_outer_if with no elements selected should return one per input.")
  {
    auto const count = set.count_outer_if(
      keys_begin,
      keys_begin + num_keys,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type) { return false; }));

    REQUIRE(count == num_keys);
  }

  SECTION(
    "Count_outer_if with alternating predicate should count selected matches and unselected rows.")
  {
    auto const count = set.count_outer_if(
      keys_begin,
      keys_begin + num_keys,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type i) { return (i % 2) == 0; }));

    REQUIRE(count == num_keys);
  }
}

template <typename Set>
void test_count_if_stencil(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  auto keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>{0},
    cuda::proclaim_return_type<Key>([] __device__(auto i) { return Key{i}; }));

  auto stencil_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>{0},
    cuda::proclaim_return_type<size_type>([] __device__(auto i) { return i + 1000; }));

  set.clear();
  set.insert(keys_begin, keys_begin + num_keys);

  SECTION("Count_if should apply the predicate to the stencil.")
  {
    auto const count = set.count_if(
      keys_begin,
      keys_begin + num_keys,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type value) { return value < 1100; }));

    REQUIRE(count == 100);
  }

  SECTION("Count_outer_if should apply the predicate to the stencil.")
  {
    auto const count = set.count_outer_if(
      keys_begin,
      keys_begin + num_keys,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type value) { return value < 1100; }));

    REQUIRE(count == num_keys);
  }
}

template <typename Set>
void test_count_if_overloads(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  auto keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>{0},
    cuda::proclaim_return_type<Key>([] __device__(auto i) { return Key{i}; }));

  auto stencil_begin = cuda::counting_iterator<size_type>{0};

  set.clear();
  set.insert(keys_begin, keys_begin + num_keys);

  auto const pred =
    cuda::proclaim_return_type<bool>([] __device__(size_type i) { return (i % 2) == 0; });

  SECTION("Count_if explicit default key equality/hash matches overload.")
  {
    auto const count_default = set.count_if(keys_begin, keys_begin + num_keys, stencil_begin, pred);

    auto const count_explicit = set.count_if(
      keys_begin, keys_begin + num_keys, stencil_begin, pred, set.key_eq(), set.hash_function());

    REQUIRE(count_explicit == count_default);
  }

  SECTION("Count_outer_if explicit default key equality/hash matches overload.")
  {
    auto const count_default =
      set.count_outer_if(keys_begin, keys_begin + num_keys, stencil_begin, pred);

    auto const count_explicit = set.count_outer_if(
      keys_begin, keys_begin + num_keys, stencil_begin, pred, set.key_eq(), set.hash_function());

    REQUIRE(count_explicit == count_default);
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
  constexpr size_type num_keys{666};

  using probe = std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                                   cuco::linear_probing<CGSize, cuco::default_hash_function<Key>>,
                                   cuco::double_hashing<CGSize, cuco::default_hash_function<Key>>>;

  auto set =
    cuco::static_multiset{num_keys, cuco::empty_key<Key>{-1}, {}, probe{}, {}, cuco::storage<2>{}};

  test_unique_sequence(set, num_keys);
  test_count_each(set, num_keys);
  test_count_each_outer(set, num_keys);
  test_count_if(set, num_keys);
  test_count_outer_if(set, num_keys);
  test_count_if_stencil(set, num_keys);
  test_count_if_overloads(set, num_keys);
}
