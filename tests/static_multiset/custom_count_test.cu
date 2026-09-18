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
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <catch2/catch_template_test_macros.hpp>

using size_type = std::size_t;

static auto constexpr XXX = 111;

template <typename T>
struct identity_hash {
  __host__ __device__ identity_hash() {};
  __host__ __device__ identity_hash([[maybe_unused]] int i) {}
  __device__ T operator()(T k) const { return k; }
};

struct custom_hash {
  __host__ __device__ custom_hash() {}
  __host__ __device__ custom_hash([[maybe_unused]] int i) {}
  template <typename custom_type>
  __device__ custom_type operator()(custom_type k) const
  {
    return k / XXX;
  };
};

struct custom_key_eq {
  template <typename lhs_type, typename rhs_type>
  __device__ bool operator()(lhs_type lhs, rhs_type rhs) const
  {
    return lhs / XXX == rhs;
  }
};

template <typename Set>
void test_custom_count(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  auto const hash = []() {
    if constexpr (cuco::is_double_hashing<typename Set::probing_scheme_type>::value) {
      return cuda::std::tuple{custom_hash{}, custom_hash{}};
    } else {
      return custom_hash{};
    }
  }();

  auto query_begin = cuda::make_transform_iterator(
    cuda::make_counting_iterator<size_type>(0),
    cuda::proclaim_return_type<Key>([] __device__(auto i) { return static_cast<Key>(i * XXX); }));

  SECTION("Count of empty set should be zero.")
  {
    auto const count = set.count(query_begin, query_begin + num_keys, custom_key_eq{}, hash);
    REQUIRE(count == 0);
  }

  SECTION("Outer count of empty set should be the same as input size.")
  {
    auto const count = set.count_outer(query_begin, query_begin + num_keys, custom_key_eq{}, hash);
    REQUIRE(count == num_keys);
  }

  auto const iter = cuda::counting_iterator<Key>{0};
  set.insert(iter, iter + num_keys);

  SECTION("Count of n unique keys should be n.")
  {
    auto const count = set.count(query_begin, query_begin + num_keys, custom_key_eq{}, hash);
    REQUIRE(count == num_keys);
  }

  SECTION("Outer count of n unique keys should be n.")
  {
    auto const count = set.count_outer(query_begin, query_begin + num_keys, custom_key_eq{}, hash);
    REQUIRE(count == num_keys);
  }

  set.clear();  // reset the set
  auto const constants = cuda::constant_iterator<Key>{1};
  set.insert(constants, constants + num_keys);  // inser the same value `num_keys` times

  SECTION("Count of a key whose multiplicity equals n should be n.")
  {
    auto const count = set.count(query_begin, query_begin + num_keys, custom_key_eq{}, hash);
    REQUIRE(count == num_keys);
  }

  SECTION("Outer count of a key whose multiplicity equals n should be n + input_size - 1.")
  {
    auto const count = set.count_outer(query_begin, query_begin + num_keys, custom_key_eq{}, hash);
    REQUIRE(count == 2 * num_keys - 1);
  }
}

template <typename Set>
void test_count_if_duplicates(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  auto constexpr multiplicity = 3;

  auto duplicate_keys_begin =
    cuda::make_transform_iterator(cuda::counting_iterator<size_type>{0},
                                  cuda::proclaim_return_type<Key>([] __device__(size_type i) {
                                    return static_cast<Key>(i / multiplicity);
                                  }));

  set.clear();
  set.insert(duplicate_keys_begin, duplicate_keys_begin + num_keys);

  auto query_begin      = cuda::counting_iterator<size_type>{0};
  auto const query_size = num_keys / multiplicity;

  auto stencil_begin = cuda::counting_iterator<size_type>{0};

  SECTION("Count_if with duplicates and all keys selected returns total multiplicity.")
  {
    auto const count =
      set.count_if(query_begin,
                   query_begin + query_size,
                   stencil_begin,
                   cuda::proclaim_return_type<bool>([] __device__(size_type) { return true; }));

    REQUIRE(count == query_size * multiplicity);
  }

  SECTION("Count_if with duplicates and no keys selected returns zero.")
  {
    auto const count =
      set.count_if(query_begin,
                   query_begin + query_size,
                   stencil_begin,
                   cuda::proclaim_return_type<bool>([] __device__(size_type) { return false; }));

    REQUIRE(count == 0);
  }

  SECTION("Count_if with duplicates counts only selected keys.")
  {
    auto const count = set.count_if(
      query_begin,
      query_begin + query_size,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type i) { return (i % 2) == 0; }));

    auto const expected = ((query_size + 1) / 2) * multiplicity;

    REQUIRE(count == expected);
  }

  SECTION("Count_if with duplicates counts a single selected key by its multiplicity.")
  {
    auto const count =
      set.count_if(query_begin,
                   query_begin + query_size,
                   stencil_begin,
                   cuda::proclaim_return_type<bool>([] __device__(size_type i) { return i == 0; }));

    REQUIRE(count == multiplicity);
  }
}

template <typename Set>
void test_count_outer_if_duplicates(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  auto constexpr multiplicity = 3;

  auto duplicate_keys_begin =
    cuda::make_transform_iterator(cuda::counting_iterator<size_type>{0},
                                  cuda::proclaim_return_type<Key>([] __device__(size_type i) {
                                    return static_cast<Key>(i / multiplicity);
                                  }));

  set.clear();
  set.insert(duplicate_keys_begin, duplicate_keys_begin + num_keys);

  // Query each unique key once.
  auto query_begin      = cuda::counting_iterator<size_type>{0};
  auto const query_size = num_keys / multiplicity;

  auto stencil_begin = cuda::counting_iterator<size_type>{0};

  SECTION("Count_outer_if with duplicates and all keys selected returns total multiplicity.")
  {
    auto const count = set.count_outer_if(
      query_begin,
      query_begin + query_size,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type) { return true; }));

    REQUIRE(count == query_size * multiplicity);
  }

  SECTION("Count_outer_if with duplicates and no keys selected returns one per query.")
  {
    auto const count = set.count_outer_if(
      query_begin,
      query_begin + query_size,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type) { return false; }));

    REQUIRE(count == query_size);
  }

  SECTION(
    "Count_outer_if with duplicates counts selected matches and one for each unselected query.")
  {
    auto const count = set.count_outer_if(
      query_begin,
      query_begin + query_size,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type i) { return (i % 2) == 0; }));

    auto const selected_count   = (query_size + 1) / 2;
    auto const unselected_count = query_size / 2;

    auto const expected = selected_count * multiplicity + unselected_count;

    REQUIRE(count == expected);
  }

  SECTION("Count_outer_if with a selected key counts its multiplicity.")
  {
    auto const count = set.count_outer_if(
      query_begin,
      query_begin + query_size,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type i) { return i == 0; }));

    auto const expected = multiplicity + (query_size - 1);

    REQUIRE(count == expected);
  }
}

template <typename Set>
void test_custom_count_if(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  auto const hash = []() {
    if constexpr (cuco::is_double_hashing<typename Set::probing_scheme_type>::value) {
      return cuda::std::tuple{custom_hash{}, custom_hash{}};
    } else {
      return custom_hash{};
    }
  }();

  constexpr auto multiplicity = 3;

  auto duplicate_keys_begin =
    cuda::make_transform_iterator(cuda::counting_iterator<size_type>{0},
                                  cuda::proclaim_return_type<Key>([] __device__(size_type i) {
                                    return static_cast<Key>(i / multiplicity);
                                  }));

  set.clear();
  set.insert(duplicate_keys_begin, duplicate_keys_begin + num_keys);

  auto query_begin      = cuda::counting_iterator<size_type>{0};
  auto const query_size = num_keys / multiplicity;
  auto stencil_begin    = cuda::counting_iterator<size_type>{0};

  SECTION("Count_if custom key equality/hash overload counts selected duplicates.")
  {
    auto const count = set.count_if(
      query_begin,
      query_begin + query_size,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type i) { return (i % 2) == 0; }),
      custom_key_eq{},
      hash);

    auto const selected_count = (query_size + 1) / 2;
    REQUIRE(count == selected_count * multiplicity);
  }

  SECTION("Count_outer_if custom key equality/hash overload counts selected duplicates.")
  {
    auto const count = set.count_outer_if(
      query_begin,
      query_begin + query_size,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type i) { return (i % 2) == 0; }),
      custom_key_eq{},
      hash);

    auto const selected_count   = (query_size + 1) / 2;
    auto const unselected_count = query_size / 2;

    REQUIRE(count == selected_count * multiplicity + unselected_count);
  }
}

template <typename Set>
void test_custom_count_if_overloads(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  auto const hash = []() {
    if constexpr (cuco::is_double_hashing<typename Set::probing_scheme_type>::value) {
      return cuda::std::tuple{custom_hash{}, custom_hash{}};
    } else {
      return custom_hash{};
    }
  }();

  constexpr auto multiplicity = 3;

  auto duplicate_keys_begin =
    cuda::make_transform_iterator(cuda::counting_iterator<size_type>{0},
                                  cuda::proclaim_return_type<Key>([] __device__(size_type i) {
                                    return static_cast<Key>(i / multiplicity);
                                  }));

  set.clear();
  set.insert(duplicate_keys_begin, duplicate_keys_begin + num_keys);

  auto query_begin   = cuda::counting_iterator<size_type>{0};
  auto query_size    = num_keys / multiplicity;
  auto stencil_begin = cuda::counting_iterator<size_type>{0};

  SECTION("Count_if explicit key equality/hash overload selects all duplicates.")
  {
    auto const count =
      set.count_if(query_begin,
                   query_begin + query_size,
                   stencil_begin,
                   cuda::proclaim_return_type<bool>([] __device__(size_type) { return true; }),
                   custom_key_eq{},
                   hash);

    REQUIRE(count == query_size * multiplicity);
  }

  SECTION("Count_if explicit key equality/hash overload selects no duplicates.")
  {
    auto const count =
      set.count_if(query_begin,
                   query_begin + query_size,
                   stencil_begin,
                   cuda::proclaim_return_type<bool>([] __device__(size_type) { return false; }),
                   custom_key_eq{},
                   hash);

    REQUIRE(count == 0);
  }

  SECTION("Count_if explicit key equality/hash overload selects alternating duplicates.")
  {
    auto const count = set.count_if(
      query_begin,
      query_begin + query_size,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type i) { return (i % 2) == 0; }),
      custom_key_eq{},
      hash);

    auto const selected_count = (query_size + 1) / 2;
    REQUIRE(count == selected_count * multiplicity);
  }

  SECTION("Count_outer_if explicit key equality/hash overload selects all duplicates.")
  {
    auto const count = set.count_outer_if(
      query_begin,
      query_begin + query_size,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type) { return true; }),
      custom_key_eq{},
      hash);

    REQUIRE(count == query_size * multiplicity);
  }

  SECTION("Count_outer_if explicit key equality/hash overload selects no duplicates.")
  {
    auto const count = set.count_outer_if(
      query_begin,
      query_begin + query_size,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type) { return false; }),
      custom_key_eq{},
      hash);

    REQUIRE(count == query_size);
  }

  SECTION("Count_outer_if explicit key equality/hash overload selects alternating duplicates.")
  {
    auto const count = set.count_outer_if(
      query_begin,
      query_begin + query_size,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type i) { return (i % 2) == 0; }),
      custom_key_eq{},
      hash);

    auto const selected_count   = (query_size + 1) / 2;
    auto const unselected_count = query_size / 2;

    REQUIRE(count == selected_count * multiplicity + unselected_count);
  }
}

template <typename Set>
void test_custom_hash_count_if(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  auto const hash = []() {
    if constexpr (cuco::is_double_hashing<typename Set::probing_scheme_type>::value) {
      return cuda::std::tuple{custom_hash{}, custom_hash{}};
    } else {
      return custom_hash{};
    }
  }();

  auto const iter = cuda::counting_iterator<Key>{0};
  set.clear();
  set.insert(iter, iter + num_keys);

  auto query_begin = cuda::make_transform_iterator(
    cuda::make_counting_iterator<size_type>(0),
    cuda::proclaim_return_type<Key>([] __device__(auto i) { return static_cast<Key>(i * XXX); }));

  auto stencil_begin = cuda::counting_iterator<size_type>{0};

  SECTION("Count_if uses custom key equality and hash.")
  {
    auto const count = set.count_if(
      query_begin,
      query_begin + num_keys,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type i) { return (i % 2) == 0; }),
      custom_key_eq{},
      hash);

    REQUIRE(count == (num_keys + 1) / 2);
  }

  SECTION("Count_outer_if uses custom key equality and hash.")
  {
    auto const count = set.count_outer_if(
      query_begin,
      query_begin + num_keys,
      stencil_begin,
      cuda::proclaim_return_type<bool>([] __device__(size_type i) { return (i % 2) == 0; }),
      custom_key_eq{},
      hash);

    REQUIRE(count == num_keys);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_multiset custom count tests",
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
  constexpr size_type num_keys{555};

  using probe = std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                                   cuco::linear_probing<CGSize, identity_hash<Key>>,
                                   cuco::double_hashing<CGSize, identity_hash<Key>>>;

  auto set =
    cuco::static_multiset{num_keys, cuco::empty_key<Key>{-1}, {}, probe{}, {}, cuco::storage<2>{}};

  test_custom_count(set, num_keys);
  test_count_if_duplicates(set, num_keys);
  test_count_outer_if_duplicates(set, num_keys);
  test_custom_count_if(set, num_keys);
  test_custom_count_if_overloads(set, num_keys);
  test_custom_hash_count_if(set, num_keys);
}
