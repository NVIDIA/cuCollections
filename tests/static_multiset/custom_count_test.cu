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
}
