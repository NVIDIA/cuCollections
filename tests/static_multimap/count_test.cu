/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/__config>
#include <cuco/static_multimap.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <catch2/catch_template_test_macros.hpp>

using size_type = int32_t;

static size_type constexpr multiplicity = 3;

template <typename Map>
void test_multiplicity_count(Map& map, size_type num_keys)
{
  using Key   = typename Map::key_type;
  using Value = typename Map::mapped_type;

  auto const keys_begin = cuda::counting_iterator<Key>{0};

  SECTION("Count of empty map should be zero.")
  {
    auto const count = map.count(keys_begin, keys_begin + num_keys);
    REQUIRE(count == 0);
  }

  SECTION("Count of n unique keys should be n.")
  {
    auto const pairs_begin = cuda::make_transform_iterator(
      cuda::make_counting_iterator<size_type>(0),
      cuda::proclaim_return_type<cuco::pair<Key, Value>>(
        [] __device__(auto i) { return cuco::pair<Key, Value>{i, i}; }));
    map.insert(pairs_begin, pairs_begin + num_keys);

    auto const count = map.count(keys_begin, keys_begin + num_keys);
    REQUIRE(count == num_keys);
  }

  SECTION("Count of n unique keys should be n x multiplicity.")
  {
    auto const pairs_begin = cuda::make_transform_iterator(
      cuda::make_counting_iterator<size_type>(0),
      cuda::proclaim_return_type<cuco::pair<Key, Value>>(
        [] __device__(auto i) { return cuco::pair<Key, Value>{i / multiplicity, i}; }));
    map.insert(pairs_begin, pairs_begin + num_keys * multiplicity);

    auto const count = map.count(keys_begin, keys_begin + num_keys);
    REQUIRE(count == num_keys * multiplicity);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_multimap count tests",
  "",
  ((typename T, cuco::test::probe_sequence Probe, int CGSize), T, Probe, CGSize),
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
  constexpr size_type num_keys{1'000};

  using probe = std::conditional_t<
    Probe == cuco::test::probe_sequence::linear_probing,
    cuco::linear_probing<CGSize, cuco::default_hash_function<T>>,
    cuco::double_hashing<CGSize, cuco::default_hash_function<T>, cuco::default_hash_function<T>>>;

  auto map = cuco::static_multimap<T,
                                   T,
                                   cuco::extent<size_type>,
                                   cuda::thread_scope_device,
                                   cuda::std::equal_to<T>,
                                   probe,
                                   cuco::cuda_allocator<cuda::std::byte>,
                                   cuco::storage<2>>{
    num_keys * multiplicity, cuco::empty_key<T>{-1}, cuco::empty_value<T>{-1}};

  test_multiplicity_count(map, num_keys);
}
