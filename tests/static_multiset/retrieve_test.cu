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
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <limits>

template <class Container>
void test_multiplicity(Container& container, std::size_t num_keys, std::size_t multiplicity)
{
  using key_type = typename Container::key_type;

  container.clear();

  auto const num_unique_keys = num_keys / multiplicity;
  REQUIRE(num_unique_keys > 0);
  auto const num_actual_keys = num_unique_keys * multiplicity;
  REQUIRE(num_actual_keys <= num_keys);

  thrust::device_vector<key_type> probed_keys(num_actual_keys);
  thrust::device_vector<key_type> matched_keys(num_actual_keys);

  auto const keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<key_type>(0),
    cuda::proclaim_return_type<key_type>([multiplicity] __device__(auto const& i) {
      return static_cast<key_type>(i / multiplicity);
    }));

  container.insert(keys_begin, keys_begin + num_actual_keys);
  REQUIRE(container.size() == num_actual_keys);

  SECTION("All inserted keys should be contained.")
  {
    auto const query_begin = cuda::counting_iterator<key_type>{0};
    REQUIRE(container.count(query_begin, query_begin + num_unique_keys) == num_actual_keys);
    auto const [probed_end, matched_end] = container.retrieve(
      query_begin, query_begin + num_unique_keys, probed_keys.begin(), matched_keys.begin());
    REQUIRE(probed_end == probed_keys.end());
    REQUIRE(matched_end == matched_keys.end());
    thrust::sort(probed_keys.begin(), probed_end);
    thrust::sort(matched_keys.begin(), matched_end);
    REQUIRE(cuco::test::equal(
      probed_keys.begin(), probed_keys.end(), keys_begin, cuda::std::equal_to<key_type>{}));
    REQUIRE(cuco::test::equal(
      matched_keys.begin(), matched_keys.end(), keys_begin, cuda::std::equal_to<key_type>{}));
  }
}

template <class Container>
void test_missing_keys(Container& container, std::size_t num_keys)
{
  using key_type = typename Container::key_type;
  container.clear();
  auto const keys_begin = cuda::counting_iterator<key_type>{0};
  auto const query_size = num_keys * 2;
  thrust::device_vector<key_type> probed_keys(num_keys);
  thrust::device_vector<key_type> matched_keys(num_keys);

  SECTION("Empty input and missing keys should produce no matches.")
  {
    for (auto const n : {std::size_t{0}, query_size}) {
      REQUIRE(container.count(keys_begin, keys_begin + n) == 0);
      auto const [probed_end, matched_end] = container.retrieve(keys_begin,
                                                                keys_begin + n,
                                                                container.key_eq(),
                                                                container.hash_function(),
                                                                probed_keys.begin(),
                                                                matched_keys.begin());
      REQUIRE(probed_end == probed_keys.begin());
      REQUIRE(matched_end == matched_keys.begin());
    }
  }

  container.insert(keys_begin, keys_begin + num_keys);

  SECTION("Mixed queries should retrieve only matching keys.")
  {
    REQUIRE(container.count(keys_begin, keys_begin + query_size) == num_keys);
    auto const [probed_end, matched_end] = container.retrieve(keys_begin,
                                                              keys_begin + query_size,
                                                              container.key_eq(),
                                                              container.hash_function(),
                                                              probed_keys.begin(),
                                                              matched_keys.begin());
    REQUIRE(probed_end == probed_keys.end());
    REQUIRE(matched_end == matched_keys.end());
    thrust::sort(probed_keys.begin(), probed_end);
    thrust::sort(matched_keys.begin(), matched_end);
    REQUIRE(cuco::test::equal(
      probed_keys.begin(), probed_end, keys_begin, cuda::std::equal_to<key_type>{}));
    REQUIRE(cuco::test::equal(
      matched_keys.begin(), matched_end, keys_begin, cuda::std::equal_to<key_type>{}));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_multiset retrieve tests",
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
  constexpr std::size_t num_keys{400};
  constexpr double desired_load_factor = 0.5;
  constexpr auto empty_key_sentinel    = std::numeric_limits<Key>::max();

  using probe = std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                                   cuco::linear_probing<CGSize, cuco::default_hash_function<Key>>,
                                   cuco::double_hashing<CGSize, cuco::default_hash_function<Key>>>;

  auto set = cuco::static_multiset{
    num_keys, desired_load_factor, cuco::empty_key<Key>{empty_key_sentinel}, {}, probe{}};

  auto const multiplicity = GENERATE(std::size_t{1}, std::size_t{2}, std::size_t{11});
  test_multiplicity(set, num_keys, multiplicity);
  test_missing_keys(set, num_keys);
}
