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
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <catch2/catch_template_test_macros.hpp>

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
    auto const [probed_end, matched_end] = container.retrieve(
      keys_begin, keys_begin + num_actual_keys, probed_keys.begin(), matched_keys.begin());

    thrust::sort(probed_keys.begin(), probed_end);
    thrust::sort(matched_keys.begin(), matched_end);

    REQUIRE(cuco::test::equal(
      probed_keys.begin(), probed_keys.end(), keys_begin, cuda::std::equal_to<key_type>{}));

    REQUIRE(cuco::test::equal(
      matched_keys.begin(), matched_keys.end(), keys_begin, cuda::std::equal_to<key_type>{}));
  }
}

template <class Container>
void test_outer(Container& container, std::size_t num_keys)
{
  using key_type                = typename Container::key_type;
  auto const empty_key_sentinel = container.empty_key_sentinel();

  container.clear();

  auto const keys_begin = cuda::counting_iterator<key_type>{0};
  auto const query_size = num_keys * 2ull;

  thrust::device_vector<key_type> probed_keys(query_size);
  thrust::device_vector<key_type> matched_keys(query_size);

  SECTION("Non-inserted keys should output sentinels.")
  {
    auto const [probed_end, matched_end] = container.retrieve_outer(keys_begin,
                                                                    keys_begin + query_size,
                                                                    container.key_eq(),
                                                                    container.hash_function(),
                                                                    probed_keys.begin(),
                                                                    matched_keys.begin());

    REQUIRE(static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end)) == query_size);
    REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) ==
            query_size);

    REQUIRE(cuco::test::all_of(
      matched_keys.begin(),
      matched_keys.end(),
      cuda::proclaim_return_type<bool>([empty_key_sentinel] __device__(auto const& k) {
        return static_cast<bool>(k == static_cast<key_type>(empty_key_sentinel));
      })));
  }

  container.insert(keys_begin, keys_begin + num_keys);

  SECTION("All inserted keys should be contained.")
  {
    auto const [probed_end, matched_end] = container.retrieve_outer(keys_begin,
                                                                    keys_begin + query_size,
                                                                    container.key_eq(),
                                                                    container.hash_function(),
                                                                    probed_keys.begin(),
                                                                    matched_keys.begin());

    REQUIRE(static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end)) == query_size);
    REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) ==
            query_size);

    thrust::sort_by_key(
      probed_keys.begin(), probed_end, matched_keys.begin(), cuda::std::less<key_type>());

    REQUIRE(cuco::test::equal(
      probed_keys.begin(), probed_keys.end(), keys_begin, cuda::std::equal_to<key_type>{}));

    REQUIRE(cuco::test::equal(matched_keys.begin(),
                              matched_keys.begin() + num_keys,
                              keys_begin,
                              cuda::std::equal_to<key_type>{}));

    REQUIRE(cuco::test::all_of(
      matched_keys.begin() + num_keys,
      matched_keys.end(),
      cuda::proclaim_return_type<bool>([empty_key_sentinel] __device__(auto const& k) {
        return static_cast<bool>(k == static_cast<key_type>(empty_key_sentinel));
      })));
  }
}

template <class Container>
void test_retrieve_if(Container& container, std::size_t num_keys)
{
  using key_type = typename Container::key_type;

  container.clear();

  auto const keys_begin = cuda::counting_iterator<key_type>{0};

  container.insert(keys_begin, keys_begin + num_keys);

  thrust::device_vector<key_type> probed_keys(num_keys);
  thrust::device_vector<key_type> matched_keys(num_keys);
  thrust::device_vector<key_type> stencil(num_keys);

  SECTION("retrieve_if should predicate on the stencil, not the probe.")
  {
    thrust::sequence(stencil.begin(), stencil.end(), key_type{1});

    auto const pred = [] __device__(key_type key) { return key % 2 == 0; };

    auto const [probed_end, matched_end] = container.retrieve_if(keys_begin,
                                                                 keys_begin + num_keys,
                                                                 stencil.begin(),
                                                                 pred,
                                                                 probed_keys.begin(),
                                                                 matched_keys.begin());

    auto const num_results =
      static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end));

    auto const expected_size = num_keys / 2;

    REQUIRE(num_results == expected_size);
    REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) ==
            expected_size);

    thrust::sort_by_key(
      probed_keys.begin(), probed_end, matched_keys.begin(), cuda::std::less<key_type>());

    for (std::size_t i = 0; i < expected_size; ++i) {
      auto const expected = static_cast<key_type>(i * 2 + 1);

      REQUIRE(probed_keys[i] == expected);
      REQUIRE(matched_keys[i] == expected);
    }
  }

  SECTION("retrieve_if should retrieve only elements satisfying the predicate.")
  {
    thrust::sequence(stencil.begin(), stencil.end(), key_type{0});

    auto const pred = [] __device__(key_type key) { return key % 2 == 0; };

    auto const [probed_end, matched_end] = container.retrieve_if(keys_begin,
                                                                 keys_begin + num_keys,
                                                                 stencil.begin(),
                                                                 pred,
                                                                 probed_keys.begin(),
                                                                 matched_keys.begin());

    auto const num_results =
      static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end));

    REQUIRE(num_results == (num_keys + 1) / 2);
    REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) ==
            num_results);

    thrust::sort_by_key(
      probed_keys.begin(), probed_end, matched_keys.begin(), cuda::std::less<key_type>());

    for (std::size_t i = 0; i < num_results; ++i) {
      auto const expected = static_cast<key_type>(i * 2);

      REQUIRE(probed_keys[i] == expected);
      REQUIRE(matched_keys[i] == expected);
    }
  }

  SECTION("retrieve_if should return nothing when the predicate is always false.")
  {
    thrust::sequence(stencil.begin(), stencil.end(), key_type{0});

    auto const pred = [] __device__(key_type) { return false; };

    auto const [probed_end, matched_end] = container.retrieve_if(keys_begin,
                                                                 keys_begin + num_keys,
                                                                 stencil.begin(),
                                                                 pred,
                                                                 probed_keys.begin(),
                                                                 matched_keys.begin());

    REQUIRE(static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end)) == 0);
    REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) == 0);
  }

  SECTION("retrieve_if should retrieve everything when the predicate is always true.")
  {
    thrust::sequence(stencil.begin(), stencil.end(), key_type{0});

    auto const pred = [] __device__(key_type) { return true; };

    auto const [probed_end, matched_end] = container.retrieve_if(keys_begin,
                                                                 keys_begin + num_keys,
                                                                 stencil.begin(),
                                                                 pred,
                                                                 probed_keys.begin(),
                                                                 matched_keys.begin());

    REQUIRE(static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end)) == num_keys);
    REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) == num_keys);

    thrust::sort_by_key(
      probed_keys.begin(), probed_end, matched_keys.begin(), cuda::std::less<key_type>());

    REQUIRE(cuco::test::equal(
      probed_keys.begin(), probed_end, keys_begin, cuda::std::equal_to<key_type>{}));

    REQUIRE(cuco::test::equal(
      matched_keys.begin(), matched_end, keys_begin, cuda::std::equal_to<key_type>{}));
  }
}

template <class Container>
void test_retrieve_if_with_probe(Container& container, std::size_t num_keys)
{
  using key_type = typename Container::key_type;

  container.clear();

  auto const keys_begin = cuda::counting_iterator<key_type>{0};

  container.insert(keys_begin, keys_begin + num_keys);

  thrust::device_vector<key_type> probed_keys(num_keys);
  thrust::device_vector<key_type> matched_keys(num_keys);
  thrust::device_vector<key_type> stencil(num_keys);

  thrust::sequence(stencil.begin(), stencil.end(), key_type{0});

  SECTION("retrieve_if should accept explicit equality and hash functions.")
  {
    auto const pred = [] __device__(key_type key) { return key % 2 == 0; };

    auto const [probed_end, matched_end] = container.retrieve_if(keys_begin,
                                                                 keys_begin + num_keys,
                                                                 stencil.begin(),
                                                                 pred,
                                                                 container.key_eq(),
                                                                 container.hash_function(),
                                                                 probed_keys.begin(),
                                                                 matched_keys.begin());

    auto const num_results =
      static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end));

    REQUIRE(num_results == (num_keys + 1) / 2);
    REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) ==
            num_results);

    thrust::sort_by_key(
      probed_keys.begin(), probed_end, matched_keys.begin(), cuda::std::less<key_type>());

    for (std::size_t i = 0; i < num_results; ++i) {
      auto const expected = static_cast<key_type>(i * 2);

      REQUIRE(probed_keys[i] == expected);
      REQUIRE(matched_keys[i] == expected);
    }
  }

  SECTION("retrieve_if with explicit equality and hash should return nothing for false predicate.")
  {
    auto const pred = [] __device__(key_type) { return false; };

    auto const [probed_end, matched_end] = container.retrieve_if(keys_begin,
                                                                 keys_begin + num_keys,
                                                                 stencil.begin(),
                                                                 pred,
                                                                 container.key_eq(),
                                                                 container.hash_function(),
                                                                 probed_keys.begin(),
                                                                 matched_keys.begin());

    REQUIRE(static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end)) == 0);
    REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) == 0);
  }
}

template <class Container>
void test_retrieve_if_multiplicity(Container& container, std::size_t num_keys)
{
  using key_type = typename Container::key_type;

  constexpr std::size_t multiplicity = 2;

  container.clear();

  auto const num_unique_keys = num_keys / multiplicity;
  auto const num_actual_keys = num_unique_keys * multiplicity;

  auto const keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<key_type>(0),
    cuda::proclaim_return_type<key_type>([multiplicity] __device__(auto const& i) {
      return static_cast<key_type>(i / multiplicity);
    }));

  container.insert(keys_begin, keys_begin + num_actual_keys);
  REQUIRE(container.size() == num_actual_keys);

  thrust::device_vector<key_type> stencil(num_actual_keys);

  thrust::device_vector<key_type> probed_keys(num_actual_keys * multiplicity);
  thrust::device_vector<key_type> matched_keys(num_actual_keys * multiplicity);

  thrust::sequence(stencil.begin(), stencil.end(), key_type{1});

  SECTION("retrieve_if should filter duplicate matches using the stencil predicate.")
  {
    auto const pred = [] __device__(key_type value) { return value % 2 == 0; };

    auto const [probed_end, matched_end] = container.retrieve_if(keys_begin,
                                                                 keys_begin + num_actual_keys,
                                                                 stencil.begin(),
                                                                 pred,
                                                                 probed_keys.begin(),
                                                                 matched_keys.begin());

    auto const num_results =
      static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end));

    auto const expected_results = (num_actual_keys / 2) * multiplicity;

    REQUIRE(num_results == expected_results);
    REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) ==
            expected_results);

    thrust::sort_by_key(
      probed_keys.begin(), probed_end, matched_keys.begin(), cuda::std::less<key_type>());

    for (std::size_t i = 0; i < expected_results; ++i) {
      auto const input_index  = i / multiplicity;
      auto const expected_key = static_cast<key_type>((input_index * 2) / multiplicity);

      REQUIRE(probed_keys[i] == expected_key);
      REQUIRE(matched_keys[i] == expected_key);
    }
  }

  SECTION("retrieve_if should return nothing when the predicate is always false.")
  {
    auto const pred = [] __device__(key_type) { return false; };

    auto const [probed_end, matched_end] = container.retrieve_if(keys_begin,
                                                                 keys_begin + num_actual_keys,
                                                                 stencil.begin(),
                                                                 pred,
                                                                 probed_keys.begin(),
                                                                 matched_keys.begin());

    REQUIRE(static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end)) == 0);
    REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) == 0);
  }

  SECTION("retrieve_if should return all matches when the predicate is always true.")
  {
    auto const pred = [] __device__(key_type) { return true; };

    auto const [probed_end, matched_end] = container.retrieve_if(keys_begin,
                                                                 keys_begin + num_actual_keys,
                                                                 stencil.begin(),
                                                                 pred,
                                                                 probed_keys.begin(),
                                                                 matched_keys.begin());

    auto const expected_results = num_actual_keys * multiplicity;

    REQUIRE(static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end)) ==
            expected_results);
    REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) ==
            expected_results);

    thrust::sort_by_key(
      probed_keys.begin(), probed_end, matched_keys.begin(), cuda::std::less<key_type>());

    for (std::size_t key = 0; key < num_unique_keys; ++key) {
      auto const expected_key = static_cast<key_type>(key);

      auto const expected_count = multiplicity * multiplicity;

      for (std::size_t j = 0; j < expected_count; ++j) {
        auto const output_index = key * expected_count + j;

        REQUIRE(probed_keys[output_index] == expected_key);
        REQUIRE(matched_keys[output_index] == expected_key);
      }
    }
  }
}

template <class Container>
void test_retrieve_outer_if(Container& container, std::size_t num_keys)
{
  using key_type                = typename Container::key_type;
  auto const empty_key_sentinel = container.empty_key_sentinel();

  container.clear();

  auto const keys_begin = cuda::counting_iterator<key_type>{0};
  auto const query_size = num_keys * 2ull;

  container.insert(keys_begin, keys_begin + num_keys);

  thrust::device_vector<key_type> probes(query_size);
  thrust::device_vector<key_type> stencil(query_size);
  thrust::device_vector<key_type> probed_keys(query_size);
  thrust::device_vector<key_type> matched_keys(query_size);

  thrust::sequence(probes.begin(), probes.end(), key_type{0});

  SECTION("retrieve_outer_if should return matches and sentinels for misses.")
  {
    thrust::sequence(stencil.begin(), stencil.end(), key_type{0});

    auto const pred = [] __device__(key_type) { return true; };

    auto const [probed_end, matched_end] = container.retrieve_outer_if(probes.begin(),
                                                                       probes.end(),
                                                                       stencil.begin(),
                                                                       pred,
                                                                       container.key_eq(),
                                                                       container.hash_function(),
                                                                       probed_keys.begin(),
                                                                       matched_keys.begin());

    REQUIRE(static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end)) == query_size);
    REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) ==
            query_size);

    thrust::sort_by_key(
      probed_keys.begin(), probed_end, matched_keys.begin(), cuda::std::less<key_type>());

    REQUIRE(cuco::test::equal(
      probed_keys.begin(), probed_keys.end(), probes.begin(), cuda::std::equal_to<key_type>{}));

    REQUIRE(cuco::test::equal(matched_keys.begin(),
                              matched_keys.begin() + num_keys,
                              keys_begin,
                              cuda::std::equal_to<key_type>{}));

    REQUIRE(cuco::test::all_of(
      matched_keys.begin() + num_keys,
      matched_keys.end(),
      cuda::proclaim_return_type<bool>([empty_key_sentinel] __device__(auto const& k) {
        return static_cast<bool>(k == static_cast<key_type>(empty_key_sentinel));
      })));
  }

  SECTION("retrieve_outer_if should predicate on the stencil, not the probe.")
  {
    thrust::sequence(stencil.begin(), stencil.end(), key_type{1});

    auto const pred = [] __device__(key_type key) { return key % 2 == 0; };

    auto const [probed_end, matched_end] = container.retrieve_outer_if(probes.begin(),
                                                                       probes.end(),
                                                                       stencil.begin(),
                                                                       pred,
                                                                       container.key_eq(),
                                                                       container.hash_function(),
                                                                       probed_keys.begin(),
                                                                       matched_keys.begin());

    auto const expected_size = query_size / 2;

    REQUIRE(static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end)) ==
            expected_size);
    REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) ==
            expected_size);

    thrust::sort_by_key(
      probed_keys.begin(), probed_end, matched_keys.begin(), cuda::std::less<key_type>());

    for (std::size_t i = 0; i < expected_size; ++i) {
      auto const expected_probe = static_cast<key_type>(i * 2 + 1);

      REQUIRE(probed_keys[i] == expected_probe);

      if (expected_probe < static_cast<key_type>(num_keys)) {
        REQUIRE(matched_keys[i] == expected_probe);
      } else {
        REQUIRE(matched_keys[i] == static_cast<key_type>(empty_key_sentinel));
      }
    }
  }

  SECTION("retrieve_outer_if should return nothing for an always-false predicate.")
  {
    thrust::sequence(stencil.begin(), stencil.end(), key_type{0});

    auto const pred = [] __device__(key_type) { return false; };

    auto const [probed_end, matched_end] = container.retrieve_outer_if(probes.begin(),
                                                                       probes.end(),
                                                                       stencil.begin(),
                                                                       pred,
                                                                       container.key_eq(),
                                                                       container.hash_function(),
                                                                       probed_keys.begin(),
                                                                       matched_keys.begin());

    REQUIRE(static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end)) == 0);
    REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) == 0);
  }
}

template <class Container>
void test_retrieve_outer_if_multiplicity(Container& container, std::size_t num_keys)
{
  using key_type                = typename Container::key_type;
  auto const empty_key_sentinel = container.empty_key_sentinel();

  constexpr std::size_t multiplicity = 2;

  container.clear();

  auto const num_unique_keys = num_keys / multiplicity;
  auto const num_actual_keys = num_unique_keys * multiplicity;

  auto const keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<key_type>(0),
    cuda::proclaim_return_type<key_type>([multiplicity] __device__(auto const& i) {
      return static_cast<key_type>(i / multiplicity);
    }));

  container.insert(keys_begin, keys_begin + num_actual_keys);

  auto const query_size = num_unique_keys * 2ull;

  thrust::device_vector<key_type> probes(query_size);
  thrust::device_vector<key_type> stencil(query_size);
  thrust::device_vector<key_type> probed_keys(query_size * multiplicity);
  thrust::device_vector<key_type> matched_keys(query_size * multiplicity);

  thrust::sequence(probes.begin(), probes.end(), key_type{0});

  thrust::sequence(stencil.begin(), stencil.end(), key_type{1});

  auto const pred = [] __device__(key_type key) { return key % 2 == 0; };

  auto const [probed_end, matched_end] = container.retrieve_outer_if(probes.begin(),
                                                                     probes.end(),
                                                                     stencil.begin(),
                                                                     pred,
                                                                     container.key_eq(),
                                                                     container.hash_function(),
                                                                     probed_keys.begin(),
                                                                     matched_keys.begin());

  auto const num_matching_probes      = query_size / 2;
  auto const num_matching_unique_keys = num_unique_keys / 2;
  auto const num_missing_probes       = num_matching_probes - num_matching_unique_keys;

  auto const expected_results = num_matching_unique_keys * multiplicity + num_missing_probes;

  auto const num_results = static_cast<std::size_t>(std::distance(probed_keys.begin(), probed_end));

  REQUIRE(num_results == expected_results);
  REQUIRE(static_cast<std::size_t>(std::distance(matched_keys.begin(), matched_end)) ==
          expected_results);

  thrust::sort_by_key(
    probed_keys.begin(), probed_end, matched_keys.begin(), cuda::std::less<key_type>());

  std::size_t output_index = 0;

  for (std::size_t probe = 1; probe < query_size; probe += 2) {
    auto const expected_probe = static_cast<key_type>(probe);

    if (probe < num_unique_keys) {
      for (std::size_t j = 0; j < multiplicity; ++j) {
        REQUIRE(probed_keys[output_index] == expected_probe);
        REQUIRE(matched_keys[output_index] == expected_probe);
        ++output_index;
      }
    } else {
      REQUIRE(probed_keys[output_index] == expected_probe);
      REQUIRE(matched_keys[output_index] == static_cast<key_type>(empty_key_sentinel));
      ++output_index;
    }
  }

  REQUIRE(output_index == expected_results);
}

TEMPLATE_TEST_CASE_SIG(
  "static_multiset retrieve if tests",
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

  test_multiplicity(set, num_keys, 1);
  test_multiplicity(set, num_keys, 2);
  test_multiplicity(set, num_keys, 11);

  test_outer(set, num_keys);

  test_retrieve_if(set, num_keys);
  test_retrieve_if_with_probe(set, num_keys);
  test_retrieve_if_multiplicity(set, num_keys);

  test_retrieve_outer_if(set, num_keys);
  test_retrieve_outer_if_multiplicity(set, num_keys);
}
