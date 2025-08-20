/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

#include <catch2/catch_template_test_macros.hpp>

#include <limits>

template <class Container>
void test_retrieve_if_inner(Container& container, std::size_t num_keys)
{
  using key_type                = typename Container::key_type;
  auto const empty_key_sentinel = container.empty_key_sentinel();

  container.clear();

  // Insert keys 0, 1, 2, ... num_keys-1
  auto const keys_begin = thrust::counting_iterator<key_type>{0};
  container.insert(keys_begin, keys_begin + num_keys);
  REQUIRE(container.size() == num_keys);

  // Create stencil and predicate for 50% filtering (even keys only)
  auto const stencil_begin = keys_begin;
  auto const pred          = [] __device__(key_type k) { return k % 2 == 0; };

  thrust::device_vector<key_type> probed_keys(num_keys);
  thrust::device_vector<key_type> matched_keys(num_keys);

  SECTION("Inner retrieve_if should only return keys where predicate is true.")
  {
    auto const [probed_end, matched_end] = container.retrieve_if<false>(keys_begin,
                                                                        keys_begin + num_keys,
                                                                        stencil_begin,
                                                                        pred,
                                                                        probed_keys.begin(),
                                                                        matched_keys.begin());

    auto const num_retrieved  = std::distance(probed_keys.begin(), probed_end);
    auto const expected_count = (num_keys + 1) / 2;  // ceiling of num_keys/2 for even keys

    REQUIRE(num_retrieved == expected_count);
    REQUIRE(std::distance(matched_keys.begin(), matched_end) == expected_count);

    // Sort results for comparison
    thrust::sort(probed_keys.begin(), probed_end);
    thrust::sort(matched_keys.begin(), matched_end);

    // Check that all retrieved keys are even
    auto const even_keys_begin = thrust::make_transform_iterator(
      thrust::counting_iterator<key_type>{0}, [] __device__(key_type i) { return i * 2; });

    REQUIRE(cuco::test::equal(
      probed_keys.begin(), probed_end, even_keys_begin, cuda::std::equal_to<key_type>{}));
    REQUIRE(cuco::test::equal(
      matched_keys.begin(), matched_end, even_keys_begin, cuda::std::equal_to<key_type>{}));
  }

  SECTION("Inner retrieve_if with always-false predicate should return empty.")
  {
    auto const always_false              = [] __device__(key_type) { return false; };
    auto const [probed_end, matched_end] = container.retrieve_if<false>(keys_begin,
                                                                        keys_begin + num_keys,
                                                                        stencil_begin,
                                                                        always_false,
                                                                        probed_keys.begin(),
                                                                        matched_keys.begin());

    REQUIRE(std::distance(probed_keys.begin(), probed_end) == 0);
    REQUIRE(std::distance(matched_keys.begin(), matched_end) == 0);
  }

  SECTION("Inner retrieve_if with always-true predicate should return all keys.")
  {
    auto const always_true               = [] __device__(key_type) { return true; };
    auto const [probed_end, matched_end] = container.retrieve_if<false>(keys_begin,
                                                                        keys_begin + num_keys,
                                                                        stencil_begin,
                                                                        always_true,
                                                                        probed_keys.begin(),
                                                                        matched_keys.begin());

    REQUIRE(std::distance(probed_keys.begin(), probed_end) == num_keys);
    REQUIRE(std::distance(matched_keys.begin(), matched_end) == num_keys);

    thrust::sort(probed_keys.begin(), probed_end);
    thrust::sort(matched_keys.begin(), matched_end);

    REQUIRE(cuco::test::equal(
      probed_keys.begin(), probed_end, keys_begin, cuda::std::equal_to<key_type>{}));
    REQUIRE(cuco::test::equal(
      matched_keys.begin(), matched_end, keys_begin, cuda::std::equal_to<key_type>{}));
  }
}

template <class Container>
void test_retrieve_if_outer(Container& container, std::size_t num_keys)
{
  using key_type                = typename Container::key_type;
  auto const empty_key_sentinel = container.empty_key_sentinel();

  container.clear();

  // Insert only half of the keys (even keys: 0, 2, 4, ...)
  auto const even_keys_begin = thrust::make_transform_iterator(
    thrust::counting_iterator<key_type>{0}, [] __device__(key_type i) { return i * 2; });
  container.insert(even_keys_begin, even_keys_begin + num_keys / 2);

  // Query all keys 0, 1, 2, ... num_keys-1
  auto const query_keys_begin = thrust::counting_iterator<key_type>{0};
  auto const stencil_begin    = query_keys_begin;
  auto const pred = [] __device__(key_type k) { return k % 4 == 0; };  // keys 0, 4, 8, ...

  thrust::device_vector<key_type> probed_keys(num_keys);
  thrust::device_vector<key_type> matched_keys(num_keys);

  SECTION("Outer retrieve_if should include all queried keys with appropriate matches.")
  {
    auto const [probed_end, matched_end] = container.retrieve_if<true>(query_keys_begin,
                                                                       query_keys_begin + num_keys,
                                                                       stencil_begin,
                                                                       pred,
                                                                       probed_keys.begin(),
                                                                       matched_keys.begin());

    // Should return entries for all queried keys
    REQUIRE(std::distance(probed_keys.begin(), probed_end) == num_keys);
    REQUIRE(std::distance(matched_keys.begin(), matched_end) == num_keys);

    // Sort by probed keys for easier verification
    thrust::sort_by_key(probed_keys.begin(), probed_end, matched_keys.begin());

    // Check that all queried keys are present
    REQUIRE(cuco::test::equal(
      probed_keys.begin(), probed_end, query_keys_begin, cuda::std::equal_to<key_type>{}));

    // For keys where predicate is false, should have empty sentinel
    // For keys where predicate is true but key not in container, should have empty sentinel
    // For keys where predicate is true and key is in container, should have the key as value
    for (std::size_t i = 0; i < num_keys; ++i) {
      auto key     = static_cast<key_type>(i);
      auto matched = matched_keys[i];

      if (key % 4 == 0 && key % 2 == 0) {
        // Predicate true and key exists in container
        REQUIRE(matched == key);
      } else {
        // Either predicate false or key doesn't exist
        REQUIRE(matched == empty_key_sentinel);
      }
    }
  }
}

template <class Container>
void test_retrieve_if_stencil_mismatch(Container& container, std::size_t num_keys)
{
  using key_type = typename Container::key_type;

  container.clear();

  // Insert keys 0, 1, 2, ... num_keys-1
  auto const keys_begin = thrust::counting_iterator<key_type>{0};
  container.insert(keys_begin, keys_begin + num_keys);

  // Use a different stencil that doesn't match the keys directly
  thrust::device_vector<key_type> stencil_values(num_keys);
  thrust::transform(
    keys_begin, keys_begin + num_keys, stencil_values.begin(), [] __device__(key_type k) {
      return k + 10;
    });  // offset by 10

  auto const pred = [] __device__(key_type s) { return s % 3 == 1; };  // 11, 14, 17, ...

  thrust::device_vector<key_type> probed_keys(num_keys);
  thrust::device_vector<key_type> matched_keys(num_keys);

  SECTION("retrieve_if should use stencil values for predicate evaluation.")
  {
    auto const [probed_end, matched_end] = container.retrieve_if<false>(keys_begin,
                                                                        keys_begin + num_keys,
                                                                        stencil_values.begin(),
                                                                        pred,
                                                                        probed_keys.begin(),
                                                                        matched_keys.begin());

    // Count expected results: stencil values 11, 14, 17, ... (where (k+10) % 3 == 1)
    // This corresponds to original keys 1, 4, 7, ...
    auto const expected_count = (num_keys + 2) / 3;  // keys where k % 3 == 1

    REQUIRE(std::distance(probed_keys.begin(), probed_end) == expected_count);
    REQUIRE(std::distance(matched_keys.begin(), matched_end) == expected_count);

    // Sort results
    thrust::sort(probed_keys.begin(), probed_end);
    thrust::sort(matched_keys.begin(), matched_end);

    // Expected keys: 1, 4, 7, ...
    auto const expected_keys_begin = thrust::make_transform_iterator(
      thrust::counting_iterator<key_type>{0}, [] __device__(key_type i) { return 1 + i * 3; });

    REQUIRE(cuco::test::equal(
      probed_keys.begin(), probed_end, expected_keys_begin, cuda::std::equal_to<key_type>{}));
    REQUIRE(cuco::test::equal(
      matched_keys.begin(), matched_end, expected_keys_begin, cuda::std::equal_to<key_type>{}));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_multiset retrieve_if tests",
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
  constexpr std::size_t num_keys{400};
  constexpr double desired_load_factor = 0.5;
  constexpr auto empty_key_sentinel    = std::numeric_limits<Key>::max();

  using probe = std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                                   cuco::linear_probing<CGSize, cuco::default_hash_function<Key>>,
                                   cuco::double_hashing<CGSize, cuco::default_hash_function<Key>>>;

  auto set = cuco::static_multiset{
    num_keys, desired_load_factor, cuco::empty_key<Key>{empty_key_sentinel}, {}, probe{}};

  test_retrieve_if_inner(set, num_keys);
  test_retrieve_if_outer(set, num_keys);
  test_retrieve_if_stencil_mismatch(set, num_keys);
}
