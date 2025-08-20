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

#include <cuco/static_multimap.cuh>

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
  using mapped_type             = typename Container::mapped_type;
  using value_type              = typename Container::value_type;
  auto const empty_key_sentinel = container.empty_key_sentinel();

  container.clear();

  // Insert key-value pairs: (0,0), (1,1), (2,2), ... (num_keys-1, num_keys-1)
  auto const pairs_begin =
    thrust::make_transform_iterator(thrust::counting_iterator<key_type>{0},
                                    cuda::proclaim_return_type<value_type>([] __device__(auto i) {
                                      return cuco::pair<key_type, mapped_type>{i, i};
                                    }));

  container.insert(pairs_begin, pairs_begin + num_keys);
  REQUIRE(container.size() == num_keys);

  // Create stencil and predicate for 25% filtering (keys divisible by 4)
  auto const stencil_begin = thrust::counting_iterator<key_type>{0};
  auto const pred          = [] __device__(key_type k) { return k % 4 == 0; };

  thrust::device_vector<key_type> probed_keys(num_keys);
  thrust::device_vector<value_type> matched_pairs(num_keys);

  SECTION("Inner retrieve_if should only return key-value pairs where predicate is true.")
  {
    auto const [probed_end, matched_end] = container.retrieve_if<false>(stencil_begin,
                                                                        stencil_begin + num_keys,
                                                                        stencil_begin,
                                                                        pred,
                                                                        probed_keys.begin(),
                                                                        matched_pairs.begin());

    auto const num_retrieved  = std::distance(probed_keys.begin(), probed_end);
    auto const expected_count = (num_keys + 3) / 4;  // ceiling of num_keys/4 for keys 0, 4, 8, ...

    REQUIRE(num_retrieved == expected_count);
    REQUIRE(std::distance(matched_pairs.begin(), matched_end) == expected_count);

    // Sort results for comparison
    thrust::sort_by_key(probed_keys.begin(), probed_end, matched_pairs.begin());

    // Check that all retrieved keys are divisible by 4
    for (std::size_t i = 0; i < expected_count; ++i) {
      auto expected_key = static_cast<key_type>(i * 4);
      REQUIRE(probed_keys[i] == expected_key);
      REQUIRE(matched_pairs[i].first == expected_key);
      REQUIRE(matched_pairs[i].second == expected_key);  // value equals key in our test
    }
  }

  SECTION("Inner retrieve_if with always-false predicate should return empty.")
  {
    auto const always_false              = [] __device__(key_type) { return false; };
    auto const [probed_end, matched_end] = container.retrieve_if<false>(stencil_begin,
                                                                        stencil_begin + num_keys,
                                                                        stencil_begin,
                                                                        always_false,
                                                                        probed_keys.begin(),
                                                                        matched_pairs.begin());

    REQUIRE(std::distance(probed_keys.begin(), probed_end) == 0);
    REQUIRE(std::distance(matched_pairs.begin(), matched_end) == 0);
  }

  SECTION("Inner retrieve_if with always-true predicate should return all pairs.")
  {
    auto const always_true               = [] __device__(key_type) { return true; };
    auto const [probed_end, matched_end] = container.retrieve_if<false>(stencil_begin,
                                                                        stencil_begin + num_keys,
                                                                        stencil_begin,
                                                                        always_true,
                                                                        probed_keys.begin(),
                                                                        matched_pairs.begin());

    REQUIRE(std::distance(probed_keys.begin(), probed_end) == num_keys);
    REQUIRE(std::distance(matched_pairs.begin(), matched_end) == num_keys);

    thrust::sort_by_key(probed_keys.begin(), probed_end, matched_pairs.begin());

    for (std::size_t i = 0; i < num_keys; ++i) {
      auto expected_key = static_cast<key_type>(i);
      REQUIRE(probed_keys[i] == expected_key);
      REQUIRE(matched_pairs[i].first == expected_key);
      REQUIRE(matched_pairs[i].second == expected_key);
    }
  }
}

template <class Container>
void test_retrieve_if_outer(Container& container, std::size_t num_keys)
{
  using key_type                  = typename Container::key_type;
  using mapped_type               = typename Container::mapped_type;
  using value_type                = typename Container::value_type;
  auto const empty_key_sentinel   = container.empty_key_sentinel();
  auto const empty_value_sentinel = container.empty_value_sentinel();

  container.clear();

  // Insert only even-indexed key-value pairs: (0,0), (2,2), (4,4), ...
  auto const even_pairs_begin =
    thrust::make_transform_iterator(thrust::counting_iterator<key_type>{0},
                                    cuda::proclaim_return_type<value_type>([] __device__(auto i) {
                                      return cuco::pair<key_type, mapped_type>{i * 2, i * 2};
                                    }));
  container.insert(even_pairs_begin, even_pairs_begin + num_keys / 2);

  // Query all keys 0, 1, 2, ... num_keys-1
  auto const query_keys_begin = thrust::counting_iterator<key_type>{0};
  auto const stencil_begin    = query_keys_begin;
  auto const pred = [] __device__(key_type k) { return k % 6 == 0; };  // keys 0, 6, 12, ...

  thrust::device_vector<key_type> probed_keys(num_keys);
  thrust::device_vector<value_type> matched_pairs(num_keys);

  SECTION("Outer retrieve_if should include all queried keys with appropriate matches.")
  {
    auto const [probed_end, matched_end] = container.retrieve_if<true>(query_keys_begin,
                                                                       query_keys_begin + num_keys,
                                                                       stencil_begin,
                                                                       pred,
                                                                       probed_keys.begin(),
                                                                       matched_pairs.begin());

    // Should return entries for all queried keys
    REQUIRE(std::distance(probed_keys.begin(), probed_end) == num_keys);
    REQUIRE(std::distance(matched_pairs.begin(), matched_end) == num_keys);

    // Sort by probed keys for easier verification
    thrust::sort_by_key(probed_keys.begin(), probed_end, matched_pairs.begin());

    // Check that all queried keys are present
    for (std::size_t i = 0; i < num_keys; ++i) {
      auto key = static_cast<key_type>(i);
      REQUIRE(probed_keys[i] == key);

      if (key % 6 == 0 && key % 2 == 0) {
        // Predicate true and key exists in container -> should match
        REQUIRE(matched_pairs[i].first == key);
        REQUIRE(matched_pairs[i].second == key);
      } else {
        // Either predicate false or key doesn't exist -> empty sentinel
        REQUIRE(matched_pairs[i].first == empty_key_sentinel);
        REQUIRE(matched_pairs[i].second == empty_value_sentinel);
      }
    }
  }
}

template <class Container>
void test_retrieve_if_multiplicity(Container& container,
                                   std::size_t num_keys,
                                   std::size_t multiplicity)
{
  using key_type    = typename Container::key_type;
  using mapped_type = typename Container::mapped_type;
  using value_type  = typename Container::value_type;

  container.clear();

  auto const num_unique_keys = num_keys / multiplicity;
  REQUIRE(num_unique_keys > 0);
  auto const num_actual_pairs = num_unique_keys * multiplicity;

  // Insert multiple instances of each key with different values
  auto const pairs_begin = thrust::make_transform_iterator(
    thrust::counting_iterator<key_type>{0},
    cuda::proclaim_return_type<value_type>([multiplicity] __device__(auto i) {
      auto key   = static_cast<key_type>(i / multiplicity);
      auto value = static_cast<mapped_type>(i);  // unique value for each pair
      return cuco::pair<key_type, mapped_type>{key, value};
    }));

  container.insert(pairs_begin, pairs_begin + num_actual_pairs);
  REQUIRE(container.size() == num_actual_pairs);

  // Use stencil based on key index, predicate for 50% of keys (even keys)
  auto const stencil_begin = thrust::make_transform_iterator(
    thrust::counting_iterator<key_type>{0},
    [] __device__(key_type i) { return i / multiplicity; });  // extract key from index
  auto const pred = [] __device__(key_type k) { return k % 2 == 0; };

  thrust::device_vector<key_type> probed_keys(num_actual_pairs);
  thrust::device_vector<value_type> matched_pairs(num_actual_pairs);

  SECTION("retrieve_if should return all instances of keys that satisfy predicate.")
  {
    auto const [probed_end, matched_end] =
      container.retrieve_if<false>(pairs_begin,
                                   pairs_begin + num_actual_pairs,
                                   stencil_begin,
                                   pred,
                                   probed_keys.begin(),
                                   matched_pairs.begin());

    // Should return all instances of even keys
    auto const expected_unique_keys = (num_unique_keys + 1) / 2;  // ceiling for even keys
    auto const expected_total_pairs = expected_unique_keys * multiplicity;

    REQUIRE(std::distance(probed_keys.begin(), probed_end) == expected_total_pairs);
    REQUIRE(std::distance(matched_pairs.begin(), matched_end) == expected_total_pairs);

    // Sort results for verification
    thrust::sort_by_key(probed_keys.begin(), probed_end, matched_pairs.begin());

    // Verify all returned keys are even and all instances are present
    for (std::size_t i = 0; i < expected_total_pairs; ++i) {
      auto expected_key = static_cast<key_type>((i / multiplicity) * 2);
      REQUIRE(probed_keys[i] == expected_key);
      REQUIRE(matched_pairs[i].first == expected_key);
      // Values should be unique but we don't need to verify exact order
      REQUIRE(matched_pairs[i].second >= 0);
    }
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_multimap retrieve_if tests",
  "",
  ((typename Key, typename Value, cuco::test::probe_sequence Probe, int CGSize),
   Key,
   Value,
   Probe,
   CGSize),
  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int32_t, int64_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, int64_t, cuco::test::probe_sequence::linear_probing, 2))
{
  constexpr std::size_t num_keys{400};
  constexpr double desired_load_factor = 0.5;
  constexpr auto empty_key_sentinel    = std::numeric_limits<Key>::max();
  constexpr auto empty_value_sentinel  = std::numeric_limits<Value>::max();

  using extent_type = cuco::extent<std::size_t>;
  using probe       = std::conditional_t<
          Probe == cuco::test::probe_sequence::linear_probing,
          cuco::linear_probing<CGSize, cuco::murmurhash3_32<Key>>,
          cuco::double_hashing<CGSize, cuco::murmurhash3_32<Key>, cuco::murmurhash3_32<Key>>>;

  auto map = cuco::experimental::static_multimap<Key,
                                                 Value,
                                                 extent_type,
                                                 cuda::thread_scope_device,
                                                 cuda::std::equal_to<Key>,
                                                 probe,
                                                 cuco::cuda_allocator<cuda::std::byte>,
                                                 cuco::storage<2>>{
    num_keys * 2,
    cuco::empty_key<Key>{empty_key_sentinel},
    cuco::empty_value<Value>{empty_value_sentinel}};

  test_retrieve_if_inner(map, num_keys);
  test_retrieve_if_outer(map, num_keys);
  test_retrieve_if_multiplicity(map, num_keys, 3);  // Each key appears 3 times
}
