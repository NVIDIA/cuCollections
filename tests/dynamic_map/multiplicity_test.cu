/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <cuco/dynamic_map.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE_SIG("dynamic_map: cross-submap duplicate handling",
                       "",
                       ((typename Key, typename T), Key, T),
                       (int32_t, int32_t),
                       (int64_t, int64_t))
{
  // Use capacity large enough to satisfy min_insert_size_ (10,000 default)
  // but small enough to force multiple submaps after several inserts
  constexpr std::size_t initial_capacity{50'000};
  constexpr std::size_t num_keys{20'000};  // Fill about 2/3 of first submap (load factor ~0.6)

  cuco::dynamic_map<Key, T> map{initial_capacity,
                                cuco::empty_key<Key>{-1},
                                cuco::empty_value<T>{-1},
                                cuco::erased_key<Key>{-2}};

  // Create pairs for first submap (keys 0 to num_keys-1)
  auto pairs_begin =
    cuda::make_transform_iterator(cuda::make_counting_iterator<int>(0),
                                  cuda::proclaim_return_type<cuco::pair<Key, T>>(
                                    [] __device__(auto i) { return cuco::pair<Key, T>(i, i); }));

  // Create pairs for second submap (keys num_keys to 2*num_keys-1)
  auto pairs_begin_2 =
    cuda::make_transform_iterator(cuda::make_counting_iterator<int>(num_keys),
                                  cuda::proclaim_return_type<cuco::pair<Key, T>>(
                                    [] __device__(auto i) { return cuco::pair<Key, T>(i, i); }));

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<T> d_results(num_keys);
  thrust::device_vector<bool> d_contained(num_keys);
  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());

  SECTION("insert does not insert duplicates across submaps")
  {
    // Fill first submap
    map.insert(pairs_begin, pairs_begin + num_keys);
    REQUIRE(map.size() == num_keys);

    // Insert new keys to trigger second submap creation
    map.insert(pairs_begin_2, pairs_begin_2 + num_keys);
    REQUIRE(map.size() == 2 * num_keys);

    // Try to insert duplicates of keys from first submap - should not increase size
    map.insert(pairs_begin, pairs_begin + num_keys);
    REQUIRE(map.size() == 2 * num_keys);

    // Try to insert duplicates with DIFFERENT values - should still not insert and preserve
    // originals
    auto duplicate_pairs_diff_values = cuda::make_transform_iterator(
      cuda::make_counting_iterator<int>(0),
      cuda::proclaim_return_type<cuco::pair<Key, T>>(
        [] __device__(auto i) { return cuco::pair<Key, T>(i, i + 999); }));
    map.insert(duplicate_pairs_diff_values, duplicate_pairs_diff_values + num_keys);
    REQUIRE(map.size() == 2 * num_keys);

    // Verify original values are preserved (not overwritten by duplicate insert attempts)
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    REQUIRE(cuco::test::equal(
      d_results.begin(), d_results.end(), cuda::counting_iterator<T>(0), cuda::std::equal_to<T>{}));
  }

  SECTION("contains finds keys in any submap")
  {
    // Fill first submap
    map.insert(pairs_begin, pairs_begin + num_keys);

    // Insert new keys to trigger second submap
    map.insert(pairs_begin_2, pairs_begin_2 + num_keys);

    // Keys in FIRST submap should be found
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());
    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));

    // Keys in SECOND submap should be found
    thrust::device_vector<Key> d_keys_2(num_keys);
    thrust::sequence(thrust::device, d_keys_2.begin(), d_keys_2.end(), num_keys);
    map.contains(d_keys_2.begin(), d_keys_2.end(), d_contained.begin());
    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));

    // Non-existent keys should NOT be found
    thrust::device_vector<Key> d_keys_nonexistent(num_keys);
    thrust::sequence(
      thrust::device, d_keys_nonexistent.begin(), d_keys_nonexistent.end(), 2 * num_keys);
    map.contains(d_keys_nonexistent.begin(), d_keys_nonexistent.end(), d_contained.begin());
    REQUIRE(cuco::test::none_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  }

  SECTION("find retrieves values from any submap")
  {
    // Fill first submap
    map.insert(pairs_begin, pairs_begin + num_keys);

    // Insert new keys to trigger second submap
    map.insert(pairs_begin_2, pairs_begin_2 + num_keys);

    // Find keys from FIRST submap - should return correct values
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    REQUIRE(cuco::test::equal(
      d_results.begin(), d_results.end(), cuda::counting_iterator<T>(0), cuda::std::equal_to<T>{}));

    // Find keys from SECOND submap - should return correct values
    thrust::device_vector<Key> d_keys_2(num_keys);
    thrust::sequence(thrust::device, d_keys_2.begin(), d_keys_2.end(), num_keys);
    map.find(d_keys_2.begin(), d_keys_2.end(), d_results.begin());
    REQUIRE(cuco::test::equal(d_results.begin(),
                              d_results.end(),
                              cuda::counting_iterator<T>(num_keys),
                              cuda::std::equal_to<T>{}));

    // Non-existent keys should return empty_value_sentinel (-1)
    thrust::device_vector<Key> d_keys_nonexistent(num_keys);
    thrust::sequence(
      thrust::device, d_keys_nonexistent.begin(), d_keys_nonexistent.end(), 2 * num_keys);
    map.find(d_keys_nonexistent.begin(), d_keys_nonexistent.end(), d_results.begin());
    REQUIRE(cuco::test::all_of(
      d_results.begin(), d_results.end(), [] __device__(T val) { return val == T{-1}; }));
  }

  SECTION("erase removes keys from any submap")
  {
    // Fill first submap
    map.insert(pairs_begin, pairs_begin + num_keys);

    // Insert new keys to trigger second submap
    map.insert(pairs_begin_2, pairs_begin_2 + num_keys);
    REQUIRE(map.size() == 2 * num_keys);

    // Erase keys from FIRST submap
    map.erase(d_keys.begin(), d_keys.end());
    REQUIRE(map.size() == num_keys);

    // Verify keys from first submap are no longer contained
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());
    REQUIRE(cuco::test::none_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));

    // Verify find returns sentinel for erased keys
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    REQUIRE(cuco::test::all_of(
      d_results.begin(), d_results.end(), [] __device__(T val) { return val == T{-1}; }));

    // Verify keys from SECOND submap are still there
    thrust::device_vector<Key> d_keys_2(num_keys);
    thrust::sequence(thrust::device, d_keys_2.begin(), d_keys_2.end(), num_keys);
    map.contains(d_keys_2.begin(), d_keys_2.end(), d_contained.begin());
    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));

    // Erase same keys again (already erased) - should not change size
    map.erase(d_keys.begin(), d_keys.end());
    REQUIRE(map.size() == num_keys);

    // Erase non-existent keys - should not change size
    thrust::device_vector<Key> d_keys_nonexistent(num_keys);
    thrust::sequence(
      thrust::device, d_keys_nonexistent.begin(), d_keys_nonexistent.end(), 3 * num_keys);
    map.erase(d_keys_nonexistent.begin(), d_keys_nonexistent.end());
    REQUIRE(map.size() == num_keys);

    // Now erase keys from SECOND submap
    map.erase(d_keys_2.begin(), d_keys_2.end());
    REQUIRE(map.size() == 0);

    // Verify all keys are gone
    map.contains(d_keys_2.begin(), d_keys_2.end(), d_contained.begin());
    REQUIRE(cuco::test::none_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  }

  SECTION("insert_or_assign updates in correct submap without creating duplicates")
  {
    // Fill first submap with values = keys
    map.insert(pairs_begin, pairs_begin + num_keys);

    // Insert new keys to trigger second submap
    map.insert(pairs_begin_2, pairs_begin_2 + num_keys);
    REQUIRE(map.size() == 2 * num_keys);

    // Create pairs with same keys as first submap but different values (value = key + 100)
    auto updated_pairs = cuda::make_transform_iterator(
      cuda::make_counting_iterator<int>(0),
      cuda::proclaim_return_type<cuco::pair<Key, T>>(
        [] __device__(auto i) { return cuco::pair<Key, T>(i, i + 100); }));

    // insert_or_assign should UPDATE values in first submap, not insert into second
    map.insert_or_assign(updated_pairs, updated_pairs + num_keys);
    REQUIRE(map.size() == 2 * num_keys);  // Size should not change

    // Verify values were updated in first submap
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    REQUIRE(cuco::test::equal(d_results.begin(),
                              d_results.end(),
                              cuda::counting_iterator<T>(100),  // Values should now be key + 100
                              cuda::std::equal_to<T>{}));

    // Verify second submap values are unchanged
    thrust::device_vector<Key> d_keys_2(num_keys);
    thrust::sequence(thrust::device, d_keys_2.begin(), d_keys_2.end(), num_keys);
    map.find(d_keys_2.begin(), d_keys_2.end(), d_results.begin());
    REQUIRE(cuco::test::equal(d_results.begin(),
                              d_results.end(),
                              cuda::counting_iterator<T>(num_keys),
                              cuda::std::equal_to<T>{}));

    // Test INSERT behavior: insert_or_assign with completely NEW keys should increase size
    auto new_pairs =
      cuda::make_transform_iterator(cuda::make_counting_iterator<int>(2 * num_keys),
                                    cuda::proclaim_return_type<cuco::pair<Key, T>>(
                                      [] __device__(auto i) { return cuco::pair<Key, T>(i, i); }));
    map.insert_or_assign(new_pairs, new_pairs + num_keys);
    REQUIRE(map.size() == 3 * num_keys);  // Size should increase by num_keys

    // Verify newly inserted keys exist with correct values
    thrust::device_vector<Key> d_keys_new(num_keys);
    thrust::sequence(thrust::device, d_keys_new.begin(), d_keys_new.end(), 2 * num_keys);
    map.find(d_keys_new.begin(), d_keys_new.end(), d_results.begin());
    REQUIRE(cuco::test::equal(d_results.begin(),
                              d_results.end(),
                              cuda::counting_iterator<T>(2 * num_keys),
                              cuda::std::equal_to<T>{}));

    // Test MIXED behavior: some keys exist (update), some don't (insert)
    // Use keys from 0 to num_keys/2 (exist in first submap) and
    // keys from 3*num_keys to 3*num_keys + num_keys/2 (don't exist)
    auto mixed_pairs = cuda::make_transform_iterator(
      cuda::make_counting_iterator<int>(0),
      cuda::proclaim_return_type<cuco::pair<Key, T>>([] __device__(auto i) {
        Key key = (i < num_keys / 2) ? Key(i) : Key(3 * num_keys + i - num_keys / 2);
        return cuco::pair<Key, T>(key, T(i + 500));
      }));
    std::size_t const size_before = map.size();
    map.insert_or_assign(mixed_pairs, mixed_pairs + num_keys);
    // Only num_keys/2 new keys should be inserted
    REQUIRE(map.size() == size_before + num_keys / 2);
  }

  SECTION("retrieve_all retrieves from all submaps")
  {
    // Fill first submap
    map.insert(pairs_begin, pairs_begin + num_keys);

    // Insert new keys to trigger second submap
    map.insert(pairs_begin_2, pairs_begin_2 + num_keys);
    REQUIRE(map.size() == 2 * num_keys);

    // Retrieve all key-value pairs
    thrust::device_vector<Key> d_retrieved_keys(2 * num_keys);
    thrust::device_vector<T> d_retrieved_values(2 * num_keys);
    auto const end = map.retrieve_all(d_retrieved_keys.begin(), d_retrieved_values.begin());
    auto const num_retrieved = std::distance(d_retrieved_keys.begin(), end.first);

    // Should retrieve all keys from both submaps
    REQUIRE(num_retrieved == 2 * num_keys);

    // Sort by key to verify all expected keys are present
    thrust::sort_by_key(
      thrust::device, d_retrieved_keys.begin(), d_retrieved_keys.end(), d_retrieved_values.begin());

    // Keys should be 0 to 2*num_keys-1
    thrust::device_vector<Key> d_expected_keys(2 * num_keys);
    thrust::sequence(thrust::device, d_expected_keys.begin(), d_expected_keys.end());
    REQUIRE(cuco::test::equal(d_retrieved_keys.begin(),
                              d_retrieved_keys.end(),
                              d_expected_keys.begin(),
                              cuda::std::equal_to<Key>{}));

    // Values should match keys (since we inserted key=value pairs)
    REQUIRE(cuco::test::equal(d_retrieved_values.begin(),
                              d_retrieved_values.end(),
                              cuda::counting_iterator<T>(0),
                              cuda::std::equal_to<T>{}));
  }
}
