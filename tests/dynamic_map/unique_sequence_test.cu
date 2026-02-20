/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.
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

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE_SIG("dynamic_map: unique sequence",
                       "",
                       ((typename Key, typename T), Key, T),
                       (int32_t, int32_t),
                       (int32_t, int64_t),
                       (int64_t, int32_t),
                       (int64_t, int64_t))
{
  constexpr std::size_t num_keys{1'000'000};

  cuco::dynamic_map<Key, T> map{30'000'000, cuco::empty_key<Key>{-1}, cuco::empty_value<T>{-1}};

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<T> d_values(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  thrust::sequence(thrust::device, d_values.begin(), d_values.end());

  auto pairs_begin =
    cuda::make_transform_iterator(cuda::make_counting_iterator<int>(0),
                                  cuda::proclaim_return_type<cuco::pair<Key, T>>(
                                    [] __device__(auto i) { return cuco::pair<Key, T>(i, i); }));

  thrust::device_vector<T> d_results(num_keys);
  thrust::device_vector<bool> d_contained(num_keys);

  SECTION("All inserted keys-value pairs should be contained")
  {
    map.insert(pairs_begin, pairs_begin + num_keys);
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());

    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  }

  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());

    REQUIRE(cuco::test::none_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  }

  SECTION("size() returns correct count after insertions")
  {
    REQUIRE(map.size() == 0);

    map.insert(pairs_begin, pairs_begin + num_keys / 2);
    REQUIRE(map.size() == num_keys / 2);

    map.insert(pairs_begin + num_keys / 2, pairs_begin + num_keys);
    REQUIRE(map.size() == num_keys);
  }

  SECTION("capacity() returns non-zero value")
  {
    REQUIRE(map.capacity() > 0);
    REQUIRE(map.capacity() >= 30'000'000);
  }

  SECTION("load_factor() is computed correctly")
  {
    REQUIRE(map.load_factor() == 0.0f);

    map.insert(pairs_begin, pairs_begin + num_keys);

    float expected_load_factor = static_cast<float>(num_keys) / map.capacity();
    REQUIRE(map.load_factor() == expected_load_factor);
  }

  SECTION("insert_or_assign inserts new keys and updates existing")
  {
    // Insert initial keys
    map.insert(pairs_begin, pairs_begin + num_keys);
    REQUIRE(map.size() == num_keys);

    // Create pairs with same keys but different values (value = key + 1)
    auto updated_pairs_begin = cuda::make_transform_iterator(
      cuda::make_counting_iterator<int>(0),
      cuda::proclaim_return_type<cuco::pair<Key, T>>(
        [] __device__(auto i) { return cuco::pair<Key, T>(i, i + 1); }));

    // insert_or_assign should update existing keys, size should stay the same
    map.insert_or_assign(updated_pairs_begin, updated_pairs_begin + num_keys);
    REQUIRE(map.size() == num_keys);

    // Verify values were updated
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    REQUIRE(cuco::test::equal(d_results.begin(),
                              d_results.end(),
                              cuda::counting_iterator<T>(1),  // Values should now be key + 1
                              cuda::std::equal_to<T>{}));

    // Insert new keys with insert_or_assign (keys from num_keys to 2*num_keys)
    auto new_pairs_begin =
      cuda::make_transform_iterator(cuda::make_counting_iterator<int>(num_keys),
                                    cuda::proclaim_return_type<cuco::pair<Key, T>>(
                                      [] __device__(auto i) { return cuco::pair<Key, T>(i, i); }));

    map.insert_or_assign(new_pairs_begin, new_pairs_begin + num_keys);
    REQUIRE(map.size() == 2 * num_keys);
  }
}
