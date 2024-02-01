/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <utils.hpp>

#include <cuco/distinct_count_estimator.cuh>
#include <cuco/hash_functions.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>

TEMPLATE_TEST_CASE_SIG("distinct_count_estimator: unique sequence",
                       "",
                       ((typename T, int32_t Precision, typename Hash), T, Precision, Hash),
                       (int32_t, 9, cuco::xxhash_32<int32_t>),
                       (int32_t, 10, cuco::xxhash_32<int32_t>),
                       (int32_t, 11, cuco::xxhash_32<int32_t>),
                       (int32_t, 12, cuco::xxhash_32<int32_t>),
                       (int32_t, 13, cuco::xxhash_32<int32_t>),
                       (int32_t, 9, cuco::xxhash_64<int32_t>),
                       (int32_t, 10, cuco::xxhash_64<int32_t>),
                       (int32_t, 11, cuco::xxhash_64<int32_t>),
                       (int32_t, 12, cuco::xxhash_64<int32_t>),
                       (int32_t, 13, cuco::xxhash_64<int32_t>),
                       (int64_t, 9, cuco::xxhash_32<int64_t>),
                       (int64_t, 10, cuco::xxhash_32<int64_t>),
                       (int64_t, 11, cuco::xxhash_32<int64_t>),
                       (int64_t, 12, cuco::xxhash_32<int64_t>),
                       (int64_t, 13, cuco::xxhash_32<int64_t>),
                       (int64_t, 9, cuco::xxhash_64<int64_t>),
                       (int64_t, 10, cuco::xxhash_64<int64_t>),
                       (int64_t, 11, cuco::xxhash_64<int64_t>),
                       (int64_t, 12, cuco::xxhash_64<int64_t>),
                       (int64_t, 13, cuco::xxhash_64<int64_t>),
                       (__int128_t, 9, cuco::xxhash_32<__int128_t>),
                       (__int128_t, 10, cuco::xxhash_32<__int128_t>),
                       (__int128_t, 11, cuco::xxhash_32<__int128_t>),
                       (__int128_t, 12, cuco::xxhash_32<__int128_t>),
                       (__int128_t, 13, cuco::xxhash_32<__int128_t>),
                       (__int128_t, 9, cuco::xxhash_64<__int128_t>),
                       (__int128_t, 10, cuco::xxhash_64<__int128_t>),
                       (__int128_t, 11, cuco::xxhash_64<__int128_t>),
                       (__int128_t, 12, cuco::xxhash_64<__int128_t>),
                       (__int128_t, 13, cuco::xxhash_64<__int128_t>))
{
  // This factor determines the error threshold for passing the test
  // TODO might be too high
  double constexpr tolerance_factor = 3.0;
  // RSD for a given precision is given by the following formula
  double const relative_standard_deviation =
    1.04 / std::sqrt(static_cast<double>(1ull << Precision));

  auto num_items_pow2 = GENERATE(25, 26, 28);
  INFO("num_items=2^" << num_items_pow2);
  auto num_items = 1ull << num_items_pow2;

  thrust::device_vector<T> items(num_items);

  // Generate `num_items` distinct items
  thrust::sequence(items.begin(), items.end(), 0);

  // Initialize the estimator
  cuco::distinct_count_estimator<T> estimator;

  REQUIRE(estimator.estimate() == 0);

  // Add all items to the estimator
  estimator.add(items.begin(), items.end());

  auto const estimate = estimator.estimate();

  // Adding the same items again should not affect the result
  estimator.add(items.begin(), items.begin() + num_items / 2);
  REQUIRE(estimator.estimate() == estimate);

  // Clearing the estimator shoult reset the estimate
  estimator.clear();
  REQUIRE(estimator.estimate() == 0);

  double const relative_error =
    std::abs(static_cast<double>(num_items) - static_cast<double>(estimate)) / num_items;

  // Check if the error is acceptable
  REQUIRE(relative_error < tolerance_factor * relative_standard_deviation);
}
