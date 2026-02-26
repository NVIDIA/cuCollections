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

#include <cuco/hash_functions.cuh>
#include <cuco/hyperloglog.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>

TEST_CASE("hyperloglog: add_if with stencil predicate")
{
  auto constexpr hll_precision = 12;
  auto constexpr num_items     = 100000;

  double constexpr tolerance_factor = 2.5;
  double const relative_standard_deviation =
    1.04 / std::sqrt(static_cast<double>(1ull << hll_precision));

  thrust::device_vector<int64_t> items(num_items);
  thrust::device_vector<int> stencil(num_items);

  thrust::sequence(items.begin(), items.end(), 0);

  SECTION("Add only even items using stencil")
  {
    thrust::sequence(stencil.begin(), stencil.end(), 0);

    auto pred = [] __device__(int x) { return x % 2 == 0; };

    cuco::hyperloglog<int64_t, cuda::thread_scope_device, cuco::xxhash_64<int64_t>> estimator{
      cuco::precision(hll_precision)};

    REQUIRE(estimator.estimate() == 0);

    estimator.add_if_async(items.begin(), items.end(), stencil.begin(), pred);

    auto const estimate       = estimator.estimate();
    auto const expected_count = num_items / 2;

    double const relative_error =
      std::abs((static_cast<double>(estimate) / static_cast<double>(expected_count)) - 1.0);

    REQUIRE(relative_error < tolerance_factor * relative_standard_deviation);
  }

  SECTION("Add all items when predicate always returns true")
  {
    thrust::fill(stencil.begin(), stencil.end(), 1);

    auto pred = [] __device__(int x) { return x != 0; };

    cuco::hyperloglog<int64_t, cuda::thread_scope_device, cuco::xxhash_64<int64_t>> estimator{
      cuco::precision(hll_precision)};

    estimator.add_if_async(items.begin(), items.end(), stencil.begin(), pred);

    auto const estimate = estimator.estimate();

    double const relative_error =
      std::abs((static_cast<double>(estimate) / static_cast<double>(num_items)) - 1.0);

    REQUIRE(relative_error < tolerance_factor * relative_standard_deviation);
  }

  SECTION("Add no items when predicate always returns false")
  {
    thrust::fill(stencil.begin(), stencil.end(), 0);

    auto pred = [] __device__(int x) { return x != 0; };

    cuco::hyperloglog<int64_t, cuda::thread_scope_device, cuco::xxhash_64<int64_t>> estimator{
      cuco::precision(hll_precision)};

    estimator.add_if_async(items.begin(), items.end(), stencil.begin(), pred);

    auto const estimate = estimator.estimate();

    REQUIRE(estimate == 0);
  }
}
