/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include <cuda/functional>
#include <cuda/iterator>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

TEST_CASE("hyperloglog: type deduction bug with hash functions returning references")
{
  auto constexpr sketch_size_kb = 1;
  auto constexpr num_items      = 1000;

  auto first = cuda::make_transform_iterator(cuda::counting_iterator<uint64_t>(0),
                                             cuco::xxhash_64<uint64_t>{});
  auto last  = first + num_items;

  cuco::hyperloglog<uint64_t, cuda::thread_scope_device, cuda::std::identity> estimator{
    cuco::sketch_size_kb(sketch_size_kb)};

  REQUIRE(estimator.estimate() == 0);

  estimator.add(first, last);

  auto const estimate = estimator.estimate();

  REQUIRE(estimate > 0);
}
