/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
