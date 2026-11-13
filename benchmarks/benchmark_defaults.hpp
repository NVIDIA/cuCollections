/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/__config>
#include <cuco/hash_functions.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/functional>

#include <cstdint>
#include <vector>

namespace cuco::benchmark::defaults {

#if defined(CUCO_HAS_128BIT_ATOMICS)
using KEY_TYPE_RANGE   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t, __int128_t>;
using VALUE_TYPE_RANGE = nvbench::type_list<nvbench::int32_t, nvbench::int64_t, __int128_t>;
#else
using KEY_TYPE_RANGE   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using VALUE_TYPE_RANGE = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
#endif
using HASH_RANGE = nvbench::type_list<cuda::hash<char, cuda::hash_algorithm::xxhash_32>,
                                      cuda::hash<char, cuda::hash_algorithm::xxhash_64>,
                                      cuda::hash<char, cuda::hash_algorithm::murmurhash3_32>>;

auto constexpr N             = 100'000'000;
auto constexpr OCCUPANCY     = 0.5;
auto constexpr MULTIPLICITY  = 1;
auto constexpr MATCHING_RATE = 1.0;
auto constexpr SKEW          = 0.5;
auto constexpr BATCH_SIZE    = 1'000'000;
auto constexpr INITIAL_SIZE  = 50'000'000;

auto const N_RANGE = nvbench::range(10'000'000, 100'000'000, 20'000'000);
auto const N_RANGE_CACHE =
  std::vector<nvbench::int64_t>{8'000, 80'000, 800'000, 8'000'000, 80'000'000};
auto const OCCUPANCY_RANGE     = nvbench::range(0.1, 0.9, 0.1);
auto const MULTIPLICITY_RANGE  = std::vector<nvbench::int64_t>{1, 2, 4, 8, 16};
auto const MATCHING_RATE_RANGE = nvbench::range(0.1, 1., 0.1);
auto const SKEW_RANGE          = nvbench::range(0.1, 1., 0.1);

}  // namespace cuco::benchmark::defaults
