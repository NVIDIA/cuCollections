/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/__config>
#include <cuco/hash_functions.cuh>

#include <nvbench/nvbench.cuh>

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
using HASH_RANGE = nvbench::type_list<cuco::identity_hash<char>,
                                      cuco::xxhash_32<char>,
                                      cuco::xxhash_64<char>,
                                      cuco::murmurhash3_32<char>>;  //,
// cuco::murmurhash3_x86_128<char>,
// cuco::murmurhash3_x64_128<char>>; // TODO handle tuple-like hash value

auto constexpr N             = 100'000'000;
auto constexpr OCCUPANCY     = 0.5;
auto constexpr MULTIPLICITY  = 1;
auto constexpr MATCHING_RATE = 1.0;
auto constexpr SKEW          = 0.5;
auto constexpr BATCH_SIZE    = 1'000'000;
auto constexpr INITIAL_SIZE  = 50'000'000;

// Sweeps keep the end points and the points where the curves bend. Denser sweeps can be run
// on demand with an nvbench axis override, e.g. `-a "Occupancy=[0.1:0.9:0.1]"`.
auto const N_RANGE         = std::vector<nvbench::int64_t>{10'000'000, 50'000'000, 90'000'000};
auto const N_RANGE_CACHE   = std::vector<nvbench::int64_t>{8'000, 800'000, 8'000'000, 80'000'000};
auto const OCCUPANCY_RANGE = std::vector<nvbench::float64_t>{0.1, 0.5, 0.8, 0.9};
auto const MULTIPLICITY_RANGE  = std::vector<nvbench::int64_t>{1, 4, 16};
auto const MATCHING_RATE_RANGE = std::vector<nvbench::float64_t>{0.1, 0.5, 1.};
auto const SKEW_RANGE          = std::vector<nvbench::float64_t>{0.1, 0.3, 1.};

}  // namespace cuco::benchmark::defaults
