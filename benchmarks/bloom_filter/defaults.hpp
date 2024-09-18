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

#pragma once

#include <cuco/bloom_filter_policy.cuh>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <vector>

namespace cuco::benchmark::defaults {

static constexpr auto BF_N       = 400'000'000;
static constexpr auto BF_SIZE_MB = 2'000;
using BF_POLICY                  = cuco::default_filter_policy<char>;
using BF_HASH                    = typename BF_POLICY::hasher;
using BF_BLOCK = cuda::std::array<typename BF_POLICY::word_type, BF_POLICY::words_per_block>;
// This is a dummy value which will be dynamically replaced with the filter's actual default
auto constexpr BF_PATTERN_BITS = 0;
auto const BF_SIZE_MB_RANGE_CACHE =
  std::vector<nvbench::int64_t>{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
auto const BF_PATTERN_BITS_RANGE = std::vector<nvbench::int64_t>{1, 2, 4, 6, 8, 16};

}  // namespace cuco::benchmark::defaults
