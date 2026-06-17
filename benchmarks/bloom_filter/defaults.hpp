/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include <nvbench/nvbench.cuh>

#include <vector>

namespace cuco::benchmark::defaults {

using BF_KEY = nvbench::int64_t;

static constexpr auto BF_N = 1'000'000'000;

auto const BF_SIZE_MB_RANGE_CACHE =
  std::vector<nvbench::int64_t>{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

}  // namespace cuco::benchmark::defaults
