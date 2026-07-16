/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
