/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstdint>

#include <vector>

namespace cuco::test {

inline std::vector<cuda::std::uint32_t> make_roaring_bitmap_without_runs_indices()
{
  std::vector<cuda::std::uint32_t> indices;
  for (cuda::std::uint32_t index = 0; index < 100000; index += 1000) {
    indices.push_back(index);
  }
  for (cuda::std::uint32_t index = 100000; index < 200000; ++index) {
    indices.push_back(3 * index);
  }
  for (cuda::std::uint32_t index = 700000; index < 800000; ++index) {
    indices.push_back(index);
  }
  return indices;
}

}  // namespace cuco::test
