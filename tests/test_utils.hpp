/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cooperative_groups.h>

#include <cstdint>

namespace cuco {
namespace test {

namespace cg = cooperative_groups;

enum class probe_sequence { linear_probing, double_hashing };

inline bool modulo_bitgen(uint64_t i) { return i % 7 == 0; }

}  // namespace test
}  // namespace cuco
