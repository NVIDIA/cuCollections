/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstdint>

namespace cuco::utility {

/**
 * @brief Size in bytes of a global-memory sector.
 */
inline constexpr cuda::std::uint32_t sector_size_bytes = 32;

}  // namespace cuco::utility
