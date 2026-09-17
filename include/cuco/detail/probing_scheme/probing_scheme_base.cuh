/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuco {
namespace detail {

/**
 * @brief Base class of public probing scheme.
 *
 * This class should not be used directly.
 *
 * @note Derived probing schemes must produce bucket-aligned slot indices. For a bucket size
 * `B`, every index must be a multiple of `B` and the complete bucket must fit within the storage.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 */
template <int32_t CGSize>
class probing_scheme_base {
 public:
  /**
   * @brief The size of the CUDA cooperative thread group.
   */
  static constexpr int32_t cg_size = CGSize;
};
}  // namespace detail
}  // namespace cuco
