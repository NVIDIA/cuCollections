/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/atomic>  // cuda::thread_scope

namespace cuco {

/**
 * @brief Strongly-typed wrapper for `cuda::thread_scope`.
 *
 * @tparam Scope `cuda::thread_scope` to be wrapped
 */
template <cuda::thread_scope Scope>
struct cuda_thread_scope {
  /**
   * @brief Implicit conversion to `cuda::thread_scope`.
   *
   * @return The wrapped `cuda::thread_scope`
   */
  __host__ __device__ constexpr operator cuda::thread_scope() const noexcept { return Scope; }
};

// alias definitions
inline constexpr auto thread_scope_system =
  cuda_thread_scope<cuda::thread_scope_system>{};  ///< `cuco::thread_scope_system`
inline constexpr auto thread_scope_device =
  cuda_thread_scope<cuda::thread_scope_device>{};  ///< `cuco::thread_scope_device`
inline constexpr auto thread_scope_block =
  cuda_thread_scope<cuda::thread_scope_block>{};  ///< `cuco::thread_scope_block`
inline constexpr auto thread_scope_thread =
  cuda_thread_scope<cuda::thread_scope_thread>{};  ///< `cuco::thread_scope_thread`

}  // namespace cuco
