/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
 */

#pragma once

#include <cuco/detail/utility/cuda.hpp>

#if defined(CUCO_DISABLE_KERNEL_VISIBILITY_WARNING_SUPPRESSION)
#define CUCO_SUPPRESS_KERNEL_WARNINGS
#elif defined(__NVCC__) && (defined(__GNUC__) || defined(__clang__))
// handle when nvcc is the CUDA compiler and gcc or clang is host
#define CUCO_SUPPRESS_KERNEL_WARNINGS _Pragma("nv_diag_suppress 1407")
_Pragma("GCC diagnostic ignored \"-Wattributes\"")
#elif defined(__clang__)
// handle when clang is the CUDA compiler
#define CUCO_SUPPRESS_KERNEL_WARNINGS _Pragma("clang diagnostic ignored \"-Wattributes\"")
#elif defined(__NVCOMPILER)
#define CUCO_SUPPRESS_KERNEL_WARNINGS #pragma diag_suppress attribute_requires_external_linkage
#endif

#ifndef CUCO_KERNEL
#define CUCO_KERNEL __attribute__((visibility("hidden"))) __global__
#endif
namespace cuco {
namespace detail {

/// CUDA warp size
__device__ constexpr int32_t warp_size() noexcept { return 32; }

/**
 * @brief Returns the global thread index in a 1D scalar grid
 *
 * @return The global thread index
 */
__device__ static index_type global_thread_id() noexcept
{
  return index_type{threadIdx.x} + index_type{blockDim.x} * index_type{blockIdx.x};
}

/**
 * @brief Returns the grid stride of a 1D grid
 *
 * @return The grid stride
 */
__device__ static index_type grid_stride() noexcept
{
  return index_type{gridDim.x} * index_type{blockDim.x};
}

}  // namespace detail
}  // namespace cuco
