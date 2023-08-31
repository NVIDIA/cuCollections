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

namespace cuco {
namespace detail {

/**
￼   * @brief Returns the global thread index in a 1D scalar grid
*
* @tparam CGSize Number of threads in each CUDA Cooperative Group
*
* @return The global thread index
￼   */
template <index_type CGSize = 1>
__device__ static constexpr index_type global_thread_id() noexcept
{
  return (index_type{threadIdx.x} + index_type{blockDim.x} * index_type{blockIdx.x}) / CGSize;
}

/**
￼   * @brief Returns the grid stride of a 1D grid
*
* @tparam CGSize Number of threads in each CUDA Cooperative Group
*
* @return The grid stride
￼   */
template <index_type CGSize = 1>
__device__ static constexpr index_type grid_stride() noexcept
{
  return index_type{gridDim.x} * index_type{blockDim.x} / CGSize;
}

}  // namespace detail
}  // namespace cuco
