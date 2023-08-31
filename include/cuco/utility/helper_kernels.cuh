/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cuco/aow_storage.cuh>
#include <cuco/detail/utils.hpp>

#include <cstddef>

namespace cuco {
namespace experimental {

/**
 * @brief Initializes each slot in the window storage to contain `value`.
 *
 * @tparam T Window slot type
 * @tparam WindowSize Number of slots per window
 *
 * @param windows Pointer to flat storage for windows
 * @param n Number of input windows
 * @param value Value to which all values in `slots` are initialized
 */
template <typename T, int32_t WindowSize>
__global__ void initialize_windows(cuco::experimental::window<T, WindowSize>* windows,
                                   cuco::detail::index_type n,
                                   T value)
{
  cuco::detail::index_type const loop_stride = gridDim.x * blockDim.x;
  cuco::detail::index_type idx               = blockDim.x * blockIdx.x + threadIdx.x;

  while (idx < n) {
    auto& window_slots = *(windows + idx);
#pragma unroll(WindowSize)
    for (auto& slot : window_slots) {
      slot = value;
    }
    idx += loop_stride;
  }
}

}  // namespace experimental
}  // namespace cuco
