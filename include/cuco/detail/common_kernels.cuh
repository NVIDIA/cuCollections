/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cstddef>

namespace cuco {
namespace experimental {
namespace detail {

/**
 * @brief Initializes each slot in the flat `slots` storage to contain `k`.
 *
 * @tparam WindowT Window type
 * @tparam Key Key type
 *
 * @param slots Pointer to flat storage for the keys
 * @param k Key to which all keys in `slots` are initialized
 * @param size Size of the storage pointed to by `slots`
 */
template <typename WindowT, typename Key>
__global__ void initialize(WindowT* windows, Key k, std::size_t size)
{
  /*
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  while (tid < size) {
    slots[tid] = k;
    tid += gridDim.x * blockDim.x;
  }
  */
}

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
