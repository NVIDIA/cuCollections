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

namespace cuco {
namespace detail {

/**
 * @brief Initializes each slot in the flat `slots` storage to contain `k` and `v`.
 *
 * Each space in `slots` that can hold a key value pair is initialized to a
 * `pair_atomic_type` containing the key `k` and the value `v`.
 *
 * @tparam Key key type
 * @tparam Value value type
 * @tparam SlotT slot type
 *
 * @param slots Pointer to flat storage for the map's key/value pairs
 * @param k Key to which all keys in `slots` are initialized
 * @param v Value to which all values in `slots` are initialized
 * @param size Size of the storage pointed to by `slots`
 */
template <typename Key, typename Value, typename SlotT>
__global__ void initialize(SlotT* const slots, Key k, Value v, std::size_t size)
{
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  while (tid < size) {
    new (&slots[tid].first) Key{k};
    new (&slots[tid].second) Value{v};
    tid += gridDim.x * blockDim.x;
  }
}

}  // namespace detail
}  // namespace cuco
