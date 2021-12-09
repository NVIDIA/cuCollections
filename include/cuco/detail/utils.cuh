/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

namespace cuco {
namespace detail {

/**
 * @brief For the `n` least significant bits in the given unsigned 32-bit integer `x`,
 * returns the number of set bits.
 */
__device__ __forceinline__ int32_t count_least_significant_bits(uint32_t x, int32_t n)
{
  return __popc(x & (1 << n) - 1);
}

}  // namespace detail
}  // namespace cuco
