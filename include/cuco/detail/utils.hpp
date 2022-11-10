/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <iterator>

namespace cuco {
namespace detail {

/**
 * @brief Compute the number of bits of a simple type.
 *
 * @tparam T The type we want to infer its size in bits
 *
 * @return Size of type T in bits
 */
template <typename T>
static constexpr std::size_t type_bits() noexcept
{
  return sizeof(T) * CHAR_BIT;
}

// safe division
#ifndef SDIV
#define SDIV(x, y) (((x) + (y)-1) / (y))
#endif

template <typename Kernel>
auto get_grid_size(Kernel kernel, std::size_t block_size, std::size_t dynamic_smem_bytes = 0)
{
  int grid_size{-1};
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size, kernel, block_size, dynamic_smem_bytes);
  int dev_id{-1};
  cudaGetDevice(&dev_id);
  int num_sms{-1};
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
  grid_size *= num_sms;
  return grid_size;
}

template <typename Iterator>
constexpr inline int64_t distance(Iterator begin, Iterator end)
{
  // `int64_t` instead of arch-dependant `long int`
  return static_cast<int64_t>(std::distance(begin, end));
}

}  // namespace detail
}  // namespace cuco
