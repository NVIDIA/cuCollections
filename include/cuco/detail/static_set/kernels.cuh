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

#include <cub/block/block_reduce.cuh>

#include <cuda/atomic>

#include <cooperative_groups.h>

namespace cuco {
namespace experimental {
namespace detail {

/**
 * @brief Inserts all keys in the range `[first, last)`.
 *
 * If multiple keys in `[first, last)` compare equal, it is unspecified which
 * element is inserted.
 *
 * @tparam BlockSize Number of threads in each block
 * @tparam InputIterator Device accessible input iterator whose `value_type` is
 * convertible to the set's `value_type`
 * @tparam Reference Type of device reference allowing access of set storage
 *
 * @param first Beginning of the sequence of key/value pairs
 * @param last End of the sequence of key/value pairs
 * @param set_ref Set device reference used to access the set's slot storage
 */
template <int BlockSize, typename InputIterator, typename Reference>
__global__ void insert(InputIterator first, InputIterator last, Reference set_ref)
{
  auto tid = BlockSize * blockIdx.x + threadIdx.x;
  auto it  = first + tid;

  while (it < last) {
    typename Reference::value_type const insert_pair{*it};
    set_ref.insert(insert_pair);
    it += gridDim.x * BlockSize;
  }
}

/**
 * @brief Inserts all keys in the range `[first, last)`.
 *
 * If multiple keys in `[first, last)` compare equal, it is unspecified which
 * element is inserted.
 *
 * @tparam CGSize Number of threads in each CG
 * @tparam BlockSize Number of threads in each block
 * @tparam InputIterator Device accessible input iterator whose `value_type` is
 * convertible to the set's `value_type`
 * @tparam Reference Type of device reference allowing access of set storage
 *
 * @param first Beginning of the sequence of key/value pairs
 * @param last End of the sequence of key/value pairs
 * @param set_ref Set device reference used to access the set's slot storage
 */
template <int CGSize, int BlockSize, typename InputIterator, typename Reference>
__global__ void insert(InputIterator first, InputIterator last, Reference set_ref)
{
  namespace cg = cooperative_groups;

  auto tile = cg::tiled_partition<CGSize>(cg::this_thread_block());
  auto tid  = BlockSize * blockIdx.x + threadIdx.x;
  auto it   = first + tid / CGSize;

  while (it < last) {
    typename Reference::value_type const insert_pair{*it};
    set_ref.insert(tile, insert_pair);
    it += (gridDim.x * BlockSize) / CGSize;
  }
}

/**
 * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
 *
 * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
 *
 * @tparam BlockSize The size of the thread block
 * @tparam InputIt Device accessible input iterator
 * @tparam OutputIt Device accessible output iterator assignable from `bool`
 * @tparam Reference Type of device reference allowing access of set storage
 *
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param set_ref Set device reference used to access the set's slot storage
 */
template <int BlockSize, typename InputIt, typename OutputIt, typename Reference>
__global__ void contains(InputIt first, InputIt last, OutputIt output_begin, Reference set_ref)
{
  namespace cg = cooperative_groups;

  auto block            = cg::this_thread_block();
  auto const thread_idx = block.thread_rank();

  auto tid     = BlockSize * blockIdx.x + thread_idx;
  auto key_idx = tid;
  __shared__ bool output_buffer[BlockSize];

  while (first + key_idx - thread_idx <
         last) {  // the whole thread block falls into the same iteration
    if (first + key_idx < last) {
      auto key = *(first + key_idx);
      /*
       * The ld.relaxed.gpu instruction used in view.find causes L1 to
       * flush more frequently, causing increased sector stores from L2 to global memory.
       * By writing results to shared memory and then synchronizing before writing back
       * to global, we no longer rely on L1, preventing the increase in sector stores from
       * L2 to global and improving performance.
       */
      output_buffer[thread_idx] = set_ref.contains(key);
    }

    block.sync();
    if (first + key_idx < last) { *(output_begin + key_idx) = output_buffer[thread_idx]; }
    key_idx += gridDim.x * BlockSize;
  }
}

/**
 * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
 *
 * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
 *
 * @tparam CGSize Number of threads in each CG
 * @tparam BlockSize The size of the thread block
 * @tparam InputIt Device accessible input iterator
 * @tparam OutputIt Device accessible output iterator assignable from `bool`
 * @tparam Reference Type of device reference allowing access of set storage
 *
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param set_ref Set device reference used to access the set's slot storage
 */
template <int CGSize, int BlockSize, typename InputIt, typename OutputIt, typename Reference>
__global__ void contains(InputIt first, InputIt last, OutputIt output_begin, Reference set_ref)
{
  namespace cg = cooperative_groups;

  auto block            = cg::this_thread_block();
  auto const thread_idx = block.thread_rank();

  auto tile    = cg::tiled_partition<CGSize>(cg::this_thread_block());
  auto tid     = BlockSize * blockIdx.x + threadIdx.x;
  auto key_idx = tid / CGSize;

  __shared__ bool output_buffer[BlockSize / CGSize];

  while (first + key_idx - thread_idx <
         last) {  // the whole thread block falls into the same iteration
    if (first + key_idx < last) {
      auto key   = *(first + key_idx);
      auto found = set_ref.contains(tile, key);
      /*
       * The ld.relaxed.gpu instruction used in view.find causes L1 to
       * flush more frequently, causing increased sector stores from L2 to global memory.
       * By writing results to shared memory and then synchronizing before writing back
       * to global, we no longer rely on L1, preventing the increase in sector stores from
       * L2 to global and improving performance.
       */
      if (tile.thread_rank() == 0) { output_buffer[thread_idx / CGSize] = found; }
    }

    block.sync();
    if (first + key_idx < last and tile.thread_rank() == 0) {
      *(output_begin + key_idx) = output_buffer[thread_idx / CGSize];
    }
    key_idx += (gridDim.x * BlockSize) / CGSize;
  }
}
}  // namespace detail
}  // namespace experimental
}  // namespace cuco
