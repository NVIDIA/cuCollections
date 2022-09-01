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
 * @tparam AtomicCounter Type of atomic counter
 * @tparam Reference Type of device reference allowing access of set storage
 *
 * @param first Beginning of the sequence of key/value pairs
 * @param last End of the sequence of key/value pairs
 * @param num_successes The number of successfully inserted key/value pairs
 * @param reference Mutable device reference used to access the set's slot storage
 */
template <int BlockSize, typename InputIterator, typename AtomicCounter, typename Reference>
__global__ void insert(InputIterator first,
                       InputIterator last,
                       AtomicCounter* num_successes,
                       Reference reference)
{
  using size_type                = typename Reference::size_type;
  size_type thread_num_successes = 0;

  auto tid = BlockSize * blockIdx.x + threadIdx.x;
  auto it  = first + tid;

  while (it < last) {
    typename Reference::value_type const insert_pair{*it};
    if (reference.insert(insert_pair)) { thread_num_successes++; }
    it += gridDim.x * BlockSize;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  using block_reduce = cub::BlockReduce<size_type, BlockSize>;
  __shared__ typename block_reduce::TempStorage temp_storage;
  size_type block_num_successes = block_reduce(temp_storage).Sum(thread_num_successes);
  if (threadIdx.x == 0) {
    num_successes->fetch_add(block_num_successes, cuda::std::memory_order_relaxed);
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
 * @param reference Mutable device reference used to access the set's slot storage
 */
template <int BlockSize, typename InputIt, typename OutputIt, typename Reference>
__global__ void contains(InputIt first, InputIt last, OutputIt output_begin, Reference reference)
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
      output_buffer[thread_idx] = reference.contains(key);
    }

    block.sync();
    if (first + key_idx < last) { *(output_begin + key_idx) = output_buffer[thread_idx]; }
    key_idx += gridDim.x * BlockSize;
  }
}
}  // namespace detail
}  // namespace experimental
}  // namespace cuco
