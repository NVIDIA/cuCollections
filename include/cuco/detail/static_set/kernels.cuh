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
  std::size_t thread_num_successes = 0;

  auto tid = BlockSize * blockIdx.x + threadIdx.x;
  auto it  = first + tid;

  while (it < last) {
    typename Reference::value_type const insert_pair{*it};
    if (reference.insert(insert_pair)) { thread_num_successes++; }
    it += gridDim.x * BlockSize;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  using block_reduce = cub::BlockReduce<std::size_t, BlockSize>;
  __shared__ typename block_reduce::TempStorage temp_storage;
  std::size_t block_num_successes = block_reduce(temp_storage).Sum(thread_num_successes);
  if (threadIdx.x == 0) {
    num_successes->fetch_add(block_num_successes, cuda::std::memory_order_relaxed);
  }
}
}  // namespace detail
}  // namespace experimental
}  // namespace cuco
