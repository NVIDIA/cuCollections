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
 * limitations under the License.
 */
#pragma once

#include <cuco/detail/bitwise_compare.cuh>
#include <cuco/detail/utils.hpp>

#include <cub/block/block_reduce.cuh>

#include <cuda/atomic>

namespace cuco {
namespace experimental {
namespace detail {
/**
 * @brief Calculates the number of filled slots for the given window storage.
 *
 * @tparam BlockSize Number of threads in each block
 * @tparam StorageRef Type of non-owning ref allowing access to storage
 * @tparam AtomicT Atomic counter type
 *
 * @param storage Non-owning device ref used to access the slot storage
 * @param empty_sentinel Sentinel indicating empty slots
 * @param count Number of filled slots
 */
template <int32_t BlockSize, typename StorageRef, typename AtomicT>
__global__ void size(StorageRef storage,
                     typename StorageRef::value_type empty_sentinel,
                     AtomicT* count)
{
  using size_type = typename StorageRef::size_type;

  cuco::detail::index_type const loop_stride = gridDim.x * BlockSize;
  cuco::detail::index_type idx               = BlockSize * blockIdx.x + threadIdx.x;

  size_type thread_count = 0;
  auto const n           = storage.num_windows();

  while (idx < n) {
    auto const window = storage[idx];
#pragma unroll
    for (auto const& it : window) {
      thread_count += static_cast<size_type>(not cuco::detail::bitwise_compare(it, empty_sentinel));
    }
    idx += loop_stride;
  }

  using BlockReduce = cub::BlockReduce<size_type, BlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  size_type const block_count = BlockReduce(temp_storage).Sum(thread_count);
  if (threadIdx.x == 0) { count->fetch_add(block_count, cuda::std::memory_order_relaxed); }
}

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
