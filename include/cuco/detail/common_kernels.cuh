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

#include <cooperative_groups.h>

namespace cuco {
namespace experimental {
namespace detail {

/**
 * @brief Inserts all elements in the range `[first, first + n)` and returns the number of
 * successful insertions if `pred` of the corresponding stencil returns true.
 *
 * @note If multiple elements in `[first, first + n)` compare equal, it is unspecified which element
 * is inserted.
 * @note The key `*(first + i)` is inserted if `pred( *(stencil + i) )` returns true.
 *
 * @tparam CGSize Number of threads in each CG
 * @tparam BlockSize Number of threads in each block
 * @tparam InputIterator Device accessible input iterator whose `value_type` is
 * convertible to the `value_type` of the data structure
 * @tparam StencilIt Device accessible random access iterator whose value_type is
 * convertible to Predicate's argument type
 * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool`
 * and argument type is convertible from `std::iterator_traits<StencilIt>::value_type`
 * @tparam AtomicT Atomic counter type
 * @tparam Ref Type of non-owning device container ref allowing access to storage
 *
 * @param first Beginning of the sequence of input elements
 * @param n Number of input elements
 * @param stencil Beginning of the stencil sequence
 * @param pred Predicate to test on every element in the range `[stencil, stencil + n)`
 * @param num_successes Number of successful inserted elements
 * @param ref Non-owning set device ref used to access the slot storage
 */
template <int32_t CGSize,
          int32_t BlockSize,
          typename InputIterator,
          typename StencilIt,
          typename Predicate,
          typename AtomicT,
          typename Ref>
__global__ void insert_if_n(InputIterator first,
                            cuco::detail::index_type n,
                            StencilIt stencil,
                            Predicate pred,
                            AtomicT* num_successes,
                            Ref ref)
{
  using BlockReduce = cub::BlockReduce<typename Ref::size_type, BlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  typename Ref::size_type thread_num_successes = 0;

  cuco::detail::index_type const loop_stride = gridDim.x * BlockSize / CGSize;
  cuco::detail::index_type idx               = (BlockSize * blockIdx.x + threadIdx.x) / CGSize;

  while (idx < n) {
    if (pred(*(stencil + idx))) {
      typename Ref::value_type const insert_pair{*(first + idx)};
      if constexpr (CGSize == 1) {
        if (ref.insert(insert_pair)) { thread_num_successes++; };
      } else {
        auto const tile =
          cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
        if (ref.insert(tile, insert_pair) && tile.thread_rank() == 0) { thread_num_successes++; };
      }
    }
    idx += loop_stride;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  typename Ref::size_type block_num_successes = BlockReduce(temp_storage).Sum(thread_num_successes);
  if (threadIdx.x == 0) {
    num_successes->fetch_add(block_num_successes, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Inserts all elements in the range `[first, first + n)` if `pred` of the corresponding
 * stencil returns true.
 *
 * @note If multiple elements in `[first, first + n)` compare equal, it is unspecified which element
 * is inserted.
 * @note The key `*(first + i)` is inserted if `pred( *(stencil + i) )` returns true.
 *
 * @tparam CGSize Number of threads in each CG
 * @tparam BlockSize Number of threads in each block
 * @tparam InputIterator Device accessible input iterator whose `value_type` is
 * convertible to the `value_type` of the data structure
 * @tparam StencilIt Device accessible random access iterator whose value_type is
 * convertible to Predicate's argument type
 * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool`
 * and argument type is convertible from `std::iterator_traits<StencilIt>::value_type`
 * @tparam Ref Type of non-owning device ref allowing access to storage
 *
 * @param first Beginning of the sequence of input elements
 * @param n Number of input elements
 * @param stencil Beginning of the stencil sequence
 * @param pred Predicate to test on every element in the range `[stencil, stencil + n)`
 * @param ref Non-owning set device ref used to access the slot storage
 */
template <int32_t CGSize,
          int32_t BlockSize,
          typename InputIterator,
          typename StencilIt,
          typename Predicate,
          typename Ref>
__global__ void insert_if_n(
  InputIterator first, cuco::detail::index_type n, StencilIt stencil, Predicate pred, Ref ref)
{
  cuco::detail::index_type const loop_stride = gridDim.x * BlockSize / CGSize;
  cuco::detail::index_type idx               = (BlockSize * blockIdx.x + threadIdx.x) / CGSize;

  while (idx < n) {
    if (pred(*(stencil + idx))) {
      typename Ref::value_type const insert_pair{*(first + idx)};
      if constexpr (CGSize == 1) {
        ref.insert(insert_pair);
      } else {
        auto const tile =
          cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
        ref.insert(tile, insert_pair);
      }
    }
    idx += loop_stride;
  }
}

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
