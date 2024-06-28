/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cuco/detail/utility/cuda.cuh>

#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/std/optional>

#include <cooperative_groups.h>

#include <iterator>

namespace cuco::static_map_ns::detail {
CUCO_SUPPRESS_KERNEL_WARNINGS

// TODO user insert_or_assign internally
/**
 * @brief For any key-value pair `{k, v}` in the range `[first, first + n)`, if a key equivalent to
 * `k` already exists in the container, assigns `v` to the mapped_type corresponding to the key `k`.
 * If the key does not exist, inserts the pair as if by insert.
 *
 * @note If multiple elements in `[first, first + n)` compare equal, it is unspecified which element
 * is inserted.
 *
 * @tparam CGSize Number of threads in each CG
 * @tparam BlockSize Number of threads in each block
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the `value_type` of the data structure
 * @tparam Ref Type of non-owning device ref allowing access to storage
 *
 * @param first Beginning of the sequence of input elements
 * @param n Number of input elements
 * @param ref Non-owning container device ref used to access the slot storage
 */
template <int32_t CGSize, int32_t BlockSize, typename InputIt, typename Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void insert_or_assign(InputIt first,
                                                               cuco::detail::index_type n,
                                                               Ref ref)
{
  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  while (idx < n) {
    typename std::iterator_traits<InputIt>::value_type const& insert_pair = *(first + idx);
    if constexpr (CGSize == 1) {
      ref.insert_or_assign(insert_pair);
    } else {
      auto const tile =
        cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
      ref.insert_or_assign(tile, insert_pair);
    }
    idx += loop_stride;
  }
}

/**
 * @brief For any key-value pair `{k, v}` in the range `[first, first + n)`, if a key equivalent to
 * `k` already exists in the container, then binary operation is applied using `op` callable object
 * on the existing value at slot and the element to insert. If the key does not exist, inserts the
 * pair as if by insert.
 *
 * @note Callable object to perform binary operation should be able to invoke as Op(cuda::atomic<T,
 * Scope>, T>)
 *
 * @tparam CGSize Number of threads in each CG
 * @tparam BlockSize Number of threads in each block
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the `value_type` of the data structure
 * @tparam Op Callable type used to peform apply operation.
 * @tparam T Type of optional idenitity element which is convertible
 * to `value_type` of the data structure
 * @tparam Ref Type of non-owning device ref allowing access to storage
 *
 * @param first Beginning of the sequence of input elements
 * @param n Number of input elements
 * @param op callable object to perform apply operation.
 * @param identity_element An optional Identity element of the binary operation
 * @param ref Non-owning container device ref used to access the slot storage
 */
template <int32_t CGSize,
          int32_t BlockSize,
          typename InputIt,
          typename Op,
          typename T,
          typename Ref>
__global__ void insert_or_apply(InputIt first,
                                cuco::detail::index_type n,
                                Op op,
                                cuda::std::optional<T> identity_element,
                                Ref ref)
{
  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  while (idx < n) {
    typename std::iterator_traits<InputIt>::value_type const& insert_pair = *(first + idx);
    if constexpr (CGSize == 1) {
      ref.insert_or_apply(insert_pair, op, identity_element);
    } else {
      auto const tile =
        cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
      ref.insert_or_apply(tile, insert_pair, op, identity_element);
    }
    idx += loop_stride;
  }
}

}  // namespace cuco::static_map_ns::detail