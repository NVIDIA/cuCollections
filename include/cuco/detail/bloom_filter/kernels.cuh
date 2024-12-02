/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuco/detail/utility/cuda.cuh>

#include <cooperative_groups.h>

#include <cstdint>
#include <iterator>

namespace cuco::detail::bloom_filter_ns {

CUCO_SUPPRESS_KERNEL_WARNINGS

template <int32_t CGSize,
          int32_t BlockSize,
          class InputIt,
          class StencilIt,
          class Predicate,
          class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void add_if_n(
  InputIt first, cuco::detail::index_type n, StencilIt stencil, Predicate pred, Ref ref)
{
  namespace cg = cooperative_groups;

  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  [[maybe_unused]] auto const tile = cg::tiled_partition<CGSize>(cg::this_thread_block());

  while (idx < n) {
    if (pred(*(stencil + idx))) {
      typename std::iterator_traits<InputIt>::value_type const& insert_element{*(first + idx)};
      ref.add(tile, insert_element);
    }
    idx += loop_stride;
  }
}

template <int32_t CGSize,
          int32_t BlockSize,
          class InputIt,
          class StencilIt,
          class Predicate,
          class OutputIt,
          class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void contains_if_n(InputIt first,
                                                            cuco::detail::index_type n,
                                                            StencilIt stencil,
                                                            Predicate pred,
                                                            OutputIt out,
                                                            Ref ref)
{
  namespace cg = cooperative_groups;

  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  [[maybe_unused]] auto const tile = cg::tiled_partition<CGSize>(cg::this_thread_block());

  if constexpr (CGSize == 1) {
    while (idx < n) {
      typename std::iterator_traits<InputIt>::value_type const& key = *(first + idx);
      *(out + idx) = pred(*(stencil + idx)) ? ref.contains(key) : false;
      idx += loop_stride;
    }
  } else {
    auto const tile = cg::tiled_partition<CGSize>(cg::this_thread_block());
    while (idx < n) {
      typename std::iterator_traits<InputIt>::value_type const& key = *(first + idx);
      auto const found = pred(*(stencil + idx)) ? ref.contains(tile, key) : false;
      if (tile.thread_rank() == 0) { *(out + idx) = found; }
      idx += loop_stride;
    }
  }
}

}  // namespace cuco::detail::bloom_filter_ns
