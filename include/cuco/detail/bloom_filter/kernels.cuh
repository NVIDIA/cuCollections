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

namespace cuco::detail {

CUCO_SUPPRESS_KERNEL_WARNINGS

template <int32_t BlockSize, class InputIt, class StencilIt, class Predicate, class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void add_if_n(
  InputIt first, cuco::detail::index_type n, StencilIt stencil, Predicate pred, Ref ref)
{
  auto constexpr block_words = Ref::block_words;

  auto const loop_stride = cuco::detail::grid_stride() / block_words;
  auto idx               = cuco::detail::global_thread_id() / block_words;

  [[maybe_unused]] auto const tile =
    cooperative_groups::tiled_partition<block_words>(cooperative_groups::this_thread_block());

  while (idx < n) {
    if (pred(*(stencil + idx))) {
      typename std::iterator_traits<InputIt>::value_type const& insert_element{*(first + idx)};
      if constexpr (block_words == 1) {
        ref.add(insert_element);
      } else {
        ref.add(tile, insert_element);
      }
    }
    idx += loop_stride;
  }
}

template <int32_t BlockSize,
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
  auto const loop_stride = cuco::detail::grid_stride();
  auto idx               = cuco::detail::global_thread_id();

  while (idx < n) {
    if (pred(*(stencil + idx))) {
      typename std::iterator_traits<InputIt>::value_type const& query{*(first + idx)};
      *(out + idx) = ref.contains(query);
    } else {
      *(out + idx) = false;
    }
    idx += loop_stride;
  }
}

}  // namespace cuco::detail