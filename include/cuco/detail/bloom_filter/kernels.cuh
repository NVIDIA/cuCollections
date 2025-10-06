/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include <cuco/detail/utility/math.cuh>

#include <cuda/std/iterator>

#include <cooperative_groups.h>

#include <cstdint>

namespace cuco::detail::bloom_filter_ns {

CUCO_SUPPRESS_KERNEL_WARNINGS

template <int32_t BlockSize, class InputIt, class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void add(InputIt first,
                                                  cuco::detail::index_type n,
                                                  Ref ref)
{
  namespace cg = cooperative_groups;

  constexpr auto tile_size = cuco::detail::warp_size();

  auto const tile_idx       = cuco::detail::global_thread_id() / tile_size;
  auto const n_tiles        = gridDim.x * BlockSize / tile_size;
  auto const items_per_tile = cuco::detail::int_div_ceil(n, n_tiles);

  auto const tile_start = tile_idx * items_per_tile;
  if (tile_start >= n) { return; }
  auto const tile_stop = (tile_start + items_per_tile < n) ? tile_start + items_per_tile : n;

  auto const tile = cg::tiled_partition<tile_size, cg::thread_block>(cg::this_thread_block());

  ref.add(tile, first + tile_start, first + tile_stop);
}

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

  [[maybe_unused]] auto const tile =
    cg::tiled_partition<CGSize, cg::thread_block>(cg::this_thread_block());

  while (idx < n) {
    if (pred(*(stencil + idx))) {
      typename cuda::std::iterator_traits<InputIt>::value_type const& insert_element{
        *(first + idx)};
      ref.add(tile, insert_element);
    }
    idx += loop_stride;
  }
}

template <int32_t BlockSize, class InputIt, class OutputIt, class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void contains(InputIt first,
                                                       cuco::detail::index_type n,
                                                       OutputIt output_begin,
                                                       Ref ref)
{
  namespace cg = cooperative_groups;

  constexpr auto tile_size = cuco::detail::warp_size();

  auto const tile_idx       = cuco::detail::global_thread_id() / tile_size;
  auto const n_tiles        = gridDim.x * BlockSize / tile_size;
  auto const items_per_tile = cuco::detail::int_div_ceil(n, n_tiles);

  auto const tile_start = tile_idx * items_per_tile;
  if (tile_start >= n) { return; }
  auto const tile_stop = (tile_start + items_per_tile < n) ? tile_start + items_per_tile : n;

  auto const tile = cg::tiled_partition<tile_size, cg::thread_block>(cg::this_thread_block());

  ref.contains(tile, first + tile_start, first + tile_stop, output_begin + tile_start);
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

  [[maybe_unused]] auto const tile =
    cg::tiled_partition<CGSize, cg::thread_block>(cg::this_thread_block());

  if constexpr (CGSize == 1) {
    while (idx < n) {
      typename cuda::std::iterator_traits<InputIt>::value_type const& key = *(first + idx);
      *(out + idx) = pred(*(stencil + idx)) ? ref.contains(key) : false;
      idx += loop_stride;
    }
  } else {
    auto const tile = cg::tiled_partition<CGSize, cg::thread_block>(cg::this_thread_block());
    while (idx < n) {
      typename cuda::std::iterator_traits<InputIt>::value_type const& key = *(first + idx);
      auto const found = pred(*(stencil + idx)) ? ref.contains(tile, key) : false;
      if (tile.thread_rank() == 0) { *(out + idx) = found; }
      idx += loop_stride;
    }
  }
}

//===--------------------------------------------------===//
// Parametric Filter Policy
//===--------------------------------------------------===//
template <int32_t CGSize, int32_t BlockSize, class InputIt, class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void add_exp_n(InputIt first,
                                                        cuco::detail::index_type n,
                                                        Ref ref)
{
  namespace cg   = cooperative_groups;
  using key_type = typename cuda::std::iterator_traits<InputIt>::value_type;

  // Only use warp-cooperative kernels when CGSize > 1
  if constexpr (Ref::use_warp_cooperative_add_kernel && CGSize > 1) {
    auto const idx          = cuco::detail::global_thread_id();
    auto group              = cg::tiled_partition<CGSize>(cg::this_thread_block());
    auto const is_full_tile = (blockIdx.x + 1) * BlockSize <= n;
    if (is_full_tile) {
      key_type const& key = *(first + idx);
      ref.add_exp_coop(group, key);
    } else {
      auto const is_valid = idx < n;
      key_type const& key = is_valid ? *(first + idx) : key_type{};
      ref.add_exp_coop(group, key, is_valid);
    }
  } else {
    auto const idx = cuco::detail::global_thread_id() / CGSize;
    if constexpr (CGSize == 1) {
      if (idx < n) {
        key_type const& key = *(first + idx);
        ref.add_exp(key);
      }
    } else {
      auto group = cg::tiled_partition<CGSize>(cg::this_thread_block());
      if (idx < n) {
        key_type const& key = *(first + idx);
        ref.add_exp(group, key);
      }
    }
  }
}

template <int32_t CGSize, int32_t BlockSize, class InputIt, class OutputIt, class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void contains_exp_n(InputIt first,
                                                             cuco::detail::index_type n,
                                                             OutputIt output_begin,
                                                             Ref ref)
{
  namespace cg   = cooperative_groups;
  using key_type = typename cuda::std::iterator_traits<InputIt>::value_type;

  // Only use warp-cooperative kernels when CGSize > 1
  if constexpr (Ref::use_warp_cooperative_contains_kernel && CGSize > 1) {
    auto const idx          = cuco::detail::global_thread_id();
    auto group              = cg::tiled_partition<CGSize>(cg::this_thread_block());
    auto const is_full_tile = (blockIdx.x + 1) * BlockSize <= n;
    if (is_full_tile) {
      key_type const& key   = *(first + idx);
      *(output_begin + idx) = ref.contains_exp_coop(group, key);
    } else {
      auto const is_valid = idx < n;
      key_type const& key = is_valid ? *(first + idx) : key_type{};
      auto const result   = ref.contains_exp_coop(group, key, is_valid);
      if (is_valid) { *(output_begin + idx) = result; }
    }
  } else {
    auto const loop_stride = cuco::detail::grid_stride() / CGSize;
    auto idx               = cuco::detail::global_thread_id() / CGSize;
    if constexpr (CGSize == 1) {
      while (idx < n) {
        key_type const& key   = *(first + idx);
        *(output_begin + idx) = ref.contains_exp(key);
        idx += loop_stride;
      }
    } else {
      auto group = cg::tiled_partition<CGSize>(cg::this_thread_block());
      while (idx < n) {
        key_type const& key = *(first + idx);
        auto const found    = group.all(ref.contains_exp(group, key));
        if (group.thread_rank() == 0) { *(output_begin + idx) = found; }
        idx += loop_stride;
      }
    }
  }
}

}  // namespace cuco::detail::bloom_filter_ns
