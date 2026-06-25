/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include <cuco/detail/__config>
#include <cuco/detail/utility/cuda.cuh>
#include <cuco/detail/utility/math.cuh>

#include <cuda/std/iterator>

#include <cooperative_groups.h>

#include <cstdint>

namespace cuco::detail::bloom_filter_ns {

CUCO_SUPPRESS_KERNEL_WARNINGS
template <bool ConditionalAdd, int32_t CGSize, int32_t BlockSize, class InputIt, class Ref>
__device__ void add_n_impl(InputIt first, cuco::detail::index_type n, Ref ref)
{
  namespace cg   = cooperative_groups;
  using key_type = typename cuda::std::iterator_traits<InputIt>::value_type;

  if constexpr (CGSize > 1) {
    auto const idx = cuco::detail::global_thread_id();
    auto group     = cg::tiled_partition<CGSize>(cg::this_thread_block());
    auto const is_full_tile =
      static_cast<cuco::detail::index_type>(blockIdx.x + 1) * BlockSize <= n;
    if (is_full_tile) {
      key_type const& key = *(first + idx);
      ref.add_coop<ConditionalAdd>(group, key);
    } else {
      auto const is_valid = idx < n;
      ref.add_coop<ConditionalAdd>(group, first, idx, is_valid);
    }
  } else {
    auto const idx = cuco::detail::global_thread_id();
    if (idx < n) {
      key_type const& key = *(first + idx);
      ref.add<ConditionalAdd>(key);
    }
  }
}

template <bool ConditionalAdd, int32_t CGSize, int32_t BlockSize, class InputIt, class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void add_n(InputIt first,
                                                    cuco::detail::index_type n,
                                                    Ref ref)
{
  add_n_impl<ConditionalAdd, CGSize, BlockSize>(first, n, ref);
}

template <int32_t CGSize, int32_t BlockSize, class InputIt, class OutputIt, class Ref>
__device__ void contains_n_impl(InputIt first,
                                cuco::detail::index_type n,
                                OutputIt output_begin,
                                Ref ref)
{
  namespace cg   = cooperative_groups;
  using key_type = typename cuda::std::iterator_traits<InputIt>::value_type;

  if constexpr (CGSize > 1) {
    auto const idx = cuco::detail::global_thread_id();
    auto group     = cg::tiled_partition<CGSize>(cg::this_thread_block());
    auto const is_full_tile =
      static_cast<cuco::detail::index_type>(blockIdx.x + 1) * BlockSize <= n;
    if (is_full_tile) {
      key_type const& key   = *(first + idx);
      *(output_begin + idx) = ref.contains_coop(group, key);
    } else {
      auto const is_valid = idx < n;
      auto const result   = ref.contains_coop(group, first, idx, is_valid);
      if (is_valid) { *(output_begin + idx) = result; }
    }
  } else {
    auto const idx = cuco::detail::global_thread_id();
    if (idx < n) {
      key_type const& key   = *(first + idx);
      *(output_begin + idx) = ref.contains(key);
    }
  }
}

template <int32_t CGSize, int32_t BlockSize, class InputIt, class OutputIt, class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void contains_n(InputIt first,
                                                         cuco::detail::index_type n,
                                                         OutputIt output_begin,
                                                         Ref ref)
{
  contains_n_impl<CGSize, BlockSize>(first, n, output_begin, ref);
}

template <bool ConditionalAdd,
          int32_t CGSize,
          int32_t BlockSize,
          class InputIt,
          class StencilIt,
          class Predicate,
          class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void add_if_n(
  InputIt first, cuco::detail::index_type n, StencilIt stencil, Predicate pred, Ref ref)
{
  namespace cg   = cooperative_groups;
  using key_type = typename cuda::std::iterator_traits<InputIt>::value_type;

  if constexpr (CGSize > 1) {
    auto const idx      = cuco::detail::global_thread_id();
    auto group          = cg::tiled_partition<CGSize>(cg::this_thread_block());
    auto const in_range = idx < n;
    auto const is_valid = in_range && pred(*(stencil + idx));
    ref.template add_coop<ConditionalAdd>(group, first, idx, is_valid);
  } else {
    auto const idx = cuco::detail::global_thread_id();
    if (idx < n && pred(*(stencil + idx))) {
      key_type const& key = *(first + idx);
      ref.template add<ConditionalAdd>(key);
    }
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
                                                            OutputIt output_begin,
                                                            Ref ref)
{
  namespace cg   = cooperative_groups;
  using key_type = typename cuda::std::iterator_traits<InputIt>::value_type;

  if constexpr (CGSize > 1) {
    auto const idx      = cuco::detail::global_thread_id();
    auto group          = cg::tiled_partition<CGSize>(cg::this_thread_block());
    auto const in_range = idx < n;
    auto const is_valid = in_range && pred(*(stencil + idx));
    auto const result   = ref.contains_coop(group, first, idx, is_valid);
    if (in_range) { *(output_begin + idx) = is_valid ? result : false; }
  } else {
    auto const idx = cuco::detail::global_thread_id();
    if (idx < n) {
      if (pred(*(stencil + idx))) {
        key_type const& key   = *(first + idx);
        *(output_begin + idx) = ref.contains(key);
      } else {
        *(output_begin + idx) = false;
      }
    }
  }
}

}  // namespace cuco::detail::bloom_filter_ns
