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

#include <cuda/ptx>
#include <cuda/std/iterator>

#include <cooperative_groups.h>

#include <cstdint>
#include <nv/target>

namespace cuco::detail::bloom_filter_ns {

CUCO_SUPPRESS_KERNEL_WARNINGS
template <bool ConditionalAtomic, int32_t CGSize, int32_t BlockSize, class InputIt, class Ref>
__device__ void add_n_impl(InputIt first, cuco::detail::index_type n, Ref ref)
{
  namespace cg   = cooperative_groups;
  using key_type = typename cuda::std::iterator_traits<InputIt>::value_type;

  // Only use warp-cooperative kernels when CGSize > 1
  if constexpr (Ref::tuning::use_warp_cooperative_add_kernel && CGSize > 1) {
    auto const idx          = cuco::detail::global_thread_id();
    auto group              = cg::tiled_partition<CGSize>(cg::this_thread_block());
    auto const is_full_tile = (blockIdx.x + 1) * BlockSize <= n;
    if (is_full_tile) {
      key_type const& key = *(first + idx);
      ref.add_coop<ConditionalAtomic>(group, key);
    } else {
      auto const is_valid = idx < n;
      key_type const& key = is_valid ? *(first + idx) : key_type{};
      ref.add_coop<ConditionalAtomic>(group, key, is_valid);
    }
  } else {
    auto const idx = cuco::detail::global_thread_id() / CGSize;
    if constexpr (CGSize == 1) {
      if (idx < n) {
        key_type const& key = *(first + idx);
        ref.add<ConditionalAtomic>(key);
      }
    } else {
      auto group = cg::tiled_partition<CGSize>(cg::this_thread_block());
      if (idx < n) {
        key_type const& key = *(first + idx);
        ref.add<ConditionalAtomic>(group, key);
      }
    }
  }
}

template <bool ConditionalAtomic, int32_t CGSize, int32_t BlockSize, class InputIt, class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void add_n(InputIt first,
                                                    cuco::detail::index_type n,
                                                    Ref ref)
{
  add_n_impl<ConditionalAtomic, CGSize, BlockSize>(first, n, ref);
}

template <bool ConditionalAtomic, int32_t CGSize, int32_t BlockSize, class InputIt, class Ref>
__device__ void add_work_stealing_n_impl(InputIt first, cuco::detail::index_type n, Ref ref)
{
  using key_type = typename cuda::std::iterator_traits<InputIt>::value_type;

  namespace cg  = cooperative_groups;
  namespace ptx = cuda::ptx;

  // Cluster launch control initialization:
  __shared__ uint4 result;
  __shared__ uint64_t bar;
  int phase = 0;

  auto const block = cg::this_thread_block();

  cg::invoke_one(block, [&]() { ptx::mbarrier_init(&bar, 1); });

  int bx = blockIdx.x;

  // Work-stealing loop:
  while (true) {
    // Protect result from overwrite in the next iteration,
    // (also ensure barrier initialization at 1st iteration):
    block.sync();

    cg::invoke_one(block, [&]() {
      // Acquire write of result in the async proxy:
      ptx::fence_proxy_async_generic_sync_restrict(
        ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

      cg::invoke_one(cg::coalesced_threads(),
                     [&]() { ptx::clusterlaunchcontrol_try_cancel(&result, &bar); });
      ptx::mbarrier_arrive_expect_tx(
        ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
    });

    // Computation:
    // Only use warp-cooperative kernels when CGSize > 1
    if constexpr (Ref::tuning::use_warp_cooperative_add_kernel && CGSize > 1) {
      cuco::detail::index_type const idx = BlockSize * bx + threadIdx.x;
      auto group                         = cg::tiled_partition<CGSize>(block);
      auto const is_full_tile            = (bx + 1) * BlockSize <= n;
      if (is_full_tile) {
        key_type const& key = *(first + idx);
        ref.add_coop<ConditionalAtomic>(group, key);
      } else {
        auto const is_valid = idx < n;
        key_type const& key = is_valid ? *(first + idx) : key_type{};
        ref.add_coop<ConditionalAtomic>(group, key, is_valid);
      }
    } else {
      cuco::detail::index_type const idx =
        (static_cast<cuco::detail::index_type>(BlockSize) * bx + threadIdx.x) / CGSize;
      if constexpr (CGSize == 1) {
        if (idx < n) {
          key_type const& key = *(first + idx);
          ref.add<ConditionalAtomic>(key);
        }
      } else {
        auto group = cg::tiled_partition<CGSize>(block);
        if (idx < n) {
          key_type const& key = *(first + idx);
          ref.add<ConditionalAtomic>(group, key);
        }
      }
    }

    // Cancellation request synchronization:
    while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase)) {}
    phase ^= 1;

    // Cancellation request decoding:
    bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
    if (!success) break;

    bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);

    // Release read of result to the async proxy:
    ptx::fence_proxy_async_generic_sync_restrict(
      ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
  }
}

template <bool ConditionalAtomic, int32_t CGSize, int32_t BlockSize, class InputIt, class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void add_work_stealing_n(InputIt first,
                                                                  cuco::detail::index_type n,
                                                                  Ref ref)
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_100,
    (add_work_stealing_n_impl<ConditionalAtomic, CGSize, BlockSize>(first, n, ref);),
    (add_n_impl<ConditionalAtomic, CGSize, BlockSize>(first, n, ref);))
}

template <int32_t CGSize, int32_t BlockSize, class InputIt, class OutputIt, class Ref>
__device__ void contains_n_impl(InputIt first,
                                cuco::detail::index_type n,
                                OutputIt output_begin,
                                Ref ref)
{
  namespace cg   = cooperative_groups;
  using key_type = typename cuda::std::iterator_traits<InputIt>::value_type;

  // Only use warp-cooperative kernels when CGSize > 1
  if constexpr (Ref::tuning::use_warp_cooperative_contains_kernel && CGSize > 1) {
    auto const idx          = cuco::detail::global_thread_id();
    auto group              = cg::tiled_partition<CGSize>(cg::this_thread_block());
    auto const is_full_tile = (blockIdx.x + 1) * BlockSize <= n;
    if (is_full_tile) {
      key_type const& key   = *(first + idx);
      *(output_begin + idx) = ref.contains_coop(group, key);
    } else {
      auto const is_valid = idx < n;
      key_type const& key = is_valid ? *(first + idx) : key_type{};
      auto const result   = ref.contains_coop(group, key, is_valid);
      if (is_valid) { *(output_begin + idx) = result; }
    }
  } else {
    auto idx = cuco::detail::global_thread_id() / CGSize;
    if constexpr (CGSize == 1) {
      if (idx < n) {
        key_type const& key   = *(first + idx);
        *(output_begin + idx) = ref.contains(key);
      }
    } else {
      auto group = cg::tiled_partition<CGSize>(cg::this_thread_block());
      if (idx < n) {
        key_type const& key = *(first + idx);
        // ref.contains(group, key) already reduces across the group via group.all(...).
        auto const found = ref.contains(group, key);
        if (group.thread_rank() == 0) { *(output_begin + idx) = found; }
      }
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

template <int32_t CGSize, int32_t BlockSize, class InputIt, class OutputIt, class Ref>
__device__ void contains_work_stealing_n_impl(InputIt first,
                                              cuco::detail::index_type n,
                                              OutputIt output_begin,
                                              Ref ref)
{
  using key_type = typename cuda::std::iterator_traits<InputIt>::value_type;

  namespace cg  = cooperative_groups;
  namespace ptx = cuda::ptx;

  // Cluster launch control initialization:
  __shared__ uint4 result;
  __shared__ uint64_t bar;
  int phase = 0;

  auto const block = cg::this_thread_block();

  cg::invoke_one(block, [&]() { ptx::mbarrier_init(&bar, 1); });

  int bx = blockIdx.x;

  // Work-stealing loop:
  while (true) {
    // Protect result from overwrite in the next iteration,
    // (also ensure barrier initialization at 1st iteration):
    block.sync();

    cg::invoke_one(block, [&]() {
      // Acquire write of result in the async proxy:
      ptx::fence_proxy_async_generic_sync_restrict(
        ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

      cg::invoke_one(cg::coalesced_threads(),
                     [&]() { ptx::clusterlaunchcontrol_try_cancel(&result, &bar); });
      ptx::mbarrier_arrive_expect_tx(
        ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
    });

    // Computation:
    // Only use warp-cooperative kernels when CGSize > 1
    if constexpr (Ref::tuning::use_warp_cooperative_contains_kernel && CGSize > 1) {
      cuco::detail::index_type const idx = BlockSize * bx + threadIdx.x;
      auto group                         = cg::tiled_partition<CGSize>(block);
      auto const is_full_tile            = (bx + 1) * BlockSize <= n;
      if (is_full_tile) {
        key_type const& key   = *(first + idx);
        *(output_begin + idx) = ref.contains_coop(group, key);
      } else {
        auto const is_valid = idx < n;
        key_type const& key = is_valid ? *(first + idx) : key_type{};
        auto const result   = ref.contains_coop(group, key, is_valid);
        if (is_valid) { *(output_begin + idx) = result; }
      }
    } else {
      cuco::detail::index_type const idx =
        (static_cast<cuco::detail::index_type>(BlockSize) * bx + threadIdx.x) / CGSize;
      if constexpr (CGSize == 1) {
        if (idx < n) {
          key_type const& key   = *(first + idx);
          *(output_begin + idx) = ref.contains(key);
        }
      } else {
        auto group = cg::tiled_partition<CGSize>(block);
        if (idx < n) {
          key_type const& key = *(first + idx);
          // ref.contains(group, key) already reduces across the group via group.all(...).
          auto const found = ref.contains(group, key);
          if (group.thread_rank() == 0) { *(output_begin + idx) = found; }
        }
      }
    }

    // Cancellation request synchronization:
    while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase)) {}
    phase ^= 1;

    // Cancellation request decoding:
    bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
    if (!success) break;

    bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);

    // Release read of result to the async proxy:
    ptx::fence_proxy_async_generic_sync_restrict(
      ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
  }
}

template <int32_t CGSize, int32_t BlockSize, class InputIt, class OutputIt, class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void contains_work_stealing_n(InputIt first,
                                                                       cuco::detail::index_type n,
                                                                       OutputIt output_begin,
                                                                       Ref ref)
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_100,
    (contains_work_stealing_n_impl<CGSize, BlockSize>(first, n, output_begin, ref);),
    (contains_n_impl<CGSize, BlockSize>(first, n, output_begin, ref);))
}

template <bool ConditionalAtomic,
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

  if constexpr (Ref::tuning::use_warp_cooperative_add_kernel && CGSize > 1) {
    auto const idx      = cuco::detail::global_thread_id();
    auto group          = cg::tiled_partition<CGSize>(cg::this_thread_block());
    auto const in_range = idx < n;
    bool is_valid       = false;
    key_type key{};
    if (in_range) {
      key      = *(first + idx);
      is_valid = pred(*(stencil + idx));
    }
    ref.template add_coop<ConditionalAtomic>(group, key, is_valid);
  } else {
    auto const idx = cuco::detail::global_thread_id() / CGSize;
    if (idx < n && pred(*(stencil + idx))) {
      key_type const& key = *(first + idx);
      if constexpr (CGSize == 1) {
        ref.template add<ConditionalAtomic>(key);
      } else {
        auto group = cg::tiled_partition<CGSize>(cg::this_thread_block());
        ref.template add<ConditionalAtomic>(group, key);
      }
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

  if constexpr (Ref::tuning::use_warp_cooperative_contains_kernel && CGSize > 1) {
    auto const idx      = cuco::detail::global_thread_id();
    auto group          = cg::tiled_partition<CGSize>(cg::this_thread_block());
    auto const in_range = idx < n;
    bool is_valid       = false;
    key_type key{};
    if (in_range) {
      key      = *(first + idx);
      is_valid = pred(*(stencil + idx));
    }
    auto const result = ref.contains_coop(group, key, is_valid);
    if (in_range) { *(output_begin + idx) = is_valid ? result : false; }
  } else {
    auto const idx = cuco::detail::global_thread_id() / CGSize;
    if (idx < n) {
      if constexpr (CGSize == 1) {
        if (pred(*(stencil + idx))) {
          key_type const& key   = *(first + idx);
          *(output_begin + idx) = ref.contains(key);
        } else {
          *(output_begin + idx) = false;
        }
      } else {
        auto group  = cg::tiled_partition<CGSize>(cg::this_thread_block());
        bool result = false;
        if (pred(*(stencil + idx))) {
          key_type const& key = *(first + idx);
          // ref.contains(group, key) already reduces across the group via group.all(...).
          result = ref.contains(group, key);
        }
        if (group.thread_rank() == 0) { *(output_begin + idx) = result; }
      }
    }
  }
}

}  // namespace cuco::detail::bloom_filter_ns
