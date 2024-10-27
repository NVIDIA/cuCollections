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
#include <cub/block/block_scan.cuh>
#include <cuda/atomic>

#include <cooperative_groups.h>

#include <iterator>

namespace cuco::static_set_ns::detail {

CUCO_SUPPRESS_KERNEL_WARNINGS

template <typename CG,
          typename Size,
          typename ProbeKey,
          typename Key,
          typename OutputIt1,
          typename OutputIt2>
__device__ void flush_buffer(CG const& tile,
                             Size buffer_size,
                             cuco::pair<ProbeKey, Key>* buffer,
                             cuda::atomic<Size, cuda::thread_scope_device>* counter,
                             OutputIt1 output_probe,
                             OutputIt2 output_match)
{
  auto i = tile.thread_rank();

#if defined(CUCO_HAS_CG_INVOKE_ONE)
  auto const offset = cooperative_groups::invoke_one_broadcast(
    tile, [&]() { return counter->fetch_add(buffer_size, cuda::std::memory_order_relaxed); });
#else
  Size offset;
  if (i == 0) { offset = counter->fetch_add(buffer_size, cuda::std::memory_order_relaxed); }
  offset = tile.shfl(offset, 0);
#endif

  while (i < buffer_size) {
    *(output_probe + offset + i) = buffer[i].first;
    *(output_match + offset + i) = buffer[i].second;

    i += tile.size();
  }
}

template <int32_t BlockSize,
          typename InputIt,
          typename OutputIt1,
          typename OutputIt2,
          typename AtomicT,
          typename Ref>
__device__ void group_retrieve(InputIt first,
                               cuco::detail::index_type n,
                               OutputIt1 output_probe,
                               OutputIt2 output_match,
                               AtomicT* counter,
                               Ref ref)
{
  namespace cg = cooperative_groups;

  using size_type = typename Ref::size_type;
  using ProbeKey  = typename std::iterator_traits<InputIt>::value_type;
  using Key       = typename Ref::key_type;

  auto constexpr tile_size   = Ref::cg_size;
  auto constexpr bucket_size = Ref::bucket_size;

  auto idx          = cuco::detail::global_thread_id() / tile_size;
  auto const stride = cuco::detail::grid_stride() / tile_size;
  auto const block  = cg::this_thread_block();
  auto const tile   = cg::tiled_partition<tile_size>(block);

  auto constexpr flushing_tile_size = cuco::detail::warp_size() / bucket_size;
  // random choice to tune
  auto constexpr flushing_buffer_size = 2 * flushing_tile_size;
  auto constexpr num_flushing_tiles   = BlockSize / flushing_tile_size;
  auto constexpr max_matches          = flushing_tile_size / tile_size;

  static_assert(flushing_tile_size > 0);

  auto const flushing_tile    = cg::tiled_partition<flushing_tile_size>(block);
  auto const flushing_tile_id = flushing_tile.meta_group_rank();

  __shared__ cuco::pair<ProbeKey, Key> flushing_tile_buffer[num_flushing_tiles][flushing_tile_size];

  using atomic_counter_type = cuda::atomic<size_type, cuda::thread_scope_block>;
  // per flushing-tile counter to track number of filled elements
  __shared__ atomic_counter_type flushing_counter[num_flushing_tiles];

#if defined(CUCO_HAS_CG_INVOKE_ONE)
  cg::invoke_one(flushing_tile,
                 [&]() { new (&flushing_counter[flushing_tile_id]) atomic_counter_type{0}; });
#else
  if (flushing_tile.thread_rank() == 0) {
    new (&flushing_counter[flushing_tile_id]) atomic_counter_type{0};
  }
#endif
  flushing_tile.sync();  // sync still needed since cg.any doesn't imply a memory barrier

  while (flushing_tile.any(idx < n)) {
    bool active_flag = idx < n;
    auto const active_flushing_tile =
      cg::binary_partition<flushing_tile_size>(flushing_tile, active_flag);
    if (active_flag) {
      auto const found = ref.find(tile, *(first + idx));
#if defined(CUCO_HAS_CG_INVOKE_ONE)
      if (found != ref.end()) {
        cg::invoke_one(tile, [&]() {
          auto const offset =
            flushing_counter[flushing_tile_id].fetch_add(1, cuda::std::memory_order_relaxed);
          flushing_tile_buffer[flushing_tile_id][offset] = {*(first + idx), *found};
        });
      }
#else
      if (tile.thread_rank() == 0 and found != ref.end()) {
        auto const offset =
          flushing_counter[flushing_tile_id].fetch_add(1, cuda::std::memory_order_relaxed);
        flushing_tile_buffer[flushing_tile_id][offset] = {*(first + idx), *found};
      }
#endif
    }

    flushing_tile.sync();
    auto const buffer_size =
      flushing_counter[flushing_tile_id].load(cuda::std::memory_order_relaxed);
    if (buffer_size + max_matches > flushing_buffer_size) {
      flush_buffer(flushing_tile,
                   buffer_size,
                   flushing_tile_buffer[flushing_tile_id],
                   counter,
                   output_probe,
                   output_match);
      flushing_tile.sync();
#if defined(CUCO_HAS_CG_INVOKE_ONE)
      cg::invoke_one(flushing_tile, [&]() {
        flushing_counter[flushing_tile_id].store(0, cuda::std::memory_order_relaxed);
      });
#else
      if (flushing_tile.thread_rank() == 0) {
        flushing_counter[flushing_tile_id].store(0, cuda::std::memory_order_relaxed);
      }
#endif
      flushing_tile.sync();
    }

    idx += stride;
  }  // while

  auto const buffer_size = flushing_counter[flushing_tile_id].load(cuda::std::memory_order_relaxed);
  if (buffer_size > 0) {
    flush_buffer(flushing_tile,
                 buffer_size,
                 flushing_tile_buffer[flushing_tile_id],
                 counter,
                 output_probe,
                 output_match);
  }
}

template <typename Size, typename ProbeKey, typename Key, typename OutputIt1, typename OutputIt2>
__device__ void flush_buffer(cooperative_groups::thread_block const& block,
                             Size buffer_size,
                             cuco::pair<ProbeKey, Key>* buffer,
                             cuda::atomic<Size, cuda::thread_scope_device>* counter,
                             OutputIt1 output_probe,
                             OutputIt2 output_match)
{
  auto i = block.thread_rank();
  __shared__ Size offset;

#if defined(CUCO_HAS_CG_INVOKE_ONE)
  cooperative_groups::invoke_one(
    block, [&]() { offset = counter->fetch_add(buffer_size, cuda::std::memory_order_relaxed); });
#else
  if (i == 0) { offset = counter->fetch_add(buffer_size, cuda::std::memory_order_relaxed); }
#endif
  block.sync();

  while (i < buffer_size) {
    *(output_probe + offset + i) = buffer[i].first;
    *(output_match + offset + i) = buffer[i].second;

    i += block.size();
  }
}

template <int32_t BlockSize,
          typename InputIt,
          typename OutputIt1,
          typename OutputIt2,
          typename AtomicT,
          typename Ref>
__device__ void scalar_retrieve(InputIt first,
                                cuco::detail::index_type n,
                                OutputIt1 output_probe,
                                OutputIt2 output_match,
                                AtomicT* counter,
                                Ref ref)
{
  namespace cg = cooperative_groups;

  using size_type = typename Ref::size_type;
  using ProbeKey  = typename std::iterator_traits<InputIt>::value_type;
  using Key       = typename Ref::key_type;

  auto idx          = cuco::detail::global_thread_id();
  auto const stride = cuco::detail::grid_stride();
  auto const block  = cg::this_thread_block();

  using block_scan = cub::BlockScan<size_type, BlockSize>;
  __shared__ typename block_scan::TempStorage block_scan_temp_storage;

  auto constexpr buffer_capacity = 2 * BlockSize;
  __shared__ cuco::pair<ProbeKey, Key> buffer[buffer_capacity];
  size_type buffer_size = 0;

  while (idx - block.thread_rank() < n) {  // the whole thread block falls into the same iteration
    auto const found     = idx < n ? ref.find(*(first + idx)) : ref.end();
    auto const has_match = found != ref.end();

    // Use a whole-block scan to calculate the output location
    size_type offset;
    size_type block_count;
    block_scan(block_scan_temp_storage)
      .ExclusiveSum(static_cast<size_type>(has_match), offset, block_count);

    if (buffer_size + block_count > buffer_capacity) {
      flush_buffer(block, buffer_size, buffer, counter, output_probe, output_match);
      block.sync();
      buffer_size = 0;
    }

    if (has_match) { buffer[buffer_size + offset] = {*(first + idx), *found}; }
    buffer_size += block_count;
    block.sync();

    idx += stride;
  }  // while

  if (buffer_size > 0) {
    flush_buffer(block, buffer_size, buffer, counter, output_probe, output_match);
  }
}

template <int32_t BlockSize,
          typename InputIt,
          typename OutputIt1,
          typename OutputIt2,
          typename AtomicT,
          typename Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void retrieve(InputIt first,
                                                       cuco::detail::index_type n,
                                                       OutputIt1 output_probe,
                                                       OutputIt2 output_match,
                                                       AtomicT* counter,
                                                       Ref ref)
{
  // Scalar retrieve without using CG
  if constexpr (Ref::cg_size == 1) {
    scalar_retrieve<BlockSize>(first, n, output_probe, output_match, counter, ref);
  } else {
    group_retrieve<BlockSize>(first, n, output_probe, output_match, counter, ref);
  }
}

}  // namespace cuco::static_set_ns::detail
