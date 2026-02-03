/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
#include <cuda/std/atomic>
#include <cuda/std/iterator>

#include <cooperative_groups.h>

namespace cuco::detail::dynamic_map_ns {
namespace cg = cooperative_groups;

CUCO_SUPPRESS_KERNEL_WARNINGS

/**
 * @brief Inserts key/value pairs into the map, checking all submaps for duplicates.
 *
 * For each key, checks all submaps except the target for existing keys. Only inserts
 * if the key doesn't exist in any other submap.
 *
 * @tparam CGSize Cooperative group size
 * @tparam BlockSize The number of threads in the thread block
 * @tparam InputIt Device accessible input iterator
 * @tparam AtomicT Atomic counter type
 * @tparam Ref Type of submap device ref with both contains and insert capabilities
 *
 * @param first Beginning of the sequence of key/value pairs
 * @param n Number of keys
 * @param num_successes Pointer to atomic counter for successful insertions
 * @param submap_refs Array of submap refs (with contains and insert ops)
 * @param insert_idx Index of the submap we're inserting into
 * @param num_submaps Total number of submaps
 */
template <int CGSize, int BlockSize, typename InputIt, typename AtomicT, typename Ref>
CUCO_KERNEL void insert(InputIt first,
                        cuco::detail::index_type n,
                        AtomicT* num_successes,
                        Ref* submap_refs,
                        uint32_t insert_idx,
                        uint32_t num_submaps)
{
  using BlockReduce = cub::BlockReduce<std::size_t, BlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  std::size_t thread_num_successes = 0;

  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  while (idx < n) {
    typename cuda::std::iterator_traits<InputIt>::value_type const pair{*(first + idx)};
    bool exists = false;

    if constexpr (CGSize == 1) {
      for (uint32_t i = 0; i < num_submaps && !exists; ++i) {
        if (i != insert_idx) { exists = submap_refs[i].contains(pair.first); }
      }

      if (!exists) {
        if (submap_refs[insert_idx].insert(pair)) { ++thread_num_successes; }
      }
    } else {
      auto const tile = cg::tiled_partition<CGSize>(cg::this_thread_block());

      for (uint32_t i = 0; i < num_submaps && !exists; ++i) {
        if (i != insert_idx) { exists = submap_refs[i].contains(tile, pair.first); }
      }
      tile.sync();

      if (!exists) {
        if (submap_refs[insert_idx].insert(tile, pair) && tile.thread_rank() == 0) {
          ++thread_num_successes;
        }
      }
    }
    idx += loop_stride;
  }

  // Aggregate success count
  std::size_t const block_num_successes = BlockReduce(temp_storage).Sum(thread_num_successes);
  if (threadIdx.x == 0) {
    num_successes->fetch_add(block_num_successes, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Inserts or assigns key/value pairs, checking all submaps.
 *
 * For each key, checks all submaps. If found, assigns the new value. If not found,
 * inserts into the target submap. Only counts new insertions (not assignments).
 *
 * @tparam CGSize Cooperative group size
 * @tparam BlockSize The number of threads in the thread block
 * @tparam InputIt Device accessible input iterator
 * @tparam AtomicT Atomic counter type
 * @tparam Ref Type of submap device ref with contains, insert, and insert_or_assign capabilities
 *
 * @param first Beginning of the sequence of key/value pairs
 * @param n Number of keys
 * @param num_insertions Pointer to atomic counter for new insertions (not assignments)
 * @param submap_refs Array of submap refs
 * @param insert_idx Index of the submap to insert into if key not found
 * @param num_submaps Total number of submaps
 */
template <int CGSize, int BlockSize, typename InputIt, typename AtomicT, typename Ref>
CUCO_KERNEL void insert_or_assign(InputIt first,
                                  cuco::detail::index_type n,
                                  AtomicT* num_insertions,
                                  Ref* submap_refs,
                                  uint32_t insert_idx,
                                  uint32_t num_submaps)
{
  using BlockReduce = cub::BlockReduce<std::size_t, BlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  std::size_t thread_num_insertions = 0;

  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  while (idx < n) {
    typename cuda::std::iterator_traits<InputIt>::value_type const pair{*(first + idx)};
    bool found = false;

    if constexpr (CGSize == 1) {
      for (uint32_t i = 0; i < num_submaps && !found; ++i) {
        if (submap_refs[i].contains(pair.first)) {
          submap_refs[i].insert_or_assign(pair);
          found = true;
        }
      }

      if (!found) {
        if (submap_refs[insert_idx].insert(pair)) { ++thread_num_insertions; }
      }
    } else {
      auto const tile = cg::tiled_partition<CGSize>(cg::this_thread_block());

      for (uint32_t i = 0; i < num_submaps && !found; ++i) {
        if (submap_refs[i].contains(tile, pair.first)) {
          submap_refs[i].insert_or_assign(tile, pair);
          found = true;
        }
      }
      tile.sync();

      if (!found) {
        if (submap_refs[insert_idx].insert(tile, pair) && tile.thread_rank() == 0) {
          ++thread_num_insertions;
        }
      }
    }
    idx += loop_stride;
  }

  // Aggregate insertion count
  std::size_t const block_num_insertions = BlockReduce(temp_storage).Sum(thread_num_insertions);
  if (threadIdx.x == 0) {
    num_insertions->fetch_add(block_num_insertions, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Erases keys from all submaps.
 *
 * For each key, attempts to erase from all submaps. Tracks total erased count.
 *
 * @tparam CGSize Cooperative group size
 * @tparam BlockSize The number of threads in the thread block
 * @tparam InputIt Device accessible input iterator
 * @tparam AtomicT Atomic counter type
 * @tparam Ref Type of submap device ref with erase capability
 *
 * @param first Beginning of the sequence of keys
 * @param n Number of keys
 * @param num_successes Pointer to atomic counter for successful erasures
 * @param submap_refs Array of submap refs for erase operations
 * @param num_submaps Total number of submaps
 */
template <int CGSize, int BlockSize, typename InputIt, typename AtomicT, typename Ref>
CUCO_KERNEL void erase(InputIt first,
                       cuco::detail::index_type n,
                       AtomicT* num_successes,
                       Ref* submap_refs,
                       uint32_t num_submaps)
{
  using BlockReduce = cub::BlockReduce<std::size_t, BlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  std::size_t thread_num_successes = 0;

  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  while (idx < n) {
    typename cuda::std::iterator_traits<InputIt>::value_type const key{*(first + idx)};

    if constexpr (CGSize == 1) {
      for (uint32_t i = 0; i < num_submaps; ++i) {
        if (submap_refs[i].erase(key)) {
          ++thread_num_successes;
          break;
        }
      }
    } else {
      auto const tile = cg::tiled_partition<CGSize>(cg::this_thread_block());

      for (uint32_t i = 0; i < num_submaps; ++i) {
        if (submap_refs[i].erase(tile, key)) {
          if (tile.thread_rank() == 0) { ++thread_num_successes; }
          break;
        }
      }
    }
    idx += loop_stride;
  }

  // Aggregate success count
  std::size_t const block_num_successes = BlockReduce(temp_storage).Sum(thread_num_successes);
  if (threadIdx.x == 0) {
    num_successes->fetch_add(block_num_successes, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Finds the values corresponding to all keys in the range `[first, last)`.
 *
 * If the key `*(first + i)` exists in any submap, copies its associated value to `(output_begin +
 * i)`. Else, copies the empty value sentinel.
 *
 * @tparam CGSize Cooperative group size
 * @tparam BlockSize The number of threads in the thread block
 * @tparam InputIt Device accessible input iterator
 * @tparam OutputIt Device accessible output iterator
 * @tparam Ref Type of submap device ref
 *
 * @param first Beginning of the sequence of keys
 * @param n Number of keys
 * @param output_begin Beginning of the sequence of values retrieved for each key
 * @param submap_refs Array of submap device refs
 * @param num_submaps The number of submaps in the map
 */
template <int CGSize, int BlockSize, typename InputIt, typename OutputIt, typename Ref>
CUCO_KERNEL void find(InputIt first,
                      cuco::detail::index_type n,
                      OutputIt output_begin,
                      Ref const* submap_refs,
                      uint32_t num_submaps)
{
  using mapped_type = typename Ref::mapped_type;

  auto const empty_value_sentinel = submap_refs[0].empty_value_sentinel();

  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  __shared__ mapped_type write_buffer[BlockSize];

  while (idx < n) {
    typename cuda::std::iterator_traits<InputIt>::value_type const key{*(first + idx)};
    auto found_value = empty_value_sentinel;
    bool found       = false;

    if constexpr (CGSize == 1) {
      for (uint32_t i = 0; i < num_submaps && !found; ++i) {
        auto const result = submap_refs[i].find(key);
        if (result != submap_refs[i].end()) {
          found_value = result->second;
          found       = true;
        }
      }
      write_buffer[threadIdx.x] = found_value;
      __syncthreads();
      *(output_begin + idx) = write_buffer[threadIdx.x];
    } else {
      auto const tile = cg::tiled_partition<CGSize>(cg::this_thread_block());

      for (uint32_t i = 0; i < num_submaps && !found; ++i) {
        auto const result = submap_refs[i].find(tile, key);
        if (result != submap_refs[i].end()) {
          found_value = result->second;
          found       = true;
        }
      }

      if (tile.thread_rank() == 0) { write_buffer[threadIdx.x / CGSize] = found_value; }
      __syncthreads();
      if (tile.thread_rank() == 0) { *(output_begin + idx) = write_buffer[threadIdx.x / CGSize]; }
    }
    idx += loop_stride;
  }
}

/**
 * @brief Indicates whether the keys in the range `[first, first + n)` are contained in any submap.
 *
 * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
 *
 * @tparam CGSize Cooperative group size
 * @tparam BlockSize The number of threads in the thread block
 * @tparam InputIt Device accessible input iterator
 * @tparam OutputIt Device accessible output iterator
 * @tparam Ref Type of submap device ref
 *
 * @param first Beginning of the sequence of keys
 * @param n Number of keys
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param submap_refs Array of submap device refs
 * @param num_submaps The number of submaps in the map
 */
/**
 * @brief Retrieves all key-value pairs from all submaps.
 *
 * Iterates through all slots across all submaps, outputting non-empty/non-erased pairs.
 *
 * @tparam BlockSize The number of threads in the thread block
 * @tparam Key Key type
 * @tparam Value Mapped value type
 * @tparam KeyOut Device accessible output iterator for keys
 * @tparam ValueOut Device accessible output iterator for values
 * @tparam SlotT Slot type (cuco::pair<Key, Value>)
 * @tparam AtomicT Atomic counter type
 *
 * @param keys_out Beginning output iterator for keys
 * @param values_out Beginning output iterator for values
 * @param slot_arrays Array of pointers to each submap's slot storage
 * @param num_submaps Number of submaps
 * @param total_capacity Total number of slots across all submaps
 * @param num_out Pointer to atomic counter for number of retrieved pairs
 * @param capacity_prefix_sum Prefix sum of submap capacities
 * @param empty_key_sentinel Sentinel value for empty key
 * @param erased_key_sentinel Sentinel value for erased key
 */
template <int BlockSize,
          typename Key,
          typename Value,
          typename KeyOut,
          typename ValueOut,
          typename SlotT,
          typename AtomicT>
CUCO_KERNEL void retrieve_all(KeyOut keys_out,
                              ValueOut values_out,
                              SlotT const* const* slot_arrays,
                              uint32_t num_submaps,
                              cuco::detail::index_type total_capacity,
                              AtomicT* num_out,
                              cuco::detail::index_type const* capacity_prefix_sum,
                              Key empty_key_sentinel,
                              Key erased_key_sentinel)
{
  using BlockScan = cub::BlockScan<unsigned int, BlockSize>;
  __shared__ typename BlockScan::TempStorage scan_temp_storage;
  __shared__ unsigned int block_base;

  auto idx = cuco::detail::global_thread_id();

  while ((idx - threadIdx.x) < total_capacity) {
    // Determine which submap this slot belongs to
    uint32_t submap_idx = 0;
    auto submap_offset  = idx;

    if (idx < total_capacity) {
      while (submap_idx < num_submaps && idx >= capacity_prefix_sum[submap_idx]) {
        ++submap_idx;
      }
      if (submap_idx > 0) { submap_offset = idx - capacity_prefix_sum[submap_idx - 1]; }
    }

    // Check if slot is filled (not empty and not erased)
    bool is_filled = false;
    Key key{};
    Value value{};

    if (idx < total_capacity && submap_idx < num_submaps) {
      auto const& slot = slot_arrays[submap_idx][submap_offset];
      // Use atomic_ref for thread-safe read
      cuda::atomic_ref<Key const, cuda::thread_scope_device> key_ref(slot.first);
      key       = key_ref.load(cuda::std::memory_order_relaxed);
      is_filled = !cuco::detail::bitwise_compare(key, empty_key_sentinel) &&
                  !cuco::detail::bitwise_compare(key, erased_key_sentinel);
      if (is_filled) {
        cuda::atomic_ref<Value const, cuda::thread_scope_device> value_ref(slot.second);
        value = value_ref.load(cuda::std::memory_order_relaxed);
      }
    }

    // Block scan to compute output positions
    unsigned int local_idx   = 0;
    unsigned int block_valid = 0;
    BlockScan(scan_temp_storage).ExclusiveSum(is_filled ? 1u : 0u, local_idx, block_valid);

    if (threadIdx.x == 0) {
      block_base = num_out->fetch_add(block_valid, cuda::std::memory_order_relaxed);
    }
    __syncthreads();

    if (is_filled) {
      keys_out[block_base + local_idx]   = key;
      values_out[block_base + local_idx] = value;
    }

    idx += cuco::detail::grid_stride();
  }
}

template <int CGSize, int BlockSize, typename InputIt, typename OutputIt, typename Ref>
CUCO_KERNEL void contains(InputIt first,
                          cuco::detail::index_type n,
                          OutputIt output_begin,
                          Ref const* submap_refs,
                          uint32_t num_submaps)
{
  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  __shared__ bool write_buffer[BlockSize];

  while (idx < n) {
    typename cuda::std::iterator_traits<InputIt>::value_type const key{*(first + idx)};
    bool found = false;

    if constexpr (CGSize == 1) {
      for (uint32_t i = 0; i < num_submaps && !found; ++i) {
        found = submap_refs[i].contains(key);
      }
      write_buffer[threadIdx.x] = found;
      __syncthreads();
      *(output_begin + idx) = write_buffer[threadIdx.x];
    } else {
      auto const tile = cg::tiled_partition<CGSize>(cg::this_thread_block());

      for (uint32_t i = 0; i < num_submaps && !found; ++i) {
        found = submap_refs[i].contains(tile, key);
      }

      if (tile.thread_rank() == 0) { write_buffer[threadIdx.x / CGSize] = found; }
      __syncthreads();
      if (tile.thread_rank() == 0) { *(output_begin + idx) = write_buffer[threadIdx.x / CGSize]; }
    }
    idx += loop_stride;
  }
}

}  // namespace cuco::detail::dynamic_map_ns
