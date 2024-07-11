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

#include <cuco/detail/utility/cuda.cuh>

#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/functional>

#include <cooperative_groups.h>

#include <iterator>

namespace cuco::detail {
CUCO_SUPPRESS_KERNEL_WARNINGS

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
 * @tparam InputIt Device accessible input iterator whose `value_type` is
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
 * @param ref Non-owning container device ref used to access the slot storage
 */
template <int32_t CGSize,
          int32_t BlockSize,
          typename InputIt,
          typename StencilIt,
          typename Predicate,
          typename AtomicT,
          typename Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void insert_if_n(InputIt first,
                                                          cuco::detail::index_type n,
                                                          StencilIt stencil,
                                                          Predicate pred,
                                                          AtomicT* num_successes,
                                                          Ref ref)
{
  using BlockReduce = cub::BlockReduce<typename Ref::size_type, BlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  typename Ref::size_type thread_num_successes = 0;

  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  while (idx < n) {
    if (pred(*(stencil + idx))) {
      typename std::iterator_traits<InputIt>::value_type const& insert_element{*(first + idx)};
      if constexpr (CGSize == 1) {
        if (ref.insert(insert_element)) { thread_num_successes++; };
      } else {
        auto const tile =
          cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
        if (ref.insert(tile, insert_element) && tile.thread_rank() == 0) { thread_num_successes++; }
      }
    }
    idx += loop_stride;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  auto const block_num_successes = BlockReduce(temp_storage).Sum(thread_num_successes);
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
 * @tparam InputIt Device accessible input iterator whose `value_type` is
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
 * @param ref Non-owning container device ref used to access the slot storage
 */
template <int32_t CGSize,
          int32_t BlockSize,
          typename InputIt,
          typename StencilIt,
          typename Predicate,
          typename Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void insert_if_n(
  InputIt first, cuco::detail::index_type n, StencilIt stencil, Predicate pred, Ref ref)
{
  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  while (idx < n) {
    if (pred(*(stencil + idx))) {
      typename std::iterator_traits<InputIt>::value_type const& insert_element{*(first + idx)};
      if constexpr (CGSize == 1) {
        ref.insert(insert_element);
      } else {
        auto const tile =
          cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
        ref.insert(tile, insert_element);
      }
    }
    idx += loop_stride;
  }
}

/**
 * @brief Asynchronously erases keys in the range `[first, first + n)`.
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
CUCO_KERNEL __launch_bounds__(BlockSize) void erase(InputIt first,
                                                    cuco::detail::index_type n,
                                                    Ref ref)
{
  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  while (idx < n) {
    typename std::iterator_traits<InputIt>::value_type const& erase_element{*(first + idx)};
    if constexpr (CGSize == 1) {
      ref.erase(erase_element);
    } else {
      auto const tile =
        cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
      ref.erase(tile, erase_element);
    }
    idx += loop_stride;
  }
}

/**
 * @brief Indicates whether the keys in the range `[first, first + n)` are contained in the data
 * structure if `pred` of the corresponding stencil returns true.
 *
 * @note If `pred( *(stencil + i) )` is true, stores `true` or `false` to `(output_begin + i)`
 * indicating if the key `*(first + i)` is present in the container. If `pred( *(stencil + i) )` is
 * false, stores false to `(output_begin + i)`.
 *
 * @tparam CGSize Number of threads in each CG
 * @tparam BlockSize The size of the thread block
 * @tparam InputIt Device accessible input iterator
 * @tparam StencilIt Device accessible random access iterator whose value_type is
 * convertible to Predicate's argument type
 * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool`
 * and argument type is convertible from `std::iterator_traits<StencilIt>::value_type`
 * @tparam OutputIt Device accessible output iterator assignable from `bool`
 * @tparam Ref Type of non-owning device ref allowing access to storage
 *
 * @param first Beginning of the sequence of keys
 * @param n Number of keys
 * @param stencil Beginning of the stencil sequence
 * @param pred Predicate to test on every element in the range `[stencil, stencil + n)`
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param ref Non-owning container device ref used to access the slot storage
 */
template <int32_t CGSize,
          int32_t BlockSize,
          typename InputIt,
          typename StencilIt,
          typename Predicate,
          typename OutputIt,
          typename Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void contains_if_n(InputIt first,
                                                            cuco::detail::index_type n,
                                                            StencilIt stencil,
                                                            Predicate pred,
                                                            OutputIt output_begin,
                                                            Ref ref)
{
  namespace cg = cooperative_groups;

  auto const block       = cg::this_thread_block();
  auto const thread_idx  = block.thread_rank();
  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  __shared__ bool output_buffer[BlockSize / CGSize];

  while (idx - thread_idx < n) {  // the whole thread block falls into the same iteration
    if constexpr (CGSize == 1) {
      if (idx < n) {
        typename std::iterator_traits<InputIt>::value_type const& key = *(first + idx);
        /*
         * The ld.relaxed.gpu instruction causes L1 to flush more frequently, causing increased
         * sector stores from L2 to global memory. By writing results to shared memory and then
         * synchronizing before writing back to global, we no longer rely on L1, preventing the
         * increase in sector stores from L2 to global and improving performance.
         */
        output_buffer[thread_idx] = pred(*(stencil + idx)) ? ref.contains(key) : false;
      }
      block.sync();
      if (idx < n) { *(output_begin + idx) = output_buffer[thread_idx]; }
    } else {
      auto const tile = cg::tiled_partition<CGSize>(cg::this_thread_block());
      if (idx < n) {
        typename std::iterator_traits<InputIt>::value_type const& key = *(first + idx);
        auto const found = pred(*(stencil + idx)) ? ref.contains(tile, key) : false;
        if (tile.thread_rank() == 0) { *(output_begin + idx) = found; }
      }
    }
    idx += loop_stride;
  }
}

/**
 * @brief Helper to determine the buffer type for the find kernel
 *
 * @tparam Container Container type
 */
template <typename Container, typename = void>
struct find_buffer {
  using type = typename Container::key_type;  ///< Buffer type
};

/**
 * @brief Helper to determine the buffer type for the find kernel
 *
 * @note Specialization if `mapped_type` exists
 *
 * @tparam Container Container type
 */
template <typename Container>
struct find_buffer<Container, cuda::std::void_t<typename Container::mapped_type>> {
  using type = typename Container::mapped_type;  ///< Buffer type
};

/**
 * @brief Finds the equivalent container elements of all keys in the range `[first, first + n)`.
 *
 * @note If the key `*(first + i)` has a match in the container, copies the match to `(output_begin
 * + i)`. Else, copies the empty sentinel. Uses the CUDA Cooperative Groups API to leverage groups
 * of multiple threads to find each key. This provides a significant boost in throughput compared to
 * the non Cooperative Group `find` at moderate to high load factors.
 *
 * @tparam CGSize Number of threads in each CG
 * @tparam BlockSize The size of the thread block
 * @tparam InputIt Device accessible input iterator
 * @tparam OutputIt Device accessible output iterator
 * @tparam Ref Type of non-owning device ref allowing access to storage
 *
 * @param first Beginning of the sequence of keys
 * @param n Number of keys to query
 * @param output_begin Beginning of the sequence of matched payloads retrieved for each key
 * @param ref Non-owning container device ref used to access the slot storage
 */
template <int32_t CGSize, int32_t BlockSize, typename InputIt, typename OutputIt, typename Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void find(InputIt first,
                                                   cuco::detail::index_type n,
                                                   OutputIt output_begin,
                                                   Ref ref)
{
  namespace cg = cooperative_groups;

  auto const block       = cg::this_thread_block();
  auto const thread_idx  = block.thread_rank();
  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  using output_type = typename find_buffer<Ref>::type;
  __shared__ output_type output_buffer[BlockSize / CGSize];

  auto constexpr has_payload = not std::is_same_v<typename Ref::key_type, typename Ref::value_type>;

  auto const sentinel = [&]() {
    if constexpr (has_payload) {
      return ref.empty_value_sentinel();
    } else {
      return ref.empty_key_sentinel();
    }
  }();

  auto output = cuda::proclaim_return_type<output_type>([&] __device__(auto found) {
    if constexpr (has_payload) {
      return found == ref.end() ? sentinel : found->second;
    } else {
      return found == ref.end() ? sentinel : *found;
    }
  });

  while (idx - thread_idx < n) {  // the whole thread block falls into the same iteration
    if (idx < n) {
      typename std::iterator_traits<InputIt>::value_type const& key = *(first + idx);
      if constexpr (CGSize == 1) {
        auto const found = ref.find(key);
        /*
         * The ld.relaxed.gpu instruction causes L1 to flush more frequently, causing increased
         * sector stores from L2 to global memory. By writing results to shared memory and then
         * synchronizing before writing back to global, we no longer rely on L1, preventing the
         * increase in sector stores from L2 to global and improving performance.
         */
        output_buffer[thread_idx] = output(found);
        block.sync();
        *(output_begin + idx) = output_buffer[thread_idx];
      } else {
        auto const tile  = cg::tiled_partition<CGSize>(block);
        auto const found = ref.find(tile, key);

        if (tile.thread_rank() == 0) { *(output_begin + idx) = output(found); }
      }
    }
    idx += loop_stride;
  }
}

// TODO docs
template <bool IsOuter,
          int32_t BlockSize,
          class InputProbeIt,
          class OutputProbeIt,
          class OutputMatchIt,
          class AtomicCounter,
          class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void retrieve(InputProbeIt input_probe,
                                                       cuco::detail::index_type n,
                                                       OutputProbeIt output_probe,
                                                       OutputMatchIt output_match,
                                                       AtomicCounter* atomic_counter,
                                                       Ref ref)
{
  auto constexpr tile_size = cuco::detail::warp_size();  // TODO include

  namespace cg        = cooperative_groups;
  auto const block    = cg::this_thread_block();
  auto const tile     = cg::tiled_partition<tile_size>(block);
  auto const tile_idx = cuco::detail::global_thread_id() / tile_size;

  auto const tiles_in_grid  = (gridDim.x * BlockSize) / tile_size;
  auto const elems_per_tile = cuco::detail::int_div_ceil(n, tiles_in_grid);  // TODO include

  auto const tile_begin_offset = tile_idx * elems_per_tile;
  auto const tile_end_offset   = max(n, tile_begin_offset + elems_per_tile);

  if (tile_begin_offset < tile_end_offset) {
    if constexpr (IsOuter) {
      ref.retrieve_outer(tile,
                         input_probe + tile_begin_offset,
                         input_probe + tile_end_offset,
                         output_probe,
                         output_match,
                         *atomic_counter);
    } else {
      ref.retrieve(tile,
                   input_probe + tile_begin_offset,
                   input_probe + tile_end_offset,
                   output_probe,
                   output_match,
                   *atomic_counter);
    }
  }
}

/**
 * @brief Counts the occurrences of keys in `[first, last)` contained in the container
 *
 * @tparam IsOuter Flag indicating whether it's an outer count or not
 * @tparam CGSize Number of threads in each CG
 * @tparam BlockSize Number of threads in each block
 * @tparam InputIt Device accessible input iterator
 * @tparam AtomicT Atomic counter type
 * @tparam Ref Type of non-owning device container ref allowing access to storage
 *
 * @param first Beginning of the sequence of input elements
 * @param n Number of input elements
 * @param count Number of matches
 * @param ref Non-owning container device ref used to access the slot storage
 */
template <bool IsOuter,
          int32_t CGSize,
          int32_t BlockSize,
          typename InputIt,
          typename AtomicT,
          typename Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void count(InputIt first,
                                                    cuco::detail::index_type n,
                                                    AtomicT* count,
                                                    Ref ref)
{
  using size_type = typename Ref::size_type;

  size_type constexpr outer_min_count = 1;

  using BlockReduce = cub::BlockReduce<size_type, BlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  size_type thread_count = 0;

  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  while (idx < n) {
    typename std::iterator_traits<InputIt>::value_type const& key = *(first + idx);
    if constexpr (CGSize == 1) {
      if constexpr (IsOuter) {
        thread_count += max(ref.count(key), outer_min_count);
      } else {
        thread_count += ref.count(key);
      }
    } else {
      auto const tile =
        cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
      if constexpr (IsOuter) {
        auto temp_count = ref.count(tile, key);
        if (tile.all(temp_count == 0) and tile.thread_rank() == 0) { ++temp_count; }
        thread_count += temp_count;
      } else {
        thread_count += ref.count(tile, key);
      }
    }
    idx += loop_stride;
  }

  auto const block_count = BlockReduce(temp_storage).Sum(thread_count);
  if (threadIdx.x == 0) { count->fetch_add(block_count, cuda::std::memory_order_relaxed); }
}

/**
 * @brief Calculates the number of filled slots for the given window storage.
 *
 * @tparam BlockSize Number of threads in each block
 * @tparam StorageRef Type of non-owning ref allowing access to storage
 * @tparam Predicate Type of predicate indicating if the given slot is filled
 * @tparam AtomicT Atomic counter type
 *
 * @param storage Non-owning device ref used to access the slot storage
 * @param is_filled Predicate indicating if the given slot is filled
 * @param count Number of filled slots
 */
template <int32_t BlockSize, typename StorageRef, typename Predicate, typename AtomicT>
CUCO_KERNEL __launch_bounds__(BlockSize) void size(StorageRef storage,
                                                   Predicate is_filled,
                                                   AtomicT* count)
{
  using size_type = typename StorageRef::size_type;

  auto const loop_stride = cuco::detail::grid_stride();
  auto idx               = cuco::detail::global_thread_id();

  size_type thread_count = 0;
  auto const n           = storage.num_windows();

  while (idx < n) {
    auto const window = storage[idx];
#pragma unroll
    for (auto const& it : window) {
      thread_count += static_cast<size_type>(is_filled(it));
    }
    idx += loop_stride;
  }

  using BlockReduce = cub::BlockReduce<size_type, BlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  auto const block_count = BlockReduce(temp_storage).Sum(thread_count);
  if (threadIdx.x == 0) { count->fetch_add(block_count, cuda::std::memory_order_relaxed); }
}

template <int32_t BlockSize, typename ContainerRef, typename Predicate>
CUCO_KERNEL __launch_bounds__(BlockSize) void rehash(
  typename ContainerRef::storage_ref_type storage_ref,
  ContainerRef container_ref,
  Predicate is_filled)
{
  namespace cg = cooperative_groups;

  __shared__ typename ContainerRef::value_type buffer[BlockSize];
  __shared__ unsigned int buffer_size;

  auto constexpr cg_size = ContainerRef::cg_size;
  auto const block       = cg::this_thread_block();
  auto const tile        = cg::tiled_partition<cg_size>(block);

  auto const thread_rank         = block.thread_rank();
  auto constexpr tiles_per_block = BlockSize / cg_size;  // tile.meta_group_size() but constexpr
  auto const tile_rank           = tile.meta_group_rank();
  auto const loop_stride         = cuco::detail::grid_stride();
  auto idx                       = cuco::detail::global_thread_id();
  auto const n                   = storage_ref.num_windows();

  while (idx - thread_rank < n) {
    if (thread_rank == 0) { buffer_size = 0; }
    block.sync();

    // gather values in shmem buffer
    if (idx < n) {
      auto const window = storage_ref[idx];

      for (auto const& slot : window) {
        if (is_filled(slot)) { buffer[atomicAdd_block(&buffer_size, 1)] = slot; }
      }
    }
    block.sync();

    auto const local_buffer_size = buffer_size;

    // insert from shmem buffer into the container
    for (auto tidx = tile_rank; tidx < local_buffer_size; tidx += tiles_per_block) {
      container_ref.insert(tile, buffer[tidx]);
    }
    block.sync();

    idx += loop_stride;
  }
}

}  // namespace cuco::detail
