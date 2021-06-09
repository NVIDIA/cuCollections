/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cooperative_groups/memcpy_async.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <cuda/barrier>

#include <cuco/detail/pair.cuh>

namespace cuco {
namespace detail {
namespace cg = cooperative_groups;

/**
 * @brief Flushes per-CG shared memory buffer into the output sequence using CG memcpy_async.
 *
 * @tparam cg_size The number of threads in the Cooperative Groups
 * @tparam CG Cooperative Group type
 * @tparam Key key type
 * @tparam Value value type
 * @tparam atomicT Type of atomic storage
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @param g The Cooperative Group used to flush output buffer
 * @param num_outputs Number of valid output in the buffer
 * @param output_buffer Shared memory buffer of the key/value pair sequence
 * @param num_matches Size of the output sequence
 * @param output_begin Beginning of the output sequence of key/value pairs
 */
template <uint32_t cg_size,
          typename CG,
          typename Key,
          typename Value,
          typename atomicT,
          typename OutputIt>
__inline__ __device__ std::enable_if_t<thrust::is_contiguous_iterator<OutputIt>::value, void>
flush_output_buffer(CG const& g,
                    uint32_t const num_outputs,
                    cuco::pair_type<Key, Value>* output_buffer,
                    atomicT* num_matches,
                    OutputIt output_begin)
{
  std::size_t offset;
  const auto lane_id = g.thread_rank();
  if (0 == lane_id) {
    offset = num_matches->fetch_add(num_outputs, cuda::std::memory_order_relaxed);
  }
  offset = g.shfl(offset, 0);

  cg::memcpy_async(g,
                   output_begin + offset,
                   output_buffer,
                   cuda::aligned_size_t<alignof(cuco::pair_type<Key, Value>)>(
                     sizeof(cuco::pair_type<Key, Value>) * num_outputs));
}

/**
 * @brief Flushes per-CG shared memory buffer into the output sequence.
 *
 * @tparam cg_size The number of threads in the Cooperative Groups
 * @tparam CG Cooperative Group type
 * @tparam Key key type
 * @tparam Value value type
 * @tparam atomicT Type of atomic storage
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @param g The Cooperative Group used to flush output buffer
 * @param num_outputs Number of valid output in the buffer
 * @param output_buffer Shared memory buffer of the key/value pair sequence
 * @param num_matches Size of the output sequence
 * @param output_begin Beginning of the output sequence of key/value pairs
 */
template <uint32_t cg_size,
          typename CG,
          typename Key,
          typename Value,
          typename atomicT,
          typename OutputIt>
__inline__ __device__ std::enable_if_t<not thrust::is_contiguous_iterator<OutputIt>::value, void>
flush_output_buffer(CG const& g,
                    uint32_t const num_outputs,
                    cuco::pair_type<Key, Value>* output_buffer,
                    atomicT* num_matches,
                    OutputIt output_begin)
{
  std::size_t offset;
  const auto lane_id = g.thread_rank();
  if (0 == lane_id) {
    offset = num_matches->fetch_add(num_outputs, cuda::std::memory_order_relaxed);
  }
  offset = g.shfl(offset, 0);

  for (auto index = lane_id; index < num_outputs; index += cg_size) {
    *(output_begin + offset + index) = output_buffer[index];
  }
}

/**
 * @brief Flushes per-warp shared memory buffer into the output sequence.
 *
 * @tparam Key key type
 * @tparam Value value type
 * @tparam atomicT Type of atomic storage
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @param activemask Mask of active threads in the warp
 * @param num_outputs Number of valid output in the buffer
 * @param output_buffer Shared memory buffer of the key/value pair sequence
 * @param num_matches Size of the output sequence
 * @param output_begin Beginning of the output sequence of key/value pairs
 */
template <typename Key, typename Value, typename atomicT, typename OutputIt>
__inline__ __device__ void flush_output_buffer(const unsigned int activemask,
                                               uint32_t const num_outputs,
                                               cuco::pair_type<Key, Value>* output_buffer,
                                               atomicT* num_matches,
                                               OutputIt output_begin)
{
  int num_threads = __popc(activemask);

  std::size_t offset;
  const auto lane_id = threadIdx.x % 32;
  if (0 == lane_id) {
    offset = num_matches->fetch_add(num_outputs, cuda::std::memory_order_relaxed);
  }
  offset = __shfl_sync(activemask, offset, 0);

  for (auto index = lane_id; index < num_outputs; index += num_threads) {
    *(output_begin + offset + index) = output_buffer[index];
  }
}

/**
 * @brief Initializes each slot in the flat `slots` storage to contain `k` and `v`.
 *
 * Each space in `slots` that can hold a key value pair is initialized to a
 * `pair_atomic_type` containing the key `k` and the value `v`.
 *
 * @tparam atomic_key_type Type of the `Key` atomic container
 * @tparam atomic_mapped_type Type of the `Value` atomic container
 * @tparam Key key type
 * @tparam Value value type
 * @tparam pair_atomic_type key/value pair type
 * @param slots Pointer to flat storage for the map's key/value pairs
 * @param k Key to which all keys in `slots` are initialized
 * @param v Value to which all values in `slots` are initialized
 * @param size Size of the storage pointed to by `slots`
 */
template <typename atomic_key_type,
          typename atomic_mapped_type,
          typename Key,
          typename Value,
          typename pair_atomic_type>
__global__ void initialize(pair_atomic_type* const slots, Key k, Value v, std::size_t size)
{
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < size) {
    new (&slots[tid].first) atomic_key_type{k};
    new (&slots[tid].second) atomic_mapped_type{v};
    tid += gridDim.x * blockDim.x;
  }
}

/**
 * @brief Inserts all key/value pairs in the range `[first, last)`.
 *
 * If multiple keys in `[first, last)` compare equal, it is unspecified which
 * element is inserted. Uses the CUDA Cooperative Groups API to leverage groups
 * of multiple threads to perform each key/value insertion. This provides a
 * significant boost in throughput compared to the non Cooperative Group
 * `insert` at moderate to high load factors.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform
 * inserts
 * @tparam is_vector_load Boolean flag indicating whether vector loads are used or not
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `value_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of key/value pairs
 * @param last End of the sequence of key/value pairs
 * @param view Mutable device view used to access the hash map's slot storage
 * @param key_equal The binary function used to compare two keys for equality
 */
template <uint32_t block_size,
          uint32_t tile_size,
          bool is_vector_load,
          typename InputIt,
          typename viewT,
          typename KeyEqual>
__global__ void insert(InputIt first, InputIt last, viewT view, KeyEqual key_equal)
{
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid  = block_size * blockIdx.x + threadIdx.x;
  auto it   = first + tid / tile_size;

  while (it < last) {
    // force conversion to value_type
    typename viewT::value_type const insert_pair{*it};
    view.insert<is_vector_load>(tile, insert_pair, key_equal);
    it += (gridDim.x * block_size) / tile_size;
  }
}

/**
 * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
 *
 * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
 * Uses the CUDA Cooperative Groups API to leverage groups of multiple threads to perform the
 * contains operation for each key. This provides a significant boost in throughput compared
 * to the non Cooperative Group `contains` at moderate to high load factors.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups
 * @tparam is_vector_load Boolean flag indicating whether vector loads are used or not
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param view Device view used to access the hash map's slot storage
 * @param key_equal The binary function to compare two keys for equality
 */
template <uint32_t block_size,
          uint32_t tile_size,
          bool is_vector_load,
          typename InputIt,
          typename OutputIt,
          typename viewT,
          typename KeyEqual>
__global__ void contains(
  InputIt first, InputIt last, OutputIt output_begin, viewT view, KeyEqual key_equal)
{
  auto tile    = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid     = block_size * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  __shared__ bool writeBuffer[block_size];

  while (first + key_idx < last) {
    auto key   = *(first + key_idx);
    auto found = view.contains<is_vector_load>(tile, key, key_equal);

    /*
     * The ld.relaxed.gpu instruction used in view.find causes L1 to
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    if (tile.thread_rank() == 0) { writeBuffer[threadIdx.x / tile_size] = found; }
    __syncthreads();
    if (tile.thread_rank() == 0) {
      *(output_begin + key_idx) = writeBuffer[threadIdx.x / tile_size];
    }
    key_idx += (gridDim.x * block_size) / tile_size;
  }
}

/**
 * @brief Counts the occurrences of keys in `[first, last)` contained in the multimap. If is_outer
 * is true, the corresponding occurrence for non-matches is 1. Otherwise, it's 0.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform counts
 * @tparam Key key type
 * @tparam Value The type of the mapped value for the map
 * @tparam is_vector_load Boolean flag indicating whether vector loads are used or not
 * @tparam is_outer Boolean flag indicating whether the current functions is used for outer join
 * operations or not
 * @tparam InputIt Device accessible input iterator whose `value_type` is convertible to the map's
 * `key_type`
 * @tparam atomicT Type of atomic storage
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam KeyEqual Binary callable
 * @param first Beginning of the sequence of keys to count
 * @param last End of the sequence of keys to count
 * @param num_matches The number of all the matches for a sequence of keys
 * @param view Device view used to access the hash map's slot storage
 * @param key_equal Binary function to compare two keys for equality
 */
template <uint32_t block_size,
          uint32_t tile_size,
          typename Key,
          typename Value,
          bool is_vector_load,
          bool is_outer = false,
          typename InputIt,
          typename atomicT,
          typename viewT,
          typename KeyEqual>
__global__ std::enable_if_t<is_vector_load, void> count(
  InputIt first, InputIt last, atomicT* num_matches, viewT view, KeyEqual key_equal)
{
  auto tile    = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid     = block_size * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;

  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_matches = 0;

  while (first + key_idx < last) {
    auto key          = *(first + key_idx);
    auto current_slot = view.initial_slot(tile, key);

    if constexpr (is_outer) {
      bool found_match = false;

      while (true) {
        pair<Key, Value> arr[2];
        if constexpr (sizeof(Key) == 4) {
          auto const tmp = *reinterpret_cast<uint4 const*>(current_slot);
          memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
        } else {
          auto const tmp = *reinterpret_cast<ulonglong4 const*>(current_slot);
          memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
        }

        auto const first_slot_is_empty  = (arr[0].first == view.get_empty_key_sentinel());
        auto const second_slot_is_empty = (arr[1].first == view.get_empty_key_sentinel());
        auto const first_equals  = (not first_slot_is_empty and key_equal(arr[0].first, key));
        auto const second_equals = (not second_slot_is_empty and key_equal(arr[1].first, key));

        if (tile.any(first_equals or second_equals)) { found_match = true; }

        thread_num_matches += (first_equals + second_equals);

        if (tile.any(first_slot_is_empty or second_slot_is_empty)) {
          if ((not found_match) && (tile.thread_rank() == 0)) { thread_num_matches++; }
          break;
        }

        current_slot = view.next_slot(current_slot);
      }
    } else {
      while (true) {
        pair<Key, Value> arr[2];
        if constexpr (sizeof(Key) == 4) {
          auto const tmp = *reinterpret_cast<uint4 const*>(current_slot);
          memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
        } else {
          auto const tmp = *reinterpret_cast<ulonglong4 const*>(current_slot);
          memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
        }

        auto const first_slot_is_empty  = (arr[0].first == view.get_empty_key_sentinel());
        auto const second_slot_is_empty = (arr[1].first == view.get_empty_key_sentinel());
        auto const first_equals  = (not first_slot_is_empty and key_equal(arr[0].first, key));
        auto const second_equals = (not second_slot_is_empty and key_equal(arr[1].first, key));

        thread_num_matches += (first_equals + second_equals);

        if (tile.any(first_slot_is_empty or second_slot_is_empty)) { break; }

        current_slot = view.next_slot(current_slot);
      }
    }
    key_idx += (gridDim.x * block_size) / tile_size;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  std::size_t block_num_matches = BlockReduce(temp_storage).Sum(thread_num_matches);
  if (threadIdx.x == 0) {
    num_matches->fetch_add(block_num_matches, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Counts the occurrences of keys in `[first, last)` contained in the multimap. If is_outer
 * is true, the corresponding occurrence for non-matches is 1. Otherwise, it's 0.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform counts
 * @tparam Key key type
 * @tparam Value The type of the mapped value for the map
 * @tparam is_vector_load Boolean flag indicating whether vector loads are used or not
 * @tparam is_outer Boolean flag indicating whether the current functions is used for outer join
 * operations or not
 * @tparam InputIt Device accessible input iterator whose `value_type` is convertible to the map's
 * `key_type`
 * @tparam atomicT Type of atomic storage
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam KeyEqual Binary callable
 * @param first Beginning of the sequence of keys to count
 * @param last End of the sequence of keys to count
 * @param num_matches The number of all the matches for a sequence of keys
 * @param view Device view used to access the hash map's slot storage
 * @param key_equal Binary function to compare two keys for equality
 */
template <uint32_t block_size,
          uint32_t tile_size,
          typename Key,
          typename Value,
          bool is_vector_load,
          bool is_outer = false,
          typename InputIt,
          typename atomicT,
          typename viewT,
          typename KeyEqual>
__global__ std::enable_if_t<not is_vector_load, void> count(
  InputIt first, InputIt last, atomicT* num_matches, viewT view, KeyEqual key_equal)
{
  auto tile    = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid     = block_size * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;

  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_matches = 0;

  while (first + key_idx < last) {
    auto key          = *(first + key_idx);
    auto current_slot = view.initial_slot(tile, key);

    if constexpr (is_outer) {
      bool found_match = false;

      while (true) {
        pair<Key, Value> slot_contents =
          *reinterpret_cast<cuco::pair_type<Key, Value> const*>(current_slot);
        auto const& current_key = slot_contents.first;

        auto const slot_is_empty = (current_key == view.get_empty_key_sentinel());
        auto const equals        = not slot_is_empty and key_equal(current_key, key);

        if (tile.any(equals)) { found_match = true; }

        thread_num_matches += equals;

        if (tile.any(slot_is_empty)) {
          if ((not found_match) && (tile.thread_rank() == 0)) { thread_num_matches++; }
          break;
        }

        current_slot = view.next_slot(current_slot);
      }
    } else {
      while (true) {
        pair<Key, Value> slot_contents =
          *reinterpret_cast<cuco::pair_type<Key, Value> const*>(current_slot);
        auto const& current_key = slot_contents.first;

        auto const slot_is_empty = (current_key == view.get_empty_key_sentinel());
        auto const equals        = not slot_is_empty and key_equal(current_key, key);

        thread_num_matches += equals;

        if (tile.any(slot_is_empty)) { break; }

        current_slot = view.next_slot(current_slot);
      }
    }
    key_idx += (gridDim.x * block_size) / tile_size;
  }
  auto const block_num_matches = BlockReduce(temp_storage).Sum(thread_num_matches);
  if (threadIdx.x == 0) {
    num_matches->fetch_add(block_num_matches, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Counts the occurrences of key/value pairs in `[first, last)` contained in the multimap.
 * If no matches can be found for a given key/value pair, the corresponding occurrence is 1.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform counts
 * @tparam Key key type
 * @tparam Value The type of the mapped value for the map
 * @tparam is_vector_load Boolean flag indicating whether vector loads are used or not
 * @tparam is_outer Boolean flag indicating whether the current functions is used for outer join
 * operations or not
 * @tparam Input Device accesible input iterator of key/value pairs
 * @tparam atomicT Type of atomic storage
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam PairEqual Binary callable
 * @param first Beginning of the sequence of pairs to count
 * @param last End of the sequence of pairs to count
 * @param num_matches The number of all the matches for a sequence of pairs
 * @param view Device view used to access the hash map's slot storage
 * @param pair_equal Binary function to compare two pairs for equality
 */
template <uint32_t block_size,
          uint32_t tile_size,
          typename Key,
          typename Value,
          bool is_vector_load,
          bool is_outer = false,
          typename InputIt,
          typename atomicT,
          typename viewT,
          typename PairEqual>
__global__ std::enable_if_t<is_vector_load, void> pair_count(
  InputIt first, InputIt last, atomicT* num_matches, viewT view, PairEqual pair_equal)
{
  auto tile     = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid      = block_size * blockIdx.x + threadIdx.x;
  auto pair_idx = tid / tile_size;

  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_matches = 0;

  while (first + pair_idx < last) {
    cuco::pair_type<Key, Value> pair = *(first + pair_idx);
    auto key                         = pair.first;
    auto current_slot                = view.initial_slot(tile, key);

    if constexpr (is_outer) {
      bool found_match = false;

      while (true) {
        cuco::pair_type<Key, Value> arr[2];
        if constexpr (sizeof(Key) == 4) {
          auto const tmp = *reinterpret_cast<uint4 const*>(current_slot);
          memcpy(&arr[0], &tmp, 2 * sizeof(cuco::pair_type<Key, Value>));
        } else {
          auto const tmp = *reinterpret_cast<ulonglong4 const*>(current_slot);
          memcpy(&arr[0], &tmp, 2 * sizeof(cuco::pair_type<Key, Value>));
        }

        auto const first_slot_is_empty  = (arr[0].first == view.get_empty_key_sentinel());
        auto const second_slot_is_empty = (arr[1].first == view.get_empty_key_sentinel());

        auto const first_slot_equals  = (not first_slot_is_empty and pair_equal(arr[0], pair));
        auto const second_slot_equals = (not second_slot_is_empty and pair_equal(arr[1], pair));

        if (tile.any(first_slot_equals or second_slot_equals)) { found_match = true; }

        thread_num_matches += (first_slot_equals + second_slot_equals);

        if (tile.any(first_slot_is_empty or second_slot_is_empty)) {
          if ((not found_match) && (tile.thread_rank() == 0)) { thread_num_matches++; }
          break;
        }

        current_slot = view.next_slot(current_slot);
      }
    } else {
      while (true) {
        cuco::pair_type<Key, Value> arr[2];
        if constexpr (sizeof(Key) == 4) {
          auto const tmp = *reinterpret_cast<uint4 const*>(current_slot);
          memcpy(&arr[0], &tmp, 2 * sizeof(cuco::pair_type<Key, Value>));
        } else {
          auto const tmp = *reinterpret_cast<ulonglong4 const*>(current_slot);
          memcpy(&arr[0], &tmp, 2 * sizeof(cuco::pair_type<Key, Value>));
        }

        auto const first_slot_is_empty  = (arr[0].first == view.get_empty_key_sentinel());
        auto const second_slot_is_empty = (arr[1].first == view.get_empty_key_sentinel());

        auto const first_slot_equals  = (not first_slot_is_empty and pair_equal(arr[0], pair));
        auto const second_slot_equals = (not second_slot_is_empty and pair_equal(arr[1], pair));

        thread_num_matches += (first_slot_equals + second_slot_equals);

        if (tile.any(first_slot_is_empty or second_slot_is_empty)) { break; }

        current_slot = view.next_slot(current_slot);
      }
    }
    pair_idx += (gridDim.x * block_size) / tile_size;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  std::size_t block_num_matches = BlockReduce(temp_storage).Sum(thread_num_matches);
  if (threadIdx.x == 0) {
    num_matches->fetch_add(block_num_matches, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Counts the occurrences of key/value pairs in `[first, last)` contained in the multimap.
 * If no matches can be found for a given key/value pair, the corresponding occurrence is 1.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform counts
 * @tparam Key key type
 * @tparam Value The type of the mapped value for the map
 * @tparam is_vector_load Boolean flag indicating whether vector loads are used or not
 * @tparam is_outer Boolean flag indicating whether the current functions is used for outer join
 * operations or not
 * @tparam Input Device accesible input iterator of key/value pairs
 * @tparam atomicT Type of atomic storage
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam PairEqual Binary callable
 * @param first Beginning of the sequence of pairs to count
 * @param last End of the sequence of pairs to count
 * @param num_matches The number of all the matches for a sequence of pairs
 * @param view Device view used to access the hash map's slot storage
 * @param pair_equal Binary function to compare two pairs for equality
 */
template <uint32_t block_size,
          uint32_t tile_size,
          typename Key,
          typename Value,
          bool is_vector_load,
          bool is_outer = false,
          typename InputIt,
          typename atomicT,
          typename viewT,
          typename PairEqual>
__global__ std::enable_if_t<not is_vector_load, void> pair_count(
  InputIt first, InputIt last, atomicT* num_matches, viewT view, PairEqual pair_equal)
{
  auto tile     = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid      = block_size * blockIdx.x + threadIdx.x;
  auto pair_idx = tid / tile_size;

  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_matches = 0;

  while (first + pair_idx < last) {
    cuco::pair_type<Key, Value> pair = *(first + pair_idx);
    auto key                         = pair.first;
    auto current_slot                = view.initial_slot(tile, key);

    if constexpr (is_outer) {
      bool found_match = false;

      while (true) {
        auto slot_contents = *reinterpret_cast<cuco::pair_type<Key, Value> const*>(current_slot);

        auto const slot_is_empty = (slot_contents.first == view.get_empty_key_sentinel());

        auto const equals = not slot_is_empty and pair_equal(slot_contents, pair);

        if (tile.any(equals)) { found_match = true; }

        thread_num_matches += equals;

        if (tile.any(slot_is_empty)) {
          if ((not found_match) && (tile.thread_rank() == 0)) { thread_num_matches++; }
          break;
        }

        current_slot = view.next_slot(current_slot);
      }
    } else {
      while (true) {
        auto slot_contents = *reinterpret_cast<cuco::pair_type<Key, Value> const*>(current_slot);

        auto const slot_is_empty = (slot_contents.first == view.get_empty_key_sentinel());

        auto const equals = not slot_is_empty and pair_equal(slot_contents, pair);

        thread_num_matches += equals;

        if (tile.any(slot_is_empty)) { break; }

        current_slot = view.next_slot(current_slot);
      }
    }
    pair_idx += (gridDim.x * block_size) / tile_size;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  std::size_t block_num_matches = BlockReduce(temp_storage).Sum(thread_num_matches);
  if (threadIdx.x == 0) {
    num_matches->fetch_add(block_num_matches, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Finds all the values corresponding to all keys in the range `[first, last)` using vector
 * oads combined with per-block shared memory buffer.
 *
 * If the key `k = *(first + i)` exists in the map, copies `k` and all associated values to
 * unspecified locations in `[output_begin, output_begin + *num_matches - 1)`. Else, copies `k` and
 * the empty value sentinel.
 *
 * Behavior is undefined if the total number of matching keys exceeds `std::distance(output_begin,
 * output_begin + *num_matches - 1)`. Use `count()` to determine the number of matching keys.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups
 * @tparam buffer_size Size of the output buffer
 * @tparam Key key type
 * @tparam Value The type of the mapped value for the map
 * @tparam is_vector_load Boolean flag indicating whether vector loads are used or not
 * @tparam is_outer Boolean flag indicating whether the current functions is used for outer join
 * operations or not
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam atomicT Type of atomic storage
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of values retrieved for each key
 * @param num_matches Size of the output sequence
 * @param view Device view used to access the hash map's slot storage
 * @param key_equal The binary function to compare two keys for equality
 */
template <uint32_t block_size,
          uint32_t tile_size,
          uint32_t buffer_size,
          typename Key,
          typename Value,
          bool is_vector_load,
          bool is_outer = false,
          typename InputIt,
          typename OutputIt,
          typename atomicT,
          typename viewT,
          typename KeyEqual>
__global__ std::enable_if_t<is_vector_load, void> retrieve(InputIt first,
                                                           InputIt last,
                                                           OutputIt output_begin,
                                                           atomicT* num_matches,
                                                           viewT view,
                                                           KeyEqual key_equal)
{
  auto tile    = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid     = block_size * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;

  constexpr uint32_t num_warps = block_size / 32;
  const uint32_t warp_id       = threadIdx.x / 32;
  const uint32_t warp_lane_id  = threadIdx.x % 32;
  const uint32_t tile_lane_id  = tile.thread_rank();

  __shared__ cuco::pair_type<Key, Value> output_buffer[num_warps][buffer_size];
  __shared__ uint32_t warp_counter[num_warps];

  if (warp_lane_id == 0) { warp_counter[warp_id] = 0; }

  const unsigned int activemask = __ballot_sync(0xffffffff, first + key_idx < last);

  while (first + key_idx < last) {
    auto key          = *(first + key_idx);
    auto current_slot = view.initial_slot(tile, key);

    bool running = true;
    if constexpr (is_outer) {
      bool found_match = false;

      while (__any_sync(activemask, running)) {
        if (running) {
          pair<Key, Value> arr[2];
          if constexpr (sizeof(Key) == 4) {
            auto const tmp = *reinterpret_cast<uint4 const*>(current_slot);
            memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
          } else {
            auto const tmp = *reinterpret_cast<ulonglong4 const*>(current_slot);
            memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
          }

          auto const first_slot_is_empty  = (arr[0].first == view.get_empty_key_sentinel());
          auto const second_slot_is_empty = (arr[1].first == view.get_empty_key_sentinel());
          auto const first_equals  = (not first_slot_is_empty and key_equal(arr[0].first, key));
          auto const second_equals = (not second_slot_is_empty and key_equal(arr[1].first, key));
          auto const first_exists  = tile.ballot(first_equals);
          auto const second_exists = tile.ballot(second_equals);

          if (first_exists or second_exists) {
            found_match = true;

            auto num_first_matches  = __popc(first_exists);
            auto num_second_matches = __popc(second_exists);

            uint32_t output_idx;
            if (0 == tile_lane_id) {
              output_idx =
                atomicAdd(&warp_counter[warp_id], (num_first_matches + num_second_matches));
            }
            output_idx = tile.shfl(output_idx, 0);

            if (first_equals) {
              auto lane_offset = __popc(first_exists & ((1 << tile_lane_id) - 1));
              Key k            = key;
              output_buffer[warp_id][output_idx + lane_offset] =
                cuco::make_pair<Key, Value>(std::move(k), std::move(arr[0].second));
            }
            if (second_equals) {
              auto lane_offset = __popc(second_exists & ((1 << tile_lane_id) - 1));
              Key k            = key;
              output_buffer[warp_id][output_idx + num_first_matches + lane_offset] =
                cuco::make_pair<Key, Value>(std::move(k), std::move(arr[1].second));
            }
          }
          if (tile.any(first_slot_is_empty or second_slot_is_empty)) {
            running = false;
            if ((not found_match) && (tile_lane_id == 0)) {
              auto output_idx = atomicAdd(&warp_counter[warp_id], 1);
              output_buffer[warp_id][output_idx] =
                cuco::make_pair<Key, Value>(key, view.get_empty_key_sentinel());
            }
          }
        }  // if running

        __syncwarp(activemask);
        if (warp_counter[warp_id] + 32 * 2 > buffer_size) {
          flush_output_buffer(
            activemask, warp_counter[warp_id], output_buffer[warp_id], num_matches, output_begin);
          // First lane reset warp-level counter
          if (warp_lane_id == 0) { warp_counter[warp_id] = 0; }
        }

        current_slot = view.next_slot(current_slot);
      }  // while running
    } else {
      while (__any_sync(activemask, running)) {
        if (running) {
          pair<Key, Value> arr[2];
          if constexpr (sizeof(Key) == 4) {
            auto const tmp = *reinterpret_cast<uint4 const*>(current_slot);
            memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
          } else {
            auto const tmp = *reinterpret_cast<ulonglong4 const*>(current_slot);
            memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
          }

          auto const first_slot_is_empty  = (arr[0].first == view.get_empty_key_sentinel());
          auto const second_slot_is_empty = (arr[1].first == view.get_empty_key_sentinel());
          auto const first_equals  = (not first_slot_is_empty and key_equal(arr[0].first, key));
          auto const second_equals = (not second_slot_is_empty and key_equal(arr[1].first, key));
          auto const first_exists  = tile.ballot(first_equals);
          auto const second_exists = tile.ballot(second_equals);

          if (first_exists or second_exists) {
            auto num_first_matches  = __popc(first_exists);
            auto num_second_matches = __popc(second_exists);

            uint32_t output_idx;
            if (0 == tile_lane_id) {
              output_idx =
                atomicAdd(&warp_counter[warp_id], (num_first_matches + num_second_matches));
            }
            output_idx = tile.shfl(output_idx, 0);

            if (first_equals) {
              auto lane_offset = __popc(first_exists & ((1 << tile_lane_id) - 1));
              Key k            = key;
              output_buffer[warp_id][output_idx + lane_offset] =
                cuco::make_pair<Key, Value>(std::move(k), std::move(arr[0].second));
            }
            if (second_equals) {
              auto lane_offset = __popc(second_exists & ((1 << tile_lane_id) - 1));
              Key k            = key;
              output_buffer[warp_id][output_idx + num_first_matches + lane_offset] =
                cuco::make_pair<Key, Value>(std::move(k), std::move(arr[1].second));
            }
          }
          if (tile.any(first_slot_is_empty or second_slot_is_empty)) { running = false; }
        }  // if running

        __syncwarp(activemask);
        if (warp_counter[warp_id] + 32 * 2 > buffer_size) {
          flush_output_buffer(
            activemask, warp_counter[warp_id], output_buffer[warp_id], num_matches, output_begin);
          // First lane reset warp-level counter
          if (warp_lane_id == 0) { warp_counter[warp_id] = 0; }
        }

        current_slot = view.next_slot(current_slot);
      }  // while running
    }
    key_idx += (gridDim.x * block_size) / tile_size;
  }

  // Final flush of output buffer
  if (warp_counter[warp_id] > 0) {
    flush_output_buffer(
      activemask, warp_counter[warp_id], output_buffer[warp_id], num_matches, output_begin);
  }
}

/**
 * @brief Finds all the values corresponding to all keys in the range `[first, last)` using scalar
 * loads combined with per-CG shared memory buffer.
 *
 * If the key `k = *(first + i)` exists in the map, copies `k` and all associated values to
 * unspecified locations in `[output_begin, output_begin + *num_matches - 1)`. Else, copies `k` and
 * the empty value sentinel.
 *
 * Behavior is undefined if the total number of matching keys exceeds `std::distance(output_begin,
 * output_begin + *num_matches - 1)`. Use `count()` to determine the number of matching keys.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups
 * @tparam buffer_size Size of the output buffer
 * @tparam Key key type
 * @tparam Value The type of the mapped value for the map
 * @tparam is_vector_load Boolean flag indicating whether vector loads are used or not
 * @tparam is_outer Boolean flag indicating whether the current functions is used for outer join
 * operations or not
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam atomicT Type of atomic storage
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of values retrieved for each key
 * @param num_matches Size of the output sequence
 * @param view Device view used to access the hash map's slot storage
 * @param key_equal The binary function to compare two keys for equality
 */
template <uint32_t block_size,
          uint32_t tile_size,
          uint32_t buffer_size,
          typename Key,
          typename Value,
          bool is_vector_load,
          bool is_outer = false,
          typename InputIt,
          typename OutputIt,
          typename atomicT,
          typename viewT,
          typename KeyEqual>
__global__ std::enable_if_t<not is_vector_load, void> retrieve(InputIt first,
                                                               InputIt last,
                                                               OutputIt output_begin,
                                                               atomicT* num_matches,
                                                               viewT view,
                                                               KeyEqual key_equal)
{
  auto tile    = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid     = block_size * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;

  constexpr uint32_t num_cgs = block_size / tile_size;
  const uint32_t cg_id       = threadIdx.x / tile_size;
  const uint32_t lane_id     = tile.thread_rank();

  __shared__ cuco::pair_type<Key, Value> output_buffer[num_cgs][buffer_size];
  __shared__ uint32_t cg_counter[num_cgs];

  if (lane_id == 0) { cg_counter[cg_id] = 0; }

  while (first + key_idx < last) {
    auto key          = *(first + key_idx);
    auto current_slot = view.initial_slot(tile, key);

    bool running = true;

    if constexpr (is_outer) {
      bool found_match = false;

      while (running) {
        // TODO: Replace reinterpret_cast with atomic ref when possible. The current implementation
        // is unsafe!
        static_assert(sizeof(Key) == sizeof(cuda::atomic<Key>));
        static_assert(sizeof(Value) == sizeof(cuda::atomic<Value>));
        pair<Key, Value> slot_contents = *reinterpret_cast<pair<Key, Value> const*>(current_slot);

        auto const slot_is_empty = (slot_contents.first == view.get_empty_key_sentinel());
        auto const equals        = (not slot_is_empty and key_equal(slot_contents.first, key));
        auto const exists        = tile.ballot(equals);

        if (exists) {
          found_match         = true;
          auto num_matches    = __popc(exists);
          uint32_t output_idx = cg_counter[cg_id];
          if (equals) {
            // Each match computes its lane-level offset
            auto lane_offset = __popc(exists & ((1 << lane_id) - 1));
            Key k            = key;
            output_buffer[cg_id][output_idx + lane_offset] =
              cuco::make_pair<Key, Value>(std::move(k), std::move(slot_contents.second));
          }
          if (0 == lane_id) { cg_counter[cg_id] += num_matches; }
        }
        if (tile.any(slot_is_empty)) {
          running = false;
          if ((not found_match) && (lane_id == 0)) {
            auto output_idx = cg_counter[cg_id]++;
            output_buffer[cg_id][output_idx] =
              cuco::make_pair<Key, Value>(key, view.get_empty_key_sentinel());
          }
        }

        tile.sync();

        // Flush if the next iteration won't fit into buffer
        if ((cg_counter[cg_id] + tile_size) > buffer_size) {
          flush_output_buffer<tile_size>(
            tile, cg_counter[cg_id], output_buffer[cg_id], num_matches, output_begin);
          // First lane reset CG-level counter
          if (lane_id == 0) { cg_counter[cg_id] = 0; }
        }
        current_slot = view.next_slot(current_slot);
      }  // while running
    } else {
      while (running) {
        // TODO: Replace reinterpret_cast with atomic ref when possible. The current implementation
        // is unsafe!
        static_assert(sizeof(Key) == sizeof(cuda::atomic<Key>));
        static_assert(sizeof(Value) == sizeof(cuda::atomic<Value>));
        pair<Key, Value> slot_contents = *reinterpret_cast<pair<Key, Value> const*>(current_slot);

        auto const slot_is_empty = (slot_contents.first == view.get_empty_key_sentinel());
        auto const equals        = (not slot_is_empty and key_equal(slot_contents.first, key));
        auto const exists        = tile.ballot(equals);

        if (exists) {
          auto num_matches    = __popc(exists);
          uint32_t output_idx = cg_counter[cg_id];
          if (equals) {
            // Each match computes its lane-level offset
            auto lane_offset = __popc(exists & ((1 << lane_id) - 1));
            Key k            = key;
            output_buffer[cg_id][output_idx + lane_offset] =
              cuco::make_pair<Key, Value>(std::move(k), std::move(slot_contents.second));
          }
          if (0 == lane_id) { cg_counter[cg_id] += num_matches; }
        }
        if (tile.any(slot_is_empty)) { running = false; }

        tile.sync();

        // Flush if the next iteration won't fit into buffer
        if ((cg_counter[cg_id] + tile_size) > buffer_size) {
          flush_output_buffer<tile_size>(
            tile, cg_counter[cg_id], output_buffer[cg_id], num_matches, output_begin);
          // First lane reset CG-level counter
          if (lane_id == 0) { cg_counter[cg_id] = 0; }
        }
        current_slot = view.next_slot(current_slot);
      }  // while running
    }
    key_idx += (gridDim.x * block_size) / tile_size;
  }

  // Final flush of output buffer
  if (cg_counter[cg_id] > 0) {
    flush_output_buffer<tile_size>(
      tile, cg_counter[cg_id], output_buffer[cg_id], num_matches, output_begin);
  }
}

}  // namespace detail
}  // namespace cuco
