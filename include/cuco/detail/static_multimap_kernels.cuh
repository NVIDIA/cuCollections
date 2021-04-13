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

#include <cuco/detail/pair.cuh>

#include <cooperative_groups/memcpy_async.h>

namespace cuco {
namespace detail {
namespace cg = cooperative_groups;

/**
 * @brief Flushes shared memory buffer into the output sequence.
 *
 * @tparam Key key type
 * @tparam Value value type
 * @tparam atomicT Type of atomic storage
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @param output_size Number of valid output in the buffer
 * @param output_buffer Shared memory buffer of the key/value pair sequence
 * @param num_items Size of the output sequence
 * @param output_begin Beginning of the output sequence of key/value pairs
 */
template <uint32_t cg_size,
          typename CG,
          typename Key,
          typename Value,
          typename atomicT,
          typename OutputIt>
__inline__ __device__ std::enable_if_t<std::is_pointer<OutputIt>::value, void> flush_output_buffer(
  CG const& g,
  uint32_t const num_outputs,
  cuco::pair_type<Key, Value>* output_buffer,
  atomicT* num_items,
  OutputIt output_begin)
{
  std::size_t offset;
  const auto lane_id = g.thread_rank();
  if (0 == lane_id) { offset = num_items->fetch_add(num_outputs, cuda::std::memory_order_relaxed); }
  offset = g.shfl(offset, 0);

  cg::memcpy_async(g,
                   output_begin + offset,
                   output_buffer,
                   cuda::aligned_size_t<16>(sizeof(cuco::pair_type<Key, Value>) * num_outputs));
}

/**
 * @brief Flushes shared memory buffer into the output sequence.
 *
 * @tparam Key key type
 * @tparam Value value type
 * @tparam atomicT Type of atomic storage
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @param output_size Number of valid output in the buffer
 * @param output_buffer Shared memory buffer of the key/value pair sequence
 * @param num_items Size of the output sequence
 * @param output_begin Beginning of the output sequence of key/value pairs
 */
template <uint32_t cg_size,
          typename CG,
          typename Key,
          typename Value,
          typename atomicT,
          typename OutputIt>
__inline__ __device__ std::enable_if_t<not std::is_pointer<OutputIt>::value, void>
flush_output_buffer(CG const& g,
                    uint32_t const num_outputs,
                    cuco::pair_type<Key, Value>* output_buffer,
                    atomicT* num_items,
                    OutputIt output_begin)
{
  std::size_t offset;
  const auto lane_id = g.thread_rank();
  if (0 == lane_id) { offset = num_items->fetch_add(num_outputs, cuda::std::memory_order_relaxed); }
  offset = g.shfl(offset, 0);

  for (auto index = lane_id; index < num_outputs; index += cg_size) {
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
 * element is inserted.
 *
 * @tparam block_size
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `value_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of key/value pairs
 * @param last End of the sequence of key/value pairs
 * @param view Mutable device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function used to compare two keys for equality
 */
template <uint32_t block_size, typename InputIt, typename viewT, typename Hash, typename KeyEqual>
__global__ void insert(InputIt first, InputIt last, viewT view, Hash hash, KeyEqual key_equal)
{
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto it  = first + tid;

  while (it < last) {
    typename viewT::value_type const insert_pair{*it};
    view.insert(insert_pair, hash, key_equal);
    it += gridDim.x * blockDim.x;
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
 * @tparam block_size
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform
 * inserts
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `value_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of key/value pairs
 * @param last End of the sequence of key/value pairs
 * @param view Mutable device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function used to compare two keys for equality
 */
template <uint32_t block_size,
          uint32_t tile_size,
          typename InputIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
__global__ void insert(InputIt first, InputIt last, viewT view, Hash hash, KeyEqual key_equal)
{
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid  = blockDim.x * blockIdx.x + threadIdx.x;
  auto it   = first + tid / tile_size;

  while (it < last) {
    // force conversion to value_type
    typename viewT::value_type const insert_pair{*it};
    view.insert(tile, insert_pair, hash, key_equal);
    it += (gridDim.x * blockDim.x) / tile_size;
  }
}

/**
 * @brief Finds the values corresponding to all keys in the range `[first, last)`.
 *
 * If the key `*(first + i)` exists in the map, copies its associated value to `(output_begin + i)`.
 * Else, copies the empty value sentinel.
 * @tparam block_size The size of the thread block
 * @tparam Value The type of the mapped value for the map
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of values retrieved for each key
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template <uint32_t block_size,
          typename Value,
          typename InputIt,
          typename OutputIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
__global__ void find(
  InputIt first, InputIt last, OutputIt output_begin, viewT view, Hash hash, KeyEqual key_equal)
{
  auto tid     = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid;
  __shared__ Value writeBuffer[block_size];

  while (first + key_idx < last) {
    auto key   = *(first + key_idx);
    auto found = view.find(key, hash, key_equal);

    /*
     * The ld.relaxed.gpu instruction used in view.find causes L1 to
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    writeBuffer[threadIdx.x] = found->second.load(cuda::std::memory_order_relaxed);
    __syncthreads();
    *(output_begin + key_idx) = writeBuffer[threadIdx.x];
    key_idx += gridDim.x * blockDim.x;
  }
}

/**
 * @brief Finds the values corresponding to all keys in the range `[first, last)`.
 *
 * If the key `*(first + i)` exists in the map, copies its associated value to `(output_begin + i)`.
 * Else, copies the empty value sentinel. Uses the CUDA Cooperative Groups API to leverage groups
 * of multiple threads to find each key. This provides a significant boost in throughput compared
 * to the non Cooperative Group `find` at moderate to high load factors.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform
 * inserts
 * @tparam Value The type of the mapped value for the map
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of values retrieved for each key
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template <uint32_t block_size,
          uint32_t tile_size,
          typename Value,
          typename InputIt,
          typename OutputIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
__global__ void find(
  InputIt first, InputIt last, OutputIt output_begin, viewT view, Hash hash, KeyEqual key_equal)
{
  auto tile    = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid     = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  __shared__ Value writeBuffer[block_size];

  while (first + key_idx < last) {
    auto key   = *(first + key_idx);
    auto found = view.find(tile, key, hash, key_equal);

    /*
     * The ld.relaxed.gpu instruction used in view.find causes L1 to
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    if (tile.thread_rank() == 0) {
      writeBuffer[threadIdx.x / tile_size] = found->second.load(cuda::std::memory_order_relaxed);
    }
    __syncthreads();
    if (tile.thread_rank() == 0) {
      *(output_begin + key_idx) = writeBuffer[threadIdx.x / tile_size];
    }
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }
}

/**
 * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
 *
 * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
 *
 * @tparam block_size The size of the thread block
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template <uint32_t block_size,
          typename InputIt,
          typename OutputIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
__global__ void contains(
  InputIt first, InputIt last, OutputIt output_begin, viewT view, Hash hash, KeyEqual key_equal)
{
  auto tid     = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid;
  __shared__ bool writeBuffer[block_size];

  while (first + key_idx < last) {
    auto key = *(first + key_idx);

    /*
     * The ld.relaxed.gpu instruction used in view.find causes L1 to
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    writeBuffer[threadIdx.x] = view.contains(key, hash, key_equal);
    __syncthreads();
    *(output_begin + key_idx) = writeBuffer[threadIdx.x];
    key_idx += gridDim.x * blockDim.x;
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
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform
 * inserts
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template <uint32_t block_size,
          uint32_t tile_size,
          typename InputIt,
          typename OutputIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
__global__ void contains(
  InputIt first, InputIt last, OutputIt output_begin, viewT view, Hash hash, KeyEqual key_equal)
{
  auto tile    = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid     = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  __shared__ bool writeBuffer[block_size];

  while (first + key_idx < last) {
    auto key   = *(first + key_idx);
    auto found = view.contains(tile, key, hash, key_equal);

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
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }
}

/**
 * @brief Finds all the values corresponding to all keys in the range `[first, last)`.
 *
 * If the key `k = *(first + i)` exists in the map, copies `k` and all associated values to
 * unspecified locations in `[output_begin, output_begin + *num_items - 1)`. Else, copies `k` and
 * the empty value sentinel.
 *
 * Behavior is undefined if the total number of matching keys exceeds `std::distance(output_begin,
 * output_begin + *num_items - 1)`. Use `count()` to determine the number of matching keys.
 *
 * @tparam block_size The size of the thread block
 * @tparam buffer_size Size of the output buffer
 * @tparam Key key type
 * @tparam Value The type of the mapped value for the map
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam atomicT Type of atomic storage
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of values retrieved for each key
 * @param num_items Size of the output sequence
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template <uint32_t block_size,
          uint32_t buffer_size,
          typename Key,
          typename Value,
          typename InputIt,
          typename OutputIt,
          typename atomicT,
          typename viewT,
          typename Hash,
          typename KeyEqual>
__global__ void find_all(InputIt first,
                         InputIt last,
                         OutputIt output_begin,
                         atomicT* num_items,
                         viewT view,
                         Hash hash,
                         KeyEqual key_equal)
{
  auto tid     = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid;

  constexpr uint32_t step = 1;
  auto const end          = view.end();

  __shared__ cuco::pair_type<Key, Value> output_buffer[buffer_size];
  __shared__ uint32_t block_counter;  // TODO: do we really need uint32_t?

  if (0 == threadIdx.x) { block_counter = 0; }

  if (first + key_idx < last) {
    auto key   = *(first + key_idx);
    auto found = view.find_all(key, hash, key_equal);

    bool running     = true;
    bool found_match = false;

    while (__syncthreads_or(running)) {
      if (running) {
        if (found == end) {
          running = false;
        } else {
          found_match = true;

          auto index           = atomicAdd_block(&block_counter, step);
          output_buffer[index] = cuco::make_pair<Key, Value>(key, (*found).second);

          ++found;
        }

        if ((not running) && (not found_match)) {
          auto index           = atomicAdd_block(&block_counter, step);
          output_buffer[index] = cuco::make_pair<Key, Value>(key, view.get_empty_value_sentinel());
        }
      }  // if (running)

      __syncthreads();

      if ((block_counter + block_size) > buffer_size) {
        flush_output_buffer<Key, Value, atomicT, OutputIt>(
          block_counter, output_buffer, num_items, output_begin);
        __syncthreads();
        if (0 == threadIdx.x) { block_counter = 0; }
        __syncthreads();
      }
    }  // while syncthreads_or
  }

  // Final flush of output cache
  if (block_counter > 0) {
    flush_output_buffer<Key, Value, atomicT, OutputIt>(
      block_counter, output_buffer, num_items, output_begin);
  }
}

/**
 * @brief Finds all the values corresponding to all keys in the range `[first, last)`.
 *
 * If the key `k = *(first + i)` exists in the map, copies `k` and all associated values to
 * unspecified locations in `[output_begin, output_begin + *num_items - 1)`. Else, copies `k` and
 * the empty value sentinel.
 *
 * Behavior is undefined if the total number of matching keys exceeds `std::distance(output_begin,
 * output_begin + *num_items - 1)`. Use `count()` to determine the number of matching keys.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform
 * inserts
 * @tparam buffer_size Size of the output buffer
 * @tparam Key key type
 * @tparam Value The type of the mapped value for the map
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam atomicT Type of atomic storage
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of values retrieved for each key
 * @param num_items Size of the output sequence
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */

template <uint32_t block_size,
          uint32_t tile_size,
          uint32_t buffer_size,
          typename Key,
          typename Value,
          typename InputIt,
          typename OutputIt,
          typename atomicT,
          typename viewT,
          typename Hash,
          typename KeyEqual>
__global__ void find_all(InputIt first,
                         InputIt last,
                         OutputIt output_begin,
                         atomicT* num_items,
                         viewT view,
                         Hash hash,
                         KeyEqual key_equal)
{
  auto tile    = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid     = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;

  constexpr uint32_t num_cgs = block_size / tile_size;
  const uint32_t cg_id       = threadIdx.x / tile_size;
  const uint32_t lane_id     = tile.thread_rank();

  __shared__ cuco::pair_type<Key, Value> output_buffer[num_cgs][buffer_size];
  __shared__ uint32_t cg_counter[num_cgs];

  if (lane_id == 0) { cg_counter[cg_id] = 0; }

  while (first + key_idx < last) {
    auto key          = *(first + key_idx);
    auto current_slot = view.initial_slot(tile, key, hash);

    bool running     = true;
    bool found_match = false;

    while (tile.any(running)) {
      // TODO: Replace reinterpret_cast with atomic ref when possible. The current implementation is
      // unsafe!
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
      } else if (tile.any(slot_is_empty)) {
        running = false;
        if ((not found_match) && (lane_id == 0)) {
          auto output_idx = cg_counter[cg_id]++;
          output_buffer[cg_id][output_idx] =
            cuco::make_pair<Key, Value>(key, view.get_empty_key_sentinel());
        }
      }

      tile.sync();

      // Flush it the next iteration won't fit into buffer
      if ((cg_counter[cg_id] + tile_size) > buffer_size) {
        flush_output_buffer<tile_size>(
          tile, cg_counter[cg_id], output_buffer[cg_id], num_items, output_begin);
        // First lane reset CG-level counter
        if (lane_id == 0) { cg_counter[cg_id] = 0; }
      }
      current_slot = view.next_slot(tile, current_slot);
    }  // while running
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }

  // Final flush of output buffer
  if (cg_counter[cg_id] > 0) {
    flush_output_buffer<tile_size>(
      tile, cg_counter[cg_id], output_buffer[cg_id], num_items, output_begin);
  }
}

/**
 * @brief Counts the occurrences of keys in `[first, last)` contained in the multimap.
 *
 * @tparam block_size The size of the thread block
 * @tparam Value The type of the mapped value for the map
 * @tparam InputIt Device accessible input iterator whose `value_type` is convertible to the map's
 * `key_type`
 * @tparam atomicT Type of atomic storage
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable
 * @tparam KeyEqual Binary callable
 * @param first Beginning of the sequence of keys to count
 * @param last End of the sequence of keys to count
 * @param num_items The number of all the matches for a sequence of keys
 * @param view Device view used to access the hash map's slot storage
 * @param hash Unary function to apply to hash each key
 * @param key_equal Binary function to compare two keys for equality
 */
template <uint32_t block_size,
          typename Value,
          typename InputIt,
          typename atomicT,
          typename viewT,
          typename Hash,
          typename KeyEqual>
__global__ void count(
  InputIt first, InputIt last, atomicT* num_items, viewT view, Hash hash, KeyEqual key_equal)
{
  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_items = 0;

  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto it  = first + tid;

  while (it < last) {
    thread_num_items += view.count(*it, hash, key_equal);
    it += gridDim.x * blockDim.x;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  std::size_t block_num_items = BlockReduce(temp_storage).Sum(thread_num_items);
  if (threadIdx.x == 0) { *num_items += block_num_items; }
}

/**
 * @brief Counts the occurrences of keys in `[first, last)` contained in the multimap.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform inserts
 * @tparam Value The type of the mapped value for the map
 * @tparam InputIt Device accessible input iterator whose `value_type` is convertible to the map's
 * `key_type`
 * @tparam atomicT Type of atomic storage
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable
 * @tparam KeyEqual Binary callable
 * @param first Beginning of the sequence of keys to count
 * @param last End of the sequence of keys to count
 * @param num_items The number of all the matches for a sequence of keys
 * @param view Device view used to access the hash map's slot storage
 * @param hash Unary function to apply to hash each key
 * @param key_equal Binary function to compare two keys for equality
 */
template <uint32_t block_size,
          uint32_t tile_size,
          typename Value,
          typename InputIt,
          typename atomicT,
          typename viewT,
          typename Hash,
          typename KeyEqual>
__global__ void count(
  InputIt first, InputIt last, atomicT* num_items, viewT view, Hash hash, KeyEqual key_equal)
{
  auto tile    = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid     = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;

  const int lane_id = tile.thread_rank();

  while (first + key_idx < last) {
    auto key          = *(first + key_idx);
    auto current_slot = view.initial_slot(tile, key, hash);

    bool running = true;

    while (tile.any(running)) {
      auto const current_key = current_slot->first.load(cuda::std::memory_order_relaxed);

      auto const slot_is_empty = (current_key == view.get_empty_key_sentinel());
      auto const equals        = (not slot_is_empty and key_equal(current_key, key));
      auto const exists        = tile.ballot(equals);

      if (exists) {
        auto num_matches = __popc(exists);
        // First lane gets the CG-level offset
        if (0 == lane_id) { num_items->fetch_add(num_matches, cuda::std::memory_order_relaxed); }
      } else if (tile.any(slot_is_empty)) {
        running = false;
      }
      current_slot = view.next_slot(tile, current_slot);
    }  // while running
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }
}

}  // namespace detail
}  // namespace cuco
