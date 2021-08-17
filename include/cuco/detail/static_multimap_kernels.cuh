/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cuco/detail/pair.cuh>

namespace cuco {
namespace detail {
namespace cg = cooperative_groups;

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
 * Uses the CUDA Cooperative Groups API to leverage groups of multiple threads to perform each
 * key/value insertion. This provides a significant boost in throughput compared to the non
 * Cooperative Group `insert` at moderate to high load factors.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform
 * inserts
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
    view.insert(tile, insert_pair, thrust::nullopt, key_equal);
    it += (gridDim.x * block_size) / tile_size;
  }
}

/**
 * @brief Inserts key/value pairs in the range `[first, last)` if `pred` returns true.
 *
 * Uses the CUDA Cooperative Groups API to leverage groups of multiple threads to perform each
 * key/value insertion. This provides a significant boost in throughput compared to the non
 * Cooperative Group `insert` at moderate to high load factors.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform
 * inserts
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `value_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Predicate Unary predicate function type
 * @tparam KeyEqual Binary callable type
 * @param first Beginning of the sequence of key/value pairs
 * @param last End of the sequence of key/value pairs
 * @param view Mutable device view used to access the hash map's slot storage
 * @param key_equal The binary function used to compare two keys for equality
 */
template <uint32_t block_size,
          uint32_t tile_size,
          typename InputIt,
          typename viewT,
          typename Predicate,
          typename KeyEqual>
__global__ void insert_if(
  InputIt first, InputIt last, viewT view, Predicate pred, KeyEqual key_equal)
{
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid  = block_size * blockIdx.x + threadIdx.x;
  auto it   = first + tid / tile_size;

  while (it < last) {
    typename viewT::value_type const insert_pair{*it};
    if (pred(insert_pair)) {
      // force conversion to value_type
      view.insert(tile, insert_pair, thrust::nullopt, key_equal);
    }
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
    auto found = view.contains(tile, key, thrust::nullopt, key_equal);

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
 * @brief Counts the occurrences of keys in `[first, last)` contained in the multimap. If `is_outer`
 * is true, the corresponding occurrence for non-matches is 1. Otherwise, it's 0.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform counts
 * @tparam uses_vector_load Boolean flag indicating whether vector loads are used or not
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
          bool is_outer,
          typename InputIt,
          typename atomicT,
          typename viewT,
          typename KeyEqual>
__global__ void count(
  InputIt first, InputIt last, atomicT* num_matches, viewT view, KeyEqual key_equal)
{
  auto tile    = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid     = block_size * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;

  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_matches = 0;

  while (first + key_idx < last) {
    auto key = *(first + key_idx);
    if constexpr (is_outer) {
      view.count_outer(tile, key, thrust::nullopt, thread_num_matches, key_equal);
    } else {
      view.count(tile, key, thrust::nullopt, thread_num_matches, key_equal);
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
 * @brief Counts the occurrences of key/value pairs in `[first, last)` contained in the multimap.
 * If no matches can be found for a given key/value pair and `is_outer` is true, the corresponding
 * occurrence is 1.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform counts
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
          bool is_outer,
          typename InputIt,
          typename atomicT,
          typename viewT,
          typename PairEqual>
__global__ void pair_count(
  InputIt first, InputIt last, atomicT* num_matches, viewT view, PairEqual pair_equal)
{
  auto tile     = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid      = block_size * blockIdx.x + threadIdx.x;
  auto pair_idx = tid / tile_size;

  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_matches = 0;

  while (first + pair_idx < last) {
    typename viewT::value_type const pair = *(first + pair_idx);
    if constexpr (is_outer) {
      view.pair_count_outer(tile, pair, thrust::nullopt, thread_num_matches, pair_equal);
    } else {
      view.pair_count(tile, pair, thrust::nullopt, thread_num_matches, pair_equal);
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
 * @tparam warp_size The size of the warp
 * @tparam tile_size The number of threads in the Cooperative Groups
 * @tparam buffer_size Size of the output buffer
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
          uint32_t warp_size,
          uint32_t tile_size,
          uint32_t buffer_size,
          bool is_outer,
          typename InputIt,
          typename OutputIt,
          typename atomicT,
          typename viewT,
          typename KeyEqual>
__global__ void vectorized_retrieve(InputIt first,
                                    InputIt last,
                                    OutputIt output_begin,
                                    atomicT* num_matches,
                                    viewT view,
                                    KeyEqual key_equal)
{
  using pair_type = typename viewT::value_type;

  constexpr uint32_t num_warps = block_size / warp_size;
  const uint32_t warp_id       = threadIdx.x / warp_size;

  auto warp    = cg::tiled_partition<warp_size>(cg::this_thread_block());
  auto tile    = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid     = block_size * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;

  __shared__ pair_type output_buffer[num_warps][buffer_size];
  // TODO: replace this with shared memory cuda::atomic variables once the dynamiic initialization
  // warning issue is solved __shared__ atomicT toto_countter[num_warps];
  __shared__ uint32_t warp_counter[num_warps];

  if (warp.thread_rank() == 0) { warp_counter[warp_id] = 0; }

  while (warp.any(first + key_idx < last)) {
    bool active_flag = first + key_idx < last;
    auto active_warp = cg::binary_partition<warp_size>(warp, active_flag);

    if (active_flag) {
      auto key = *(first + key_idx);
      if constexpr (is_outer) {
        view.retrieve_outer<buffer_size>(active_warp,
                                         tile,
                                         key,
                                         thrust::nullopt,
                                         &warp_counter[warp_id],
                                         output_buffer[warp_id],
                                         num_matches,
                                         output_begin,
                                         key_equal);
      } else {
        view.retrieve<buffer_size>(active_warp,
                                   tile,
                                   key,
                                   thrust::nullopt,
                                   &warp_counter[warp_id],
                                   output_buffer[warp_id],
                                   num_matches,
                                   output_begin,
                                   key_equal);
      }
    }
    key_idx += (gridDim.x * block_size) / tile_size;
  }

  // Final flush of output buffer
  if (warp_counter[warp_id] > 0) {
    view.flush_output_buffer(
      warp, warp_counter[warp_id], output_buffer[warp_id], num_matches, output_begin);
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
 * @tparam warp_size The size of the warp
 * @tparam tile_size The number of threads in the Cooperative Groups
 * @tparam buffer_size Size of the output buffer
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
          uint32_t warp_size,
          uint32_t tile_size,
          uint32_t buffer_size,
          bool is_outer,
          typename InputIt,
          typename OutputIt,
          typename atomicT,
          typename viewT,
          typename KeyEqual>
__global__ void retrieve(InputIt first,
                         InputIt last,
                         OutputIt output_begin,
                         atomicT* num_matches,
                         viewT view,
                         KeyEqual key_equal)
{
  using pair_type = typename viewT::value_type;

  auto tile    = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid     = block_size * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;

  constexpr uint32_t num_cgs = block_size / tile_size;
  const uint32_t cg_id       = threadIdx.x / tile_size;
  const uint32_t lane_id     = tile.thread_rank();

  __shared__ pair_type output_buffer[num_cgs][buffer_size];
  __shared__ uint32_t cg_counter[num_cgs];

  if (lane_id == 0) { cg_counter[cg_id] = 0; }

  while (first + key_idx < last) {
    auto key = *(first + key_idx);
    if constexpr (is_outer) {
      view.retrieve_outer<tile_size, buffer_size>(tile,
                                                  key,
                                                  thrust::nullopt,
                                                  &cg_counter[cg_id],
                                                  output_buffer[cg_id],
                                                  num_matches,
                                                  output_begin,
                                                  key_equal);
    } else {
      view.retrieve<tile_size, buffer_size>(tile,
                                            key,
                                            thrust::nullopt,
                                            &cg_counter[cg_id],
                                            output_buffer[cg_id],
                                            num_matches,
                                            output_begin,
                                            key_equal);
    }
    key_idx += (gridDim.x * block_size) / tile_size;
  }

  // Final flush of output buffer
  if (cg_counter[cg_id] > 0) {
    view.flush_output_buffer(
      tile, cg_counter[cg_id], output_buffer[cg_id], num_matches, output_begin);
  }
}

template <uint32_t block_size,
          uint32_t warp_size,
          uint32_t tile_size,
          uint32_t buffer_size,
          bool is_outer,
          typename InputIt,
          typename OutputIt1,
          typename OutputIt2,
          typename atomicT,
          typename viewT,
          typename PairEqual>
__global__ void vectorized_pair_retrieve(InputIt first,
                                         InputIt last,
                                         OutputIt1 probe_output_begin,
                                         OutputIt2 contained_output_begin,
                                         atomicT* num_matches,
                                         viewT view,
                                         PairEqual pair_equal)
{
  using pair_type = typename viewT::value_type;

  constexpr uint32_t num_warps = block_size / warp_size;
  const uint32_t warp_id       = threadIdx.x / warp_size;

  auto warp     = cg::tiled_partition<warp_size>(cg::this_thread_block());
  auto tile     = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid      = block_size * blockIdx.x + threadIdx.x;
  auto pair_idx = tid / tile_size;

  __shared__ pair_type probe_output_buffer[num_warps][buffer_size];
  __shared__ pair_type contained_output_buffer[num_warps][buffer_size];
  // TODO: replace this with shared memory cuda::atomic variables once the dynamiic initialization
  // warning issue is solved __shared__ atomicT toto_countter[num_warps];
  __shared__ uint32_t warp_counter[num_warps];

  if (warp.thread_rank() == 0) { warp_counter[warp_id] = 0; }

  while (warp.any(first + pair_idx < last)) {
    bool active_flag = first + pair_idx < last;
    auto active_warp = cg::binary_partition<warp_size>(warp, active_flag);

    if (active_flag) {
      pair_type pair = *(first + pair_idx);
      if constexpr (is_outer) {
        view.pair_retrieve_outer<buffer_size>(active_warp,
                                              tile,
                                              pair,
                                              thrust::nullopt,
                                              &warp_counter[warp_id],
                                              probe_output_buffer[warp_id],
                                              contained_output_buffer[warp_id],
                                              num_matches,
                                              probe_output_begin,
                                              contained_output_begin,
                                              pair_equal);
      } else {
        view.pair_retrieve<buffer_size>(active_warp,
                                        tile,
                                        pair,
                                        thrust::nullopt,
                                        &warp_counter[warp_id],
                                        probe_output_buffer[warp_id],
                                        contained_output_buffer[warp_id],
                                        num_matches,
                                        probe_output_begin,
                                        contained_output_begin,
                                        pair_equal);
      }
    }
    pair_idx += (gridDim.x * block_size) / tile_size;
  }

  // Final flush of output buffer
  if (warp_counter[warp_id] > 0) {
    view.flush_output_buffer(warp,
                             warp_counter[warp_id],
                             probe_output_buffer[warp_id],
                             contained_output_buffer[warp_id],
                             num_matches,
                             probe_output_begin,
                             contained_output_begin);
  }
}

template <uint32_t block_size,
          uint32_t warp_size,
          uint32_t tile_size,
          uint32_t buffer_size,
          bool is_outer,
          typename InputIt,
          typename OutputIt1,
          typename OutputIt2,
          typename atomicT,
          typename viewT,
          typename PairEqual>
__global__ void pair_retrieve(InputIt first,
                              InputIt last,
                              OutputIt1 probe_output_begin,
                              OutputIt2 contained_output_begin,
                              atomicT* num_matches,
                              viewT view,
                              PairEqual pair_equal)
{
  using pair_type = typename viewT::value_type;

  auto tile     = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid      = block_size * blockIdx.x + threadIdx.x;
  auto pair_idx = tid / tile_size;

  constexpr uint32_t num_cgs = block_size / tile_size;
  const uint32_t cg_id       = threadIdx.x / tile_size;
  const uint32_t lane_id     = tile.thread_rank();

  __shared__ pair_type probe_output_buffer[num_cgs][buffer_size];
  __shared__ pair_type contained_output_buffer[num_cgs][buffer_size];
  __shared__ uint32_t cg_counter[num_cgs];

  if (lane_id == 0) { cg_counter[cg_id] = 0; }

  while (first + pair_idx < last) {
    typename viewT::value_type pair = *(first + pair_idx);
    if constexpr (is_outer) {
      view.pair_retrieve_outer<tile_size, buffer_size>(tile,
                                                       pair,
                                                       thrust::nullopt,
                                                       &cg_counter[cg_id],
                                                       probe_output_buffer[cg_id],
                                                       contained_output_buffer[cg_id],
                                                       num_matches,
                                                       probe_output_begin,
                                                       contained_output_begin,
                                                       pair_equal);
    } else {
      view.pair_retrieve<tile_size, buffer_size>(tile,
                                                 pair,
                                                 thrust::nullopt,
                                                 &cg_counter[cg_id],
                                                 probe_output_buffer[cg_id],
                                                 contained_output_buffer[cg_id],
                                                 num_matches,
                                                 probe_output_begin,
                                                 contained_output_begin,
                                                 pair_equal);
    }
    pair_idx += (gridDim.x * block_size) / tile_size;
  }

  // Final flush of output buffer
  if (cg_counter[cg_id] > 0) {
    view.flush_output_buffer(tile,
                             cg_counter[cg_id],
                             probe_output_buffer[cg_id],
                             contained_output_buffer[cg_id],
                             num_matches,
                             probe_output_begin,
                             contained_output_begin);
  }
}

}  // namespace detail
}  // namespace cuco
