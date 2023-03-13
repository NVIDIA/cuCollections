/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/transform.h>

#include <cub/block/block_reduce.cuh>

#include <cuda/std/atomic>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

/**
 * @file count_by_key_example.cu
 * @brief Demonstrates usage of the device side APIs for individual operations like insert/find in
 * the context of a count-by-key operation, i.e. for a histogram over keys.
 *
 * Individual operations like a single insert or find can be performed in device code via the
 * static_map "device_view" types.
 *
 * @note This example is for demonstration purposes only. It is not intended to show the most
 * performant way to do the example algorithm.
 *
 */

/**
 * @brief Inserts keys and counts how often they occur in the input sequence.
 *
 * @tparam BlockSize CUDA block size
 * @tparam Map Type of the map returned from static_map::get_device_mutable_view
 * @tparam KeyIter Input iterator whose value_type convertible to Map::key_type
 * @tparam UniqueIter Output iterator whose value_type is convertible to uint64_t
 *
 * @param[in] map_view View of the map into which inserts will be performed
 * @param[in] key_begin The beginning of the range of keys to insert
 * @param[in] num_keys The total number of keys and values
 * @param[out] num_unique_keys The total number of distinct keys inserted
 */
template <int64_t BlockSize, typename Map, typename KeyIter, typename UniqueIter>
__global__ void count_by_key(Map map_view,
                             KeyIter keys,
                             uint64_t num_keys,
                             UniqueIter num_unique_keys)
{
  typedef cub::BlockReduce<uint64_t, BlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int64_t const loop_stride = gridDim.x * BlockSize;
  int64_t idx               = BlockSize * blockIdx.x + threadIdx.x;

  uint64_t thread_unique_keys = 0;
  while (idx < num_keys) {
    // insert key into the map with a count of 1
    auto [slot, is_new_key] = map_view.insert_and_find({keys[idx], 1});
    if (is_new_key) {
      // first occurrence of the key
      thread_unique_keys++;
    } else {
      // key is already in the map -> increment count
      slot->second.fetch_add(1, cuda::memory_order_relaxed);
    }
    idx += loop_stride;
  }

  // compute number of successfully inserted new keys for each block
  // and atomically add to the grand total
  uint64_t block_unique_keys = BlockReduce(temp_storage).Sum(thread_unique_keys);
  if (threadIdx.x == 0) {
    cuda::atomic_ref<uint64_t> grid_unique_keys(*thrust::raw_pointer_cast(num_unique_keys));
    grid_unique_keys.fetch_add(block_unique_keys, cuda::memory_order_relaxed);
  }
}

int main(void)
{
  using Key   = uint64_t;
  using Count = uint64_t;

  // Empty slots are represented by reserved "sentinel" values. These values should be selected such
  // that they never occur in your input data.
  Key constexpr empty_key_sentinel     = static_cast<Key>(-1);
  Count constexpr empty_value_sentinel = static_cast<Key>(-1);

  // Number of keys to be inserted
  auto constexpr num_keys = 50'000;
  // How often each distinct key occurs in the example input
  auto constexpr key_duplicates = 5;
  static_assert((num_keys % key_duplicates) == 0,
                "For this example, num_keys must be divisible by key_duplicates in order to pass "
                "the unit test.");

  thrust::device_vector<Key> insert_keys(num_keys);
  // Create a sequence of keys. Eeach distinct key has key_duplicates many matches.
  thrust::transform(
    thrust::make_counting_iterator<Key>(0),
    thrust::make_counting_iterator<Key>(insert_keys.size()),
    insert_keys.begin(),
    [] __device__(auto i) { return static_cast<Key>(i % (num_keys / key_duplicates)); });

  // Allocate storage for count of number of unique keys
  thrust::device_vector<uint64_t> num_unique_keys(1);

  // Compute capacity based on a 50% load factor
  auto constexpr load_factor = 0.5;

  // If the number of unique keys is known in advance, we can use it to calculate the map capacity
  std::size_t const capacity = std::ceil((num_keys / key_duplicates) / load_factor);
  // If we can't give an estimated upper bound on the number of unique keys
  // we conservatively assume each key in the input is distinct
  // std::size_t const capacity = std::ceil(num_keys / load_factor);

  // Constructs a map with "capacity" slots.
  cuco::static_map<Key, Count> map{
    capacity, cuco::empty_key{empty_key_sentinel}, cuco::empty_value{empty_value_sentinel}};

  // Get a non-owning, mutable view of the map that allows inserts to pass by value into the kernel
  auto device_insert_view = map.get_device_mutable_view();

  auto constexpr block_size = 256;
  auto const grid_size      = (num_keys + block_size - 1) / block_size;
  count_by_key<block_size><<<grid_size, block_size>>>(
    device_insert_view, insert_keys.begin(), num_keys, num_unique_keys.data());

  // Retrieve contents of all the non-empty slots in the map
  thrust::device_vector<Key> result_keys(num_unique_keys[0]);
  thrust::device_vector<Count> result_counts(num_unique_keys[0]);
  map.retrieve_all(result_keys.begin(), result_counts.begin());

  // Check if the number of result keys is correct
  auto num_keys_check = num_unique_keys[0] == (num_keys / key_duplicates);

  // Iterate over all result counts and verify that they are correct
  auto counts_check = thrust::all_of(
    result_counts.begin(), result_counts.end(), [] __host__ __device__(Count const count) {
      return count == key_duplicates;
    });

  if (num_keys_check and counts_check) { std::cout << "Success!\n"; }

  return 0;
}
