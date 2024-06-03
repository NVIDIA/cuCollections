/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#undef NDEBUG
#include <cuco/static_multimap.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cooperative_groups.h>

#include <assert.h>

#include <limits>

#define TILE_SIZE 8

namespace cg = cooperative_groups;

// This is actually a count kernel by utilizing self-defined cuco multimap api
// that returns iterator and next_iterator
template <typename MapViewT, typename InputIt, uint32_t tile_size = TILE_SIZE>
__global__ void find(MapViewT multi_map, InputIt first, int n, int* num_matches)
{
  // Similar stuff as cuco device-side count does
  auto tile                 = cg::tiled_partition<tile_size>(cg::this_thread_block());
  int64_t const loop_stride = gridDim.x * blockDim.x / tile_size;
  int64_t idx               = (blockDim.x * blockIdx.x + threadIdx.x) / tile_size;
  while (idx < n) {
    // printf("before: idx is %ld\n", idx);
    auto const key   = static_cast<int>(first[idx]);
    auto it          = multi_map.find(tile, key);
    using value_type = typename MapViewT::value_type;
    while (true) {
      value_type slot_contents = *reinterpret_cast<value_type const*>(it);
      auto const& current_key  = slot_contents.first;

      auto const slot_is_empty =
        cuco::detail::bitwise_compare(current_key, multi_map.get_empty_key_sentinel());
      auto const equals = not slot_is_empty and cuco::detail::bitwise_compare(current_key, key);

      atomicAdd(num_matches, equals);
      if (tile.any(slot_is_empty)) { break; }

      it = multi_map.next_iterator(tile, key, it);
    }

    idx += loop_stride;
  }
}

// Count by just calling device count api and then increment atomic sum
template <typename MapViewT, typename InputIt, uint32_t tile_size = TILE_SIZE>
__global__ void count(MapViewT multi_map, InputIt first, int n, int* num_matches)
{
  auto tile                 = cg::tiled_partition<tile_size>(cg::this_thread_block());
  int64_t const loop_stride = gridDim.x * blockDim.x / tile_size;
  int64_t idx               = (blockDim.x * blockIdx.x + threadIdx.x) / tile_size;
  while (idx < n) {
    // printf("before: idx is %ld\n", idx);
    auto key   = static_cast<int>(first[idx]);
    auto count = multi_map.count(tile, key);
    atomicAdd(num_matches, count);
    idx += loop_stride;
  }
}

// Official implementation for the count kernel
template <typename MapViewT, typename InputIt, uint32_t tile_size = TILE_SIZE>
__global__ void count_official(MapViewT multi_map, InputIt first, int n, int* num_matches)
{
  constexpr int block_size  = 64;
  auto tile                 = cg::tiled_partition<tile_size>(cg::this_thread_block());
  int64_t const loop_stride = gridDim.x * block_size / tile_size;
  int64_t idx               = (block_size * blockIdx.x + threadIdx.x) / tile_size;

  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_matches = 0;

  while (idx < n) {
    auto key = *(first + idx);
    thread_num_matches += multi_map.count(tile, key);
    idx += loop_stride;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  std::size_t block_num_matches = BlockReduce(temp_storage).Sum(thread_num_matches);
  if (threadIdx.x == 0) { atomicAdd(num_matches, block_num_matches); }
}

struct MyHash {
  __device__ __host__ int operator()(int key) const { return 2 * key; }
};

int main(void)
{
  using key_type   = int;
  using value_type = int;

  key_type empty_key_sentinel     = -1;
  value_type empty_value_sentinel = -1;

  constexpr std::size_t N = 50'000;

  auto const stride     = 1;
  auto const block_size = 32;
  auto const grid_size  = (TILE_SIZE * N + stride * block_size - 1) / (stride * block_size);

  // Constructs a multimap with 100,000 slots using -1 and -1 as the empty
  // key/value sentinels. Note the capacity is chosen knowing we will insert
  // 50,000 keys, for an load factor of 50%.

  // cuco::default_hash_function<key_type>
  // cuco::static_multimap<key_type, value_type, cuda::thread_scope_device,
  //                       cuco::cuda_allocator<char>,
  //                       cuco::legacy::linear_probing<
  //                           TILE_SIZE,
  //                           cuco::default_hash_function<key_type>>>
  //     map{N * 2, cuco::empty_key{empty_key_sentinel},
  //         cuco::empty_value{empty_value_sentinel}};
  cuco::static_multimap<
    key_type,
    value_type,
    cuda::thread_scope_device,
    cuco::cuda_allocator<char>,
    cuco::legacy::double_hashing<TILE_SIZE, cuco::default_hash_function<key_type>>>
    map{N * 2, cuco::empty_key{empty_key_sentinel}, cuco::empty_value{empty_value_sentinel}};
  // cuco::static_multimap<key_type, value_type> map{
  //     N * 2, cuco::empty_key{empty_key_sentinel},
  //     cuco::empty_value{empty_value_sentinel}};

  thrust::device_vector<thrust::pair<key_type, value_type>> pairs(N);

  // Create a sequence of pairs. Each key has two matches.
  // E.g., {{0,0}, {1,1}, ... {0,25'000}, {1, 25'001}, ...}
  // For each key between 0-24999, there will be two pairs with the same key,
  // but different values
  thrust::transform(thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(pairs.size()),
                    pairs.begin(),
                    [] __device__(auto i) { return thrust::make_pair(i % (N / 2), i); });

  // Inserts all pairs into the map
  map.insert(pairs.begin(), pairs.end());

  // Sequence of probe keys {0, 1, 2, ... 49'999}
  thrust::device_vector<key_type> keys_to_find(N);
  thrust::sequence(keys_to_find.begin(), keys_to_find.end(), 0);

  int* num_matches;
  cudaMalloc(&num_matches, sizeof(int));
  cudaMemset(num_matches, 0, sizeof(int));
  auto device_view = map.get_device_view();
  find<<<grid_size, block_size>>>(device_view, keys_to_find.begin(), N, num_matches);
  // find<<<32, 32>>>(device_view, pairs.data().get(), N, num_matches);

  // All of the following printing should be 50,000
  // Issue is that find kernel is not working correctly by returning way fewer
  // matches
  cudaDeviceSynchronize();
  int* num_matches_host;
  num_matches_host = (int*)malloc(sizeof(int));
  cudaMemcpy(num_matches_host, num_matches, sizeof(int), cudaMemcpyDeviceToHost);
  printf("find: num_matches: %d\n", *num_matches_host);
  cudaMemset(num_matches, 0, sizeof(int));

  // Reference count kernel and result
  count<<<32, 64>>>(device_view, keys_to_find.begin(), N, num_matches);
  cudaMemcpy(num_matches_host, num_matches, sizeof(int), cudaMemcpyDeviceToHost);
  printf("count: num_matches: %d\n", *num_matches_host);
  cudaMemset(num_matches, 0, sizeof(int));

  count_official<<<32, 64>>>(device_view, keys_to_find.begin(), N, num_matches);
  cudaMemcpy(num_matches_host, num_matches, sizeof(int), cudaMemcpyDeviceToHost);
  printf("count official: num_matches: %d\n", *num_matches_host);

  auto count = map.count(keys_to_find.begin(), keys_to_find.end());
  printf("count: %ld\n", count);
  return 0;
}
