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

#include <cuco/static_set_ref.cuh>
#include <cuco/storage.cuh>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <cooperative_groups.h>

#include <cuda/std/array>

#include <algorithm>
#include <cstddef>
#include <iostream>

auto constexpr cg_size     = 8;   ///< A CUDA Cooperative Group of 8 threads to handle each subset
auto constexpr window_size = 1;   ///< TODO: how to explain window size (vector length) to users
auto constexpr N           = 10;  ///< Number of elements to insert and query

using key_type = int;
using probing_scheme_type =
  cuco::experimental::linear_probing<cg_size, cuco::default_hash_function<key_type>>;
using storage_ref_type = cuco::experimental::aow_storage_ref<key_type, window_size>;
template <typename Operator>
using ref_type = cuco::experimental::static_set_ref<key_type,
                                                    cuda::thread_scope_device,
                                                    thrust::equal_to<key_type>,
                                                    probing_scheme_type,
                                                    storage_ref_type,
                                                    Operator>;

/// data to insert/query
__device__ constexpr std::array<key_type, N> data = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
/// Empty slots are represented by reserved "sentinel" values. These values should be selected such
/// that they never occur in your input data.
key_type constexpr empty_key_sentinel = -1;

template <typename WindowT>
__global__ void initialize(WindowT* windows, std::size_t n, typename WindowT::value_type value)
{
  using T = typename WindowT::value_type;

  auto const loop_stride = gridDim.x * blockDim.x;
  auto idx               = blockDim.x * blockIdx.x + threadIdx.x;

  while (idx < n) {
    auto& window_slots = *(windows + idx);
#pragma unroll
    for (auto& slot : window_slots) {
      new (&slot) T{value};
    }
    idx += loop_stride;
  }
}

// insert a set of keys into a hash set using one cooperative group for each task
template <typename Window, typename Size, typename Offset>
__global__ void insert(Window* windows, Size* sizes, Offset* offsets)
{
  namespace cg = cooperative_groups;

  auto const tile = cg::tiled_partition<cg_size>(cg::this_thread_block());
  auto const idx  = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;

  auto set_ref = ref_type<cuco::experimental::insert_tag>{
    cuco::empty_key<key_type>{-1}, {}, {}, storage_ref_type{sizes[idx], windows + offsets[idx]}};

  // Each cooperative_groups inserts all elements in `data` into its own subset
  for (int i = 0; i < N; i++) {
    set_ref.insert(tile, data[i]);
  }
}

// insert a set of keys into a hash set using one cooperative group for each task
template <typename Window, typename Size, typename Offset>
__global__ void find(Window* windows, Size* sizes, Offset* offsets)
{
  namespace cg = cooperative_groups;

  auto const tile = cg::tiled_partition<cg_size>(cg::this_thread_block());
  auto const idx  = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;

  auto set_ref = ref_type<cuco::experimental::find_tag>{
    cuco::empty_key<key_type>{-1}, {}, {}, storage_ref_type{sizes[idx], windows + offsets[idx]}};

  __shared__ int result;
  if (threadIdx.x == 0) { result = 0; }
  __syncthreads();

  for (int i = 0; i < N; i++) {
    auto const found = set_ref.find(tile, data[i]);
    // Record if the inserted data has been found
    atomicOr(&result, *found != data[i]);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (result == 0) { printf("Success! Found all inserted elements.\n"); }
  }
}

/**
 * @file device_subsets_example.cu
 * @brief Demonstrates usage of the static_set device-side APIs.
 *
 * static_set provides a non-owning reference which can be used to interact with
 * the container from within device code.
 */
int main()
{
  // Number of subsets
  auto constexpr num = 16;
  // Sizes of the 16 subsets to be created on the device
  auto constexpr subset_sizes =
    std::array<std::size_t, num>{20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50};

  auto valid_sizes = std::vector<std::size_t>(num);
  std::generate(valid_sizes.begin(), valid_sizes.end(), [&, n = 0]() mutable {
    return cuco::experimental::make_window_extent<cg_size, window_size>(subset_sizes[n++]);
  });

  auto const d_sizes = thrust::device_vector<std::size_t>{valid_sizes};
  auto d_offsets     = thrust::device_vector<std::size_t>(num);
  thrust::exclusive_scan(d_sizes.begin(), d_sizes.end(), d_offsets.begin());

  auto const num_windows = thrust::reduce(valid_sizes.begin(), valid_sizes.end());

  // One allocation for all subsets
  auto d_set_storage = cuco::experimental::aow_storage<key_type, window_size>{num_windows};
  // Initializes the storage with the given sentinel
  d_set_storage.initialize(empty_key_sentinel);

  insert<<<1, 128>>>(d_set_storage.data(), d_sizes.data().get(), d_offsets.data().get());
  find<<<1, 128>>>(d_set_storage.data(), d_sizes.data().get(), d_offsets.data().get());

  return 0;
}
