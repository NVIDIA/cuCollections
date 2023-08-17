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

/**
 * @file device_subsets_example.cu
 * @brief Demonstrates how to use one bulk set storage to create multiple subsets and perform
 * individual operations via device-side ref APIs.
 *
 * To optimize memory usage, especially when dealing with expensive data allocation and multiple
 * hashsets, a practical solution involves employing a single bulk storage for generating subsets.
 * This eliminates the need for separate memory allocation and deallocation for each container. This
 * can be achieved by using the lightweight non-owning ref type.
 *
 * @note This example is for demonstration purposes only. It is not intended to show the most
 * performant way to do the example algorithm.
 */

auto constexpr cg_size     = 8;   ///< A CUDA Cooperative Group of 8 threads to handle each subset
auto constexpr window_size = 1;   ///< Number of concurrent slots handled by each thread
auto constexpr N           = 10;  ///< Number of elements to insert and query

using key_type            = int;  ///< Key type
using probing_scheme_type = cuco::experimental::linear_probing<
  cg_size,
  cuco::default_hash_function<key_type>>;  ///< Type controls CG granularity and probing scheme
                                           ///< (linear probing v.s. double hashing)
using storage_type = cuco::experimental::aow_storage<key_type, window_size>;  ///< Storage type
using storage_ref_type =
  typename storage_type::ref_type;  ///< Lightweight non-owning storage ref type
template <typename Operator>
using ref_type = cuco::experimental::static_set_ref<key_type,
                                                    cuda::thread_scope_device,
                                                    thrust::equal_to<key_type>,
                                                    probing_scheme_type,
                                                    storage_ref_type,
                                                    Operator>;  ///< Set ref type

/// Sample data to insert and query
__device__ constexpr std::array<key_type, N> data = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
/// Empty slots are represented by reserved "sentinel" values. These values should be selected such
/// that they never occur in your input data.
key_type constexpr empty_key_sentinel = -1;

/**
 * @brief Inserts sample data into subsets by using cooperative group
 *
 * Each Cooperative Group creates its own subset and inserts `N` sample data.
 *
 * @tparam WindowType Storage window type
 * @tparam SizeType Size type
 * @tparam OffsetType Offset type
 *
 * @param windows Pointer to the window array
 * @param sizes Pointer to the subset sizes array
 * @param offsets Pointer to the subset offsets array
 */
template <typename WindowType, typename SizeType, typename OffsetType>
__global__ void insert(WindowType* windows, SizeType* sizes, OffsetType* offsets)
{
  namespace cg = cooperative_groups;

  auto const tile = cg::tiled_partition<cg_size>(cg::this_thread_block());
  // Get subset (or CG) index
  auto const idx = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;

  // Construct an "insert" ref with the given storage
  auto set_ref = ref_type<cuco::experimental::insert_tag>{
    cuco::empty_key<key_type>{-1}, {}, {}, storage_ref_type{sizes[idx], windows + offsets[idx]}};

  // Insert `N` elemtns into the set with CG insert
  for (int i = 0; i < N; i++) {
    set_ref.insert(tile, data[i]);
  }
}

/**
 * @brief All inserted data can be found
 *
 * Each Cooperative Group reconstructs its own subset ref based on the storage parameters and
 * verifies all inserted data can be found.
 *
 * @tparam WindowType Storage window type
 * @tparam SizeType Size type
 * @tparam OffsetType Offset type
 *
 * @param windows Pointer to the window array
 * @param sizes Pointer to the subset sizes array
 * @param offsets Pointer to the subset offsets array
 */
template <typename WindowType, typename SizeType, typename OffsetType>
__global__ void find(WindowType* windows, SizeType* sizes, OffsetType* offsets)
{
  namespace cg = cooperative_groups;

  auto const tile = cg::tiled_partition<cg_size>(cg::this_thread_block());
  auto const idx  = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;

  // Reconstruct an "find" ref with the same storage
  auto set_ref = ref_type<cuco::experimental::find_tag>{
    cuco::empty_key<key_type>{-1}, {}, {}, storage_ref_type{sizes[idx], windows + offsets[idx]}};

  // Result denoting if any of the inserted data is not found
  __shared__ int result;
  if (threadIdx.x == 0) { result = 0; }
  __syncthreads();

  for (int i = 0; i < N; i++) {
    // Query the set with inserted data
    auto const found = set_ref.find(tile, data[i]);
    // Record if the inserted data has been found
    atomicOr(&result, *found != data[i]);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // If the result is still 0, all inserted data are found.
    if (result == 0) { printf("Success! Found all inserted elements.\n"); }
  }
}

int main()
{
  // Number of subsets to be created
  auto constexpr num = 16;
  // Each subset may have a different requested size
  auto constexpr subset_sizes =
    std::array<std::size_t, num>{20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50};

  auto valid_sizes = std::vector<std::size_t>(num);
  // Compute the valid sizes based on requested sizes
  std::generate(valid_sizes.begin(), valid_sizes.end(), [&, n = 0]() mutable {
    // The requested size could cause infinite probing sequences for hash sets thus the valid size
    // required by the container MUST be computed via `make_window_extent`
    return cuco::experimental::make_window_extent<cg_size, window_size>(subset_sizes[n++]);
  });

  // Copy host data to device
  auto const d_sizes = thrust::device_vector<std::size_t>{valid_sizes};
  auto d_offsets     = thrust::device_vector<std::size_t>(num);
  // Compute the offset for each subset
  thrust::exclusive_scan(d_sizes.begin(), d_sizes.end(), d_offsets.begin());

  // Get the total size of all subsets.
  auto const num_windows = thrust::reduce(valid_sizes.begin(), valid_sizes.end());

  // Create a single bulk storage used by all subsets
  auto d_set_storage = storage_type{num_windows};
  // Initializes the storage with the given sentinel
  d_set_storage.initialize(empty_key_sentinel);

  // Insert sample data
  insert<<<1, 128>>>(d_set_storage.data(), d_sizes.data().get(), d_offsets.data().get());
  // Find all inserted data
  find<<<1, 128>>>(d_set_storage.data(), d_sizes.data().get(), d_offsets.data().get());

  return 0;
}
