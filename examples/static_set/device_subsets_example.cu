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

#include <cuco/static_set.cuh>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <cooperative_groups.h>

#include <cuda/std/array>

#include <algorithm>
#include <cstddef>
#include <iostream>

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
template <int CGSize, typename Window, typename Size, typename Offset>
__global__ void kernel(Window* windows, Size* sizes, Offset* offsets)
{
  namespace cg = cooperative_groups;

  using Key = typename Window::value_type;

  auto const tile = cg::tiled_partition<CGSize>(cg::this_thread_block());
  auto const idx  = (blockDim.x * blockIdx.x + threadIdx.x) / CGSize;

  auto const probing_scheme =
    cuco::experimental::linear_probing<CGSize, cuco::default_hash_function<Key>>{};

  cuco::experimental::detail::aow_storage_ref<1, Key, cuco::experimental::extent<std::size_t>>{
    cuco::experimental::extent{sizes[idx]}, windows + offsets[idx]};
}

/**
 * @file device_reference_example.cu
 * @brief Demonstrates usage of the static_set device-side APIs.
 *
 * static_set provides a non-owning reference which can be used to interact with
 * the container from within device code.
 *
 */
int main()
{
  using Key = int;

  // Empty slots are represented by reserved "sentinel" values. These values should be selected such
  // that they never occur in your input data.
  Key constexpr empty_key_sentinel = -1;

  // Number of subsets
  auto constexpr num = 16;
  // Sizes of the 16 subsets to be created on the device
  auto constexpr subset_sizes =
    std::array<std::size_t, num>{20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50};
  // A CUDA Cooperative Group of 8 threads to handle each subset
  auto constexpr cg_size     = 8;
  auto constexpr window_size = 1;  // TODO: how to explain window size (vector length) to users

  auto valid_sizes = std::vector<std::size_t>(num);
  std::generate(valid_sizes.begin(), valid_sizes.end(), [&, n = 0]() mutable {
    return cuco::experimental::make_window_extent<cg_size, window_size>(subset_sizes[n++]);
  });

  auto const d_sizes = thrust::device_vector<std::size_t>{valid_sizes};
  auto d_offsets     = thrust::device_vector<std::size_t>(num);
  thrust::exclusive_scan(d_sizes.begin(), d_sizes.end(), d_offsets.begin());

  auto const num_windows = thrust::reduce(valid_sizes.begin(), valid_sizes.end());

  auto d_set_storage =
    thrust::device_vector<cuco::experimental::window<Key, window_size>>(num_windows);

  // Sets all slot contents to the sentinel value
  initialize<<<128, 128>>>(d_set_storage.data().get(), num_windows, empty_key_sentinel);

  kernel<cg_size>
    <<<1, 128>>>(d_set_storage.data().get(), d_sizes.data().get(), d_offsets.data().get());

  /*
// Number of keys to be inserted
std::size_t constexpr num_keys = 50'000;

// Compute capacity based on a 50% load factor
auto constexpr load_factor = 0.5;
std::size_t const capacity = std::ceil(num_keys / load_factor);

using set_type = cuco::experimental::static_set<Key>;

// Constructs a hash set with at least "capacity" slots using -1 as the empty key sentinel.
set_type set{capacity, cuco::empty_key{empty_key_sentinel}};

// Create a sequence of keys {0, 1, 2, .., i}
thrust::device_vector<Key> keys(num_keys);
thrust::sequence(keys.begin(), keys.end(), 0);

// Insert the first half of the keys into the set
set.insert(keys.begin(), keys.begin() + num_keys / 2);

// Insert the second half of keys using a custom CUDA kernel.
custom_cooperative_insert<<<128, 128>>>(
  set.ref(cuco::experimental::insert), keys.begin() + num_keys / 2, num_keys / 2);

// Storage for result
thrust::device_vector<bool> found(num_keys);

// Check if all keys are now contained in the set. Note that we pass a reference that already has
// the `contains` operator.
// In general, using two or more reference objects to the same container but with
// a different set of operators concurrently is undefined behavior.
// This does not apply here since the two kernels do not overlap.
custom_contains<<<128, 128>>>(
  set.ref(cuco::experimental::contains), keys.begin(), num_keys, found.begin());

// Verify that all keys have been found
bool const all_keys_found = thrust::all_of(found.begin(), found.end(), thrust::identity<bool>());

if (all_keys_found) { std::cout << "Success! Found all keys.\n"; }
*/
  return 0;
}
