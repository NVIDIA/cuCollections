/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>

#include <cooperative_groups.h>

#include <cstddef>
#include <iostream>

// insert a set of keys into a hash set using one cooperative group for each task
template <typename SetRef, typename InputIterator>
__global__ void custom_cooperative_insert(SetRef raw_set, InputIterator keys, std::size_t n)
{
  namespace cg = cooperative_groups;

  constexpr auto cg_size = SetRef::cg_size;

  // we haven't spcified any functions yet so we make a copy with the desired functionality
  auto set = raw_set.template with_functions<cuco::experimental::insert>();

  auto tile = cg::tiled_partition<cg_size>(cg::this_thread_block());

  int64_t const loop_stride = gridDim.x * blockDim.x / cg_size;
  int64_t idx               = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;

  while (idx < n) {
    set.insert(tile, *(keys + idx));
    idx += loop_stride;
  }
}

template <typename SetRef, typename InputIterator, typename OutputIterator>
__global__ void custom_contains(SetRef set, InputIterator keys, std::size_t n, OutputIterator found)
{
  int64_t const loop_stride = gridDim.x * blockDim.x;
  int64_t idx               = blockDim.x * blockIdx.x + threadIdx.x;

  while (idx < n) {
    found[idx] = set.contains(*(keys + idx)) ? true : false;
    idx += loop_stride;
  }
}

/**
 * @file device_reference_example.cu
 * @brief Demonstrates usage of the static_set device-side APIs.
 *
 * static_set provides a non-owning reference which can be used to interact with
 * the container from within device code.
 *
 */
int main(void)
{
  using Key = int;

  // Empty slots are represented by reserved "sentinel" values. These values should be selected such
  // that they never occur in your input data.
  Key constexpr empty_key_sentinel = -1;

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
    set.reference(), keys.begin() + num_keys / 2, num_keys / 2);

  // Storage for result
  thrust::device_vector<bool> found(num_keys);

  // Check if all keys are now contained in the set. Note that we pass a reference that already has
  // the `contains` functions
  custom_contains<<<128, 128>>>(
    set.template reference<cuco::experimental::contains>(), keys.begin(), num_keys, found.begin());

  // Verify that all keys have been found
  bool const all_keys_found = thrust::all_of(found.begin(), found.end(), thrust::identity<bool>());

  if (all_keys_found) { std::cout << "Success! Found all keys.\n"; }

  return 0;
}