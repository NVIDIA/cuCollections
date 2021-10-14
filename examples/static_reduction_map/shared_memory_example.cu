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

#include <iomanip>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/logical.h>

#include <cuco/static_reduction_map.cuh>

template <typename MapType, std::size_t Capacity, typename OutputIt>
__global__ void static_reduction_map_shared_memory_kernel(OutputIt key_found)
{
  using Key   = typename MapType::key_type;
  using Value = typename MapType::mapped_type;

  namespace cg = cooperative_groups;
  // define a mutable view for insert operations
  using mutable_view_type = typename MapType::device_mutable_view<>;
  // define a immutable view for find/contains operations
  using view_type = typename MapType::device_view<>;

  // hash table storage in shared memory
  #pragma diag_suppress static_var_with_dynamic_init
  __shared__ typename mutable_view_type::slot_type slots[Capacity];

  // construct the table from the provided array in shared memory
  auto map = mutable_view_type::make_from_uninitialized_slots(
    cg::this_thread_block(), &slots[0], Capacity, -1);

  auto g            = cg::this_thread_block();
  std::size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  int rank          = g.thread_rank();

  // insert {thread_rank, thread_rank} for each thread in thread-block
  map.insert(cuco::pair<Key, Value>(rank, rank));
  g.sync();

  auto find_map = view_type(map);
  // check if all previously inserted keys are present in the table
  key_found[index] = find_map.contains(rank);
}

/**
 * @brief Demonstrates usage of the static_reduction_map in shared memory.
 *
 * We make use of the device-side API to construct and query a
 * static reduction map in SM-local shared memory.
 *
 */
int main(void)
{
  using Key   = int;
  using Value = int;

  // define the capacity of the map
  static constexpr int capacity = 2048;

  // define the hash table typewith block-local thread scope
  using map_type =
    cuco::static_reduction_map<cuco::reduce_add<Value>, Key, Value, cuda::thread_scope_block>;

  // allocate storage for the result
  thrust::device_vector<bool> result(1024, false);

  static_reduction_map_shared_memory_kernel<map_type, capacity><<<1, 1024>>>(result.begin());

  auto success =
    thrust::all_of(thrust::device, result.begin(), result.end(), thrust::identity<bool>());

  std::cout << "Success: " << std::boolalpha << success << std::endl;
}