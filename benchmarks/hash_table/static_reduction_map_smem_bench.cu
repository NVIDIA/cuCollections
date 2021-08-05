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

#include <cuda_benchmark.h>
#include <thrust/device_vector.h>
#include <cuco/static_reduction_map.cuh>
#include <iostream>
#include <key_generator.hpp>
#include <util.hpp>

template <typename Key, typename Value>
void static_reduction_map_smem_insert_bench(cuda_benchmark::controller& controller,
                                            std::size_t num_elems,
                                            float occupancy,
                                            dist_type dist,
                                            std::size_t multiplicity = 8)
{
  using map_type  = cuco::static_reduction_map<typename cuco::reduce_add<Value>, Key, Value>;
  using pair_type = typename map_type::value_type;

  int dev_id;
  cudaGetDevice(&dev_id);
  struct cudaDeviceProp dev_props;
  cudaGetDeviceProperties(&dev_props, dev_id);
  std::size_t const max_smem     = dev_props.sharedMemPerBlock;
  std::size_t const max_capacity = max_smem / sizeof(pair_type);

  std::vector<Key> h_keys_in(num_elems);
  std::vector<Value> h_values_in(num_elems);

  if (not generate_keys<Key>(dist, h_keys_in.begin(), h_keys_in.end(), multiplicity)) {
    std::cerr << "[ERROR] Invalid input distribution.\n";
    return;
  }

  // generate uniform random values
  generate_keys<Value>("UNIFORM", h_values_in.begin(), h_values_in.end(), 1);

  // the size of the hash table under a given target occupancy depends on the
  // number of unique keys in the input
  std::size_t const unique   = count_unique(h_keys_in.begin(), h_keys_in.end());
  std::size_t const capacity = std::ceil(SDIV(unique, occupancy));

  if (capacity > max_capacity) {
    std::cerr << "[ERROR] Not enough shared memory available. (" << capacity * sizeof(pair_type)
              << ">" << max_capacity * sizeof(pair_type) << " bytes)\n";
    return;
  }

  thrust::device_vector<Key> d_keys_in(h_keys_in);
  thrust::device_vector<Value> d_values_in(h_values_in);

  controller.benchmark(
    "static_reduction_map shared memory insert",
    [=, keys_ptr = d_keys_in.data().get(), values_ptr = d_values_in.data().get()] __device__(
      cuda_benchmark::state & state) {
      using map_type      = typename cuco::static_reduction_map<typename cuco::reduce_add<Value>,
                                                           Key,
                                                           Value,
                                                           cuda::thread_scope_block>;
      using map_view_type = typename map_type::device_mutable_view;

      __shared__ typename map_type::pair_atomic_type* slots;

      auto g    = cooperative_groups::this_thread_block();
      auto map  = map_view_type::make_from_uninitialized_slots(g, slots, capacity, -1);
      auto pair = pair_type(keys_ptr[g.thread_rank()], values_ptr[g.thread_rank()]);

      g.sync();

      for (auto _ : state) {
        map.insert(pair);
        g.sync();
      }
    },
    max_smem);
}

int main()
{
  cuda_benchmark::controller controller(1024, 1);

  static_reduction_map_smem_insert_bench<std::int32_t, std::int32_t>(
    controller, 10'000, 0.8, dist_type::UNIFORM);
}