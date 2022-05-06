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

#include <cub/cub.cuh>
#include <key_generator.hpp>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

/**
 * @brief A benchmark evaluating CUB's reduce-by-key performance.
 */
template <typename Key, typename Value>
void nvbench_cub_reduce_by_key(nvbench::state& state, nvbench::type_list<Key, Value>)
{
  auto const num_elems_in = state.get_int64("NumInputs");
  auto const dist         = state.get_string("Distribution");
  auto const multiplicity = state.get_int64_or_default("Multiplicity", 8);

  std::vector<Key> h_keys(num_elems_in);
  std::vector<Value> h_values(num_elems_in);

  if (not generate_keys<Key>(dist, h_keys.begin(), h_keys.end(), multiplicity)) {
    state.skip("Invalid input distribution.");
    return;
  }

  // generate uniform random values
  generate_keys<Value>("UNIFORM", h_values.begin(), h_values.end(), 1);

  // double buffer (ying/yang)
  thrust::device_vector<Key> d_keys_ying(h_keys);
  thrust::device_vector<Value> d_values_ying(h_values);

  thrust::device_vector<Key> d_keys_yang(num_elems_in);
  thrust::device_vector<Value> d_values_yang(num_elems_in);

  // CUB requires a dry-run in order to determine the size of required temp memory
  std::size_t temp_bytes_sort = 0;
  cub::DeviceRadixSort::SortPairs(nullptr,
                                  temp_bytes_sort,
                                  d_keys_ying.data().get(),
                                  d_keys_yang.data().get(),
                                  d_values_ying.data().get(),
                                  d_values_yang.data().get(),
                                  num_elems_in);

  thrust::device_vector<int> d_num_elems_out(1);

  std::size_t temp_bytes_reduce = 0;
  cub::DeviceReduce::ReduceByKey(nullptr,
                                 temp_bytes_reduce,
                                 d_keys_yang.data().get(),
                                 d_keys_ying.data().get(),
                                 d_values_yang.data().get(),
                                 d_values_ying.data().get(),
                                 d_num_elems_out.data().get(),
                                 cub::Sum(),
                                 num_elems_in);

  thrust::device_vector<char> d_temp(std::max(temp_bytes_sort, temp_bytes_reduce));

  state.add_element_count(num_elems_in);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               cub::DeviceRadixSort::SortPairs(d_temp.data().get(),
                                               temp_bytes_sort,
                                               d_keys_ying.data().get(),
                                               d_keys_yang.data().get(),
                                               d_values_ying.data().get(),
                                               d_values_yang.data().get(),
                                               num_elems_in,
                                               0,
                                               sizeof(Key) * 8,
                                               launch.get_stream());

               cub::DeviceReduce::ReduceByKey(d_temp.data().get(),
                                              temp_bytes_reduce,
                                              d_keys_yang.data().get(),
                                              d_keys_ying.data().get(),
                                              d_values_yang.data().get(),
                                              d_values_ying.data().get(),
                                              d_num_elems_out.data().get(),
                                              cub::Sum(),
                                              num_elems_in,
                                              launch.get_stream());
               timer.stop();
             });
}

// type parameter dimensions for benchmark
using key_type_range   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using value_type_range = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;

// benchmark setups
NVBENCH_BENCH_TYPES(nvbench_cub_reduce_by_key, NVBENCH_TYPE_AXES(key_type_range, value_type_range))
  .set_name("nvbench_cub_reduce_by_key_distribution")
  .set_type_axes_names({"Key", "Value"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs
  .add_int64_axis("Multiplicity", {8})         // only applies to uniform distribution
  .add_string_axis("Distribution", {"GAUSSIAN", "UNIFORM", "UNIQUE", "SAME"});

NVBENCH_BENCH_TYPES(nvbench_cub_reduce_by_key, NVBENCH_TYPE_AXES(key_type_range, value_type_range))
  .set_name("nvbench_cub_reduce_by_key_multiplicity")
  .set_type_axes_names({"Key", "Value"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs
  .add_int64_axis("Multiplicity",
                  {1, 10, 100, 1'000, 10'000, 100'000, 1'000'000})  // key multiplicity range
  .add_string_axis("Distribution", {"UNIFORM"});