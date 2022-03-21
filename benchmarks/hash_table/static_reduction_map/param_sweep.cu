/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cuco/detail/utils.hpp>
#include <cuco/static_reduction_map.cuh>
#include <key_generator.hpp>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <utils.hpp>

/**
 * @brief Grid search evaluating backoff delay params for cuco::custom_op
 */
template <typename Key,
          typename Value,
          nvbench::int32_t BackoffBaseDelay,
          nvbench::int32_t BackoffMaxDelay>
void nvbench_cuco_static_reduction_map_custom_op_backoff_delay(
  nvbench::state& state,
  nvbench::type_list<Key,
                     Value,
                     nvbench::enum_type<BackoffBaseDelay>,
                     nvbench::enum_type<BackoffMaxDelay>>)
{
  using custom_op_type =
    cuco::custom_op<Value, 0, thrust::plus<Value>, BackoffBaseDelay, BackoffMaxDelay>;
  using map_type = cuco::static_reduction_map<custom_op_type, Key, Value>;

  auto const num_elems    = state.get_int64("NumInputs");
  auto const occupancy    = state.get_float64("Occupancy");
  auto const dist         = state.get_string("Distribution");
  auto const multiplicity = state.get_int64_or_default("Multiplicity", 8);

  std::vector<Key> h_keys(num_elems);
  std::vector<Value> h_values(num_elems);

  if (not generate_keys<Key>(dist, h_keys.begin(), h_keys.end(), multiplicity)) {
    state.skip("Invalid input distribution.");
    return;
  }

  // generate uniform random values
  generate_keys<Value>("UNIFORM", h_values.begin(), h_values.end(), 1);

  // the size of the hash table under a given target occupancy depends on the
  // number of unique keys in the input
  std::size_t const unique   = count_unique(h_keys.begin(), h_keys.end());
  std::size_t const capacity = std::ceil(SDIV(unique, occupancy));

  // alternative occupancy calculation based on the total number of inputs
  // std::size_t const capacity = num_elems / occupancy;

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<Value> d_values(h_values);

  auto d_pairs_begin =
    thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_values.begin()));
  auto d_pairs_end = d_pairs_begin + num_elems;

  state.add_element_count(num_elems);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               map_type map{capacity, -1};

               timer.start();
               map.insert(d_pairs_begin, d_pairs_end, launch.get_stream());
               timer.stop();
             });
}

// type parameter dimensions for benchmark
using key_type_range   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using value_type_range = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using base_delay_range = nvbench::enum_type_list<4, 8, 16, 32, 64, 128, 256, 512>;
using max_delay_range  = nvbench::enum_type_list<2'048, 4'096, 8'192, 16'384>;

// benchmark setups
NVBENCH_BENCH_TYPES(
  nvbench_cuco_static_reduction_map_custom_op_backoff_delay,
  NVBENCH_TYPE_AXES(key_type_range, value_type_range, base_delay_range, max_delay_range))
  .set_name("cuco_static_reduction_map_custom_op_backoff_delay")
  .set_type_axes_names({"Key", "Value", "BackoffBaseDelay", "BackoffMaxDelay"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs
  .add_float64_axis("Occupancy", {0.8})        // fixed occupancy
  .add_int64_axis("Multiplicity",
                  {1, 10, 100, 1'000, 10'000, 100'000, 1'000'000})  // key multiplicity range
  .add_string_axis("Distribution", {"UNIFORM"});