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

#include <key_generator.hpp>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

/**
 * @brief Reduce-by-key implementation in Thrust.
 */
template <typename KeyRandomIterator, typename ValueRandomIterator>
void thrust_reduce_by_key(KeyRandomIterator keys_begin,
                          KeyRandomIterator keys_end,
                          ValueRandomIterator values_begin)
{
  using Key   = typename thrust::iterator_traits<KeyRandomIterator>::value_type;
  using Value = typename thrust::iterator_traits<ValueRandomIterator>::value_type;

  // Exact size of output is unknown (number of unique keys), but upper-bounded
  // by the number of keys
  auto maximum_output_size = thrust::distance(keys_begin, keys_end);
  thrust::device_vector<Key> output_keys(maximum_output_size);
  thrust::device_vector<Value> output_values(maximum_output_size);

  thrust::sort_by_key(thrust::device, keys_begin, keys_end, values_begin);
  thrust::reduce_by_key(
    thrust::device, keys_begin, keys_end, values_begin, output_keys.begin(), output_values.begin());
}

/**
 * @brief A benchmark evaluating Thrust's reduce-by-key performance.
 */
template <typename Key, typename Value>
void nvbench_thrust_reduce_by_key(nvbench::state& state, nvbench::type_list<Key, Value>)
{
  auto const num_elems    = state.get_int64("NumInputs");
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

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<Value> d_values(h_values);

  state.add_element_count(num_elems);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               thrust_reduce_by_key(d_keys.begin(), d_keys.end(), d_values.begin());
               timer.stop();
             });
}

// type parameter dimensions for benchmark
using key_type_range   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using value_type_range = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;

// benchmark setups
NVBENCH_BENCH_TYPES(nvbench_thrust_reduce_by_key,
                    NVBENCH_TYPE_AXES(key_type_range, value_type_range))
  .set_name("thrust_reduce_by_key_distribution")
  .set_type_axes_names({"Key", "Value"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs
  .add_int64_axis("Multiplicity", {8})         // only applies to uniform distribution
  .add_string_axis("Distribution", {"GAUSSIAN", "UNIFORM", "UNIQUE", "SAME"});

NVBENCH_BENCH_TYPES(nvbench_thrust_reduce_by_key,
                    NVBENCH_TYPE_AXES(key_type_range, value_type_range))
  .set_name("thrust_reduce_by_key_multiplicity")
  .set_type_axes_names({"Key", "Value"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs
  .add_int64_axis("Multiplicity",
                  {1, 10, 100, 1'000, 10'000, 100'000, 1'000'000})  // key multiplicity range
  .add_string_axis("Distribution", {"UNIFORM"});