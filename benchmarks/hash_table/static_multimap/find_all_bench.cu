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

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <random>

#include "cuco/static_multimap.cuh"

/**
 * @brief Generates input keys by a given number of repetitions per key.
 *
 */
template <typename Key, typename OutputIt>
static void generate_multikeys(OutputIt output_begin,
                               OutputIt output_end,
                               size_t const multiplicity)
{
  auto num_keys = std::distance(output_begin, output_end);

  for (auto i = 0; i < num_keys; ++i) {
    output_begin[i] = (i % (num_keys / multiplicity)) + 1;
  }
}

/**
 * @brief A benchmark evaluating multi-value retrieval performance by varing number of repetitions
 * per key:
 * - 100'000'000 keys are inserted
 * - Map occupancy is fixed at 0.4
 * - Number of repetitions per key: 1, ... , 128, 256
 *
 */
template <typename Key, typename Value, nvbench::int32_t CGSize, nvbench::int32_t BufferSize>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_find_all(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<CGSize>, nvbench::enum_type<BufferSize>>)
{
  std::size_t const num_keys     = state.get_int64("NumInputs");
  auto const occupancy           = state.get_float64("Occupancy");
  std::size_t const size         = num_keys / occupancy;
  std::size_t const multiplicity = state.get_int64("Multiplicity");

  state.add_element_count(num_keys, "NumKeys");
  state.add_global_memory_writes<Key>(num_keys * 2);

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_multikeys<Key>(h_keys.begin(), h_keys.end(), multiplicity);
  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  cuco::static_multimap<Key,
                        Value,
                        cuda::thread_scope_device,
                        cuco::detail::double_hashing<Key, Value, CGSize>>
    map{size, -1, -1};
  map.insert(d_pairs.begin(), d_pairs.end());

  auto const output_size = map.count_outer(d_keys.begin(), d_keys.end());
  thrust::device_vector<cuco::pair_type<Key, Value>> d_results(output_size);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    map.retrieve_outer(d_keys.begin(), d_keys.end(), d_results.data().get(), launch.get_stream());
  });
}

template <typename Key, typename Value, nvbench::int32_t CGSize, nvbench::int32_t BufferSize>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_find_all(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<CGSize>, nvbench::enum_type<BufferSize>>)
{
  state.skip("Key should be the same type as Value.");
}

using key_type    = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using value_type  = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using cg_size     = nvbench::enum_type_list<1, 2, 4, 8, 16, 32>;
using buffer_size = nvbench::enum_type_list<1, 2, 4, 8, 16>;

NVBENCH_BENCH_TYPES(nvbench_find_all,
                    NVBENCH_TYPE_AXES(key_type, value_type, cg_size, nvbench::enum_type_list<2>))
  .set_type_axes_names({"Key", "Value", "CGSize", "BufferSize"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.4})
  .add_int64_power_of_two_axis("Multiplicity", nvbench::range(0, 8, 1));

NVBENCH_BENCH_TYPES(
  nvbench_find_all,
  NVBENCH_TYPE_AXES(key_type, value_type, nvbench::enum_type_list<8>, buffer_size))
  .set_type_axes_names({"Key", "Value", "CGSize", "BufferSize"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.4})
  .add_int64_power_of_two_axis("Multiplicity", nvbench::range(0, 8, 1));
