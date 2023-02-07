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

#include <key_generator.hpp>

#include <cuco/static_set.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>

#include <vector>

/**
 * @brief A benchmark evaluating multi-value `insert` performance:
 * - Total number of insertions: 100'000'000
 */
template <typename Key, dist_type Dist, nvbench::int32_t Multiplicity>
void nvbench_static_set_insert(
  nvbench::state& state,
  nvbench::type_list<Key, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  auto const num_keys  = state.get_int64("NumInputs");
  auto const occupancy = state.get_float64("Occupancy");

  std::size_t const size = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);

  generate_keys<Dist, Multiplicity, Key>(h_keys.begin(), h_keys.end());
  thrust::device_vector<Key> d_keys(h_keys);

  state.add_element_count(num_keys, "NumKeys");

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               cuco::experimental::static_set<Key> myset{size, cuco::empty_key<Key>{-1}};

               // Use timers to explicitly mark the target region
               timer.start();
               myset.insert(d_keys.begin(), d_keys.end(), launch.get_stream());
               timer.stop();
             });
}

using key_type = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using d_type =
  nvbench::enum_type_list<dist_type::GAUSSIAN, dist_type::GEOMETRIC, dist_type::UNIFORM>;

using multiplicity = nvbench::enum_type_list<1, 2, 4, 8, 16, 32, 64, 128, 256>;

NVBENCH_BENCH_TYPES(nvbench_static_set_insert,
                    NVBENCH_TYPE_AXES(key_type,
                                      nvbench::enum_type_list<dist_type::UNIFORM>,
                                      multiplicity))
  .set_name("staic_set_insert_uniform_multiplicity")
  .set_type_axes_names({"Key", "Distribution", "Multiplicity"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.5});

NVBENCH_BENCH_TYPES(nvbench_static_set_insert,
                    NVBENCH_TYPE_AXES(key_type, d_type, nvbench::enum_type_list<1>))
  .set_name("staic_set_insert_occupancy")
  .set_type_axes_names({"Key", "Distribution", "Multiplicity"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1));
