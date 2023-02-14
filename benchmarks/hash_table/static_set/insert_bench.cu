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

#include <defaults.hpp>
#include <key_generator.hpp>

#include <cuco/static_set.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

using namespace cuco::benchmark;
using namespace cuco::benchmark::defaults;

/**
 * @brief A benchmark evaluating `insert` performance:
 */
template <typename Key, typename Dist>
void static_set_insert(nvbench::state& state, nvbench::type_list<Key, Dist>)
{
  auto const num_keys  = state.get_int64_or_default("NumInputs", defaults::N);
  auto const occupancy = state.get_float64_or_default("Occupancy", defaults::OCCUPANCY);

  std::size_t const size = num_keys / occupancy;

  thrust::device_vector<Key> keys(num_keys);

  key_generator gen;
  gen.generate<Dist>(state, thrust::device, keys.begin(), keys.end());

  state.add_element_count(num_keys, "NumKeys");
  state.set_global_memory_rw_bytes(num_keys * sizeof(Key));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               cuco::experimental::static_set<Key> myset{size, cuco::empty_key<Key>{-1}};

               // Use timers to explicitly mark the target region
               timer.start();
               myset.insert(keys.begin(), keys.end(), launch.get_stream());
               timer.stop();
             });
}

NVBENCH_BENCH_TYPES(static_set_insert,
                    NVBENCH_TYPE_AXES(KEY_TYPE_RANGE, nvbench::type_list<dist_type::uniform>))
  .set_name("static_set_insert_uniform_multiplicity")
  .set_type_axes_names({"Key", "Distribution"})
  .set_max_noise(MAX_NOISE)  // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("Multiplicity", MULTIPLICITY_RANGE);

NVBENCH_BENCH_TYPES(static_set_insert,
                    NVBENCH_TYPE_AXES(KEY_TYPE_RANGE, nvbench::type_list<dist_type::unique>))
  .set_name("static_set_insert_unique_occupancy")
  .set_type_axes_names({"Key", "Distribution"})
  .set_max_noise(MAX_NOISE)  // Custom noise: 3%. By default: 0.5%.
  .add_float64_axis("Occupancy", OCCUPANCY_RANGE);

NVBENCH_BENCH_TYPES(static_set_insert,
                    NVBENCH_TYPE_AXES(KEY_TYPE_RANGE, nvbench::type_list<dist_type::gaussian>))
  .set_name("static_set_insert_gaussian")
  .set_type_axes_names({"Key", "Distribution"})
  .set_max_noise(MAX_NOISE)  // Custom noise: 3%. By default: 0.5%.
  .add_float64_axis("Skew", SKEW_RANGE);
