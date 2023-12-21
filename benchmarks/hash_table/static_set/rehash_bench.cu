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
#include <utils.hpp>

#include <cuco/static_set.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>

/**
 * @brief A benchmark evaluating `cuco::static_set::rehash` performance
 */
template <typename Key, typename Dist>
void static_set_rehash(nvbench::state& state, nvbench::type_list<Key, Dist>)
{
  std::size_t const capacity = state.get_int64_or_default("Capacity", cuco::benchmark::defaults::N);
  auto const occupancy =
    state.get_float64_or_default("Occupancy", cuco::benchmark::defaults::OCCUPANCY);

  std::size_t const num_keys = capacity * occupancy;

  thrust::device_vector<Key> keys(num_keys);  // slots per second

  cuco::utility::key_generator gen;
  gen.generate(cuco::benchmark::dist_from_state<Dist>(state), keys.begin(), keys.end());

  state.add_element_count(capacity);

  cuco::experimental::static_set<Key> set{capacity, cuco::empty_key<Key>{-1}};

  set.insert(keys.begin(), keys.end());

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { set.rehash({launch.get_stream()}); });
}

NVBENCH_BENCH_TYPES(static_set_rehash,
                    NVBENCH_TYPE_AXES(cuco::benchmark::defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<cuco::utility::distribution::unique>))
  .set_name("static_set_rehash_unique_occupancy")
  .set_type_axes_names({"Key", "Distribution"})
  .set_max_noise(cuco::benchmark::defaults::MAX_NOISE)
  .add_float64_axis("Occupancy", cuco::benchmark::defaults::OCCUPANCY_RANGE);
