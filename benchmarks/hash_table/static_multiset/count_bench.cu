/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuco/static_multiset.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

using namespace cuco::benchmark;
using namespace cuco::utility;

template <typename Key, typename Dist>
void static_multimap_count(nvbench::state& state, nvbench::type_list<Key, Dist>)
{
  auto const num_keys      = state.get_int64_or_default("NumInputs", defaults::N);
  auto const occupancy     = state.get_float64_or_default("Occupancy", defaults::OCCUPANCY);
  auto const matching_rate = state.get_float64_or_default("MatchingRate", defaults::MATCHING_RATE);

  std::size_t const size = num_keys / occupancy;

  thrust::device_vector<Key> keys(num_keys);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  auto map = cuco::static_multiset{size, cuco::empty_key<Key>{-1}};
  map.insert(keys.begin(), keys.end());

  gen.dropout(keys.begin(), keys.end(), matching_rate);

  state.add_element_count(num_keys);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto const count = map.count(keys.begin(), keys.end(), {launch.get_stream()});
  });
}

NVBENCH_BENCH_TYPES(static_multimap_count,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<distribution::uniform>))
  .set_name("static_multimap_count_uniform_occupancy")
  .set_type_axes_names({"Key", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_float64_axis("Occupancy", defaults::OCCUPANCY_RANGE);

NVBENCH_BENCH_TYPES(static_multimap_count,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<distribution::uniform>))
  .set_name("static_multimap_count_uniform_matching_rate")
  .set_type_axes_names({"Key", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_float64_axis("MatchingRate", defaults::MATCHING_RATE_RANGE);

NVBENCH_BENCH_TYPES(static_multimap_count,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<distribution::uniform>))
  .set_name("static_multimap_count_uniform_multiplicity")
  .set_type_axes_names({"Key", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("Multiplicity", defaults::MULTIPLICITY_RANGE);
