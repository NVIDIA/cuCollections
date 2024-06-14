/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <benchmark_defaults.hpp>
#include <benchmark_utils.hpp>

#include <cuco/static_set.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>

#include <hash_table/static_set_perfect/static_set_perfect_defaults.hpp>

using namespace cuco::benchmark;
using namespace cuco::utility;

/**
 * @brief A benchmark evaluating `cuco::static_set::size` performance
 */
template <typename Key, typename Dist>
void static_set_size(nvbench::state& state, nvbench::type_list<Key, Dist>)
{
  auto const occupancy        = state.get_float64_or_default("Occupancy", defaults::OCCUPANCY);
  auto const size             = perfect_static_set::defaults::CAPACITY;
  auto const default_num_keys = (std::uint64_t)size * occupancy;
  auto const num_keys         = state.get_int64_or_default("NumInputs", default_num_keys);

  thrust::device_vector<Key> keys(num_keys);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  state.add_element_count(num_keys);

  cuco::static_set<Key,
                   cuco::extent<std::size_t>,
                   cuda::thread_scope_device,
                   thrust::equal_to<Key>,
                   perfect_static_set::defaults::PROBER<Key>>
    set{size, cuco::empty_key<Key>{-1}};
  set.insert(keys.begin(), keys.end());

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto const size = set.size({launch.get_stream()}); });
}

NVBENCH_BENCH_TYPES(static_set_size,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<distribution::unique>))
  .set_name("static_set_size_unique_occupancy")
  .set_type_axes_names({"Key", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_float64_axis("Occupancy", defaults::OCCUPANCY_RANGE);
