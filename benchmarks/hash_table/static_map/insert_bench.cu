/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include <utils.hpp>

#include <cuco/static_map.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

using namespace cuco::benchmark;
using namespace cuco::benchmark::defaults;

/**
 * @brief A benchmark evaluating `insert` performance:
 */
template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> static_map_insert(
  nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  using pair_type = cuco::pair_type<Key, Value>;

  auto const num_keys  = state.get_int64_or_default("NumInputs", defaults::N);
  auto const occupancy = state.get_float64_or_default("Occupancy", defaults::OCCUPANCY);

  std::size_t const size = num_keys / occupancy;

  thrust::device_vector<Key> keys(num_keys);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  thrust::device_vector<pair_type> pairs(num_keys);
  thrust::transform(
    thrust::device, keys.begin(), keys.end(), pairs.begin(), [] __device__(Key const& key) {
      return pair_type(key, {});
    });

  state.add_element_count(num_keys, "NumInputs");
  state.set_global_memory_rw_bytes(num_keys * sizeof(pair_type));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               cuco::static_map<Key, Value> map{size,
                                                cuco::empty_key<Key>{-1},
                                                cuco::empty_value<Value>{-1},
                                                cuco::cuda_allocator<char>{},
                                                launch.get_stream()};

               // Use timers to explicitly mark the target region
               timer.start();
               map.insert(pairs.begin(),
                          pairs.end(),
                          cuco::murmurhash3_32<Key>{},
                          thrust::equal_to<Key>{},
                          launch.get_stream());
               timer.stop();
             });
}

template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> static_map_insert(
  nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  state.skip("Key should be the same type as Value.");
}

NVBENCH_BENCH_TYPES(static_map_insert,
                    NVBENCH_TYPE_AXES(KEY_TYPE_RANGE,
                                      VALUE_TYPE_RANGE,
                                      nvbench::type_list<dist_type::uniform>))
  .set_name("static_map_insert_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_max_noise(MAX_NOISE)  // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("Multiplicity", MULTIPLICITY_RANGE);

NVBENCH_BENCH_TYPES(static_map_insert,
                    NVBENCH_TYPE_AXES(KEY_TYPE_RANGE,
                                      VALUE_TYPE_RANGE,
                                      nvbench::type_list<dist_type::unique>))
  .set_name("static_map_insert_unique_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_max_noise(MAX_NOISE)  // Custom noise: 3%. By default: 0.5%.
  .add_float64_axis("Occupancy", OCCUPANCY_RANGE);

NVBENCH_BENCH_TYPES(static_map_insert,
                    NVBENCH_TYPE_AXES(KEY_TYPE_RANGE,
                                      VALUE_TYPE_RANGE,
                                      nvbench::type_list<dist_type::gaussian>))
  .set_name("static_map_insert_gaussian")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_max_noise(MAX_NOISE)  // Custom noise: 3%. By default: 0.5%.
  .add_float64_axis("Skew", SKEW_RANGE);
