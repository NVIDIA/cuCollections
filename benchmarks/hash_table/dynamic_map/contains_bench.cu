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

#include <cuco/dynamic_map.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

using namespace cuco::benchmark;
using namespace cuco::utility;

/**
 * @brief A benchmark evaluating `cuco::dynamic_map::contains` performance
 */
template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> dynamic_map_contains(
  nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  using pair_type = cuco::pair<Key, Value>;

  auto const num_keys      = state.get_int64_or_default("NumInputs", defaults::N);
  auto const initial_size  = state.get_int64_or_default("InitSize", defaults::INITIAL_SIZE);
  auto const matching_rate = state.get_float64_or_default("MatchingRate", defaults::MATCHING_RATE);

  thrust::device_vector<Key> keys(num_keys);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  thrust::device_vector<pair_type> pairs(num_keys);
  thrust::transform(keys.begin(), keys.end(), pairs.begin(), [] __device__(Key const& key) {
    return pair_type(key, {});
  });

  cuco::dynamic_map<Key, Value> map{
    static_cast<size_t>(initial_size), cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};
  map.insert(pairs.begin(), pairs.end());

  gen.dropout(keys.begin(), keys.end(), matching_rate);

  thrust::device_vector<bool> result(num_keys);

  state.add_element_count(num_keys);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    map.contains(keys.begin(), keys.end(), result.begin(), {}, {}, launch.get_stream());
  });
}

template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> dynamic_map_contains(
  nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  state.skip("Key should be the same type as Value.");
}

NVBENCH_BENCH_TYPES(dynamic_map_contains,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      defaults::VALUE_TYPE_RANGE,
                                      nvbench::type_list<distribution::unique>))
  .set_name("dynamic_map_contains_unique_num_inputs")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", defaults::N_RANGE);

NVBENCH_BENCH_TYPES(dynamic_map_contains,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      defaults::VALUE_TYPE_RANGE,
                                      nvbench::type_list<distribution::unique>))
  .set_name("dynamic_map_contains_unique_matching_rate")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_float64_axis("MatchingRate", defaults::MATCHING_RATE_RANGE);
