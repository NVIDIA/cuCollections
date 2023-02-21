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
#include <distribution.hpp>
#include <key_generator.hpp>
#include <utils.hpp>

#include <cuco/static_map.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>

using namespace cuco::benchmark;
using namespace cuco::benchmark::defaults;

/**
 * @brief A benchmark evaluating `contains` performance:
 */
template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> static_map_contains(
  nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  auto const num_keys      = state.get_int64_or_default("NumInputs", N);
  auto const occupancy     = state.get_float64_or_default("Occupancy", OCCUPANCY);
  auto const matching_rate = state.get_float64_or_default("MatchingRate", MATCHING_RATE);

  std::size_t const size = num_keys / occupancy;

  thrust::device_vector<Key> keys(num_keys);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  auto pairs_begin = thrust::make_transform_iterator(
    keys.begin(), [] __device__(auto i) { return cuco::pair_type<Key, Value>(i, {}); });

  cuco::static_map<Key, Value> map{size, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};
  map.insert(pairs_begin, pairs_begin + num_keys);
  CUCO_CUDA_TRY(cudaStreamSynchronize(nullptr));

  gen.dropout(keys.begin(), keys.end(), matching_rate);

  state.add_element_count(num_keys, "NumInputs");
  state.set_global_memory_rw_bytes(num_keys * sizeof(Key));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    map.contains(keys.begin(),
                 keys.end(),
                 thrust::make_discard_iterator(),
                 cuco::murmurhash3_32<Key>{},
                 thrust::equal_to<Key>{},
                 launch.get_stream());
  });
}

template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> static_map_contains(
  nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  state.skip("Key should be the same type as Value.");
}

NVBENCH_BENCH_TYPES(static_map_contains,
                    NVBENCH_TYPE_AXES(KEY_TYPE_RANGE,
                                      VALUE_TYPE_RANGE,
                                      nvbench::type_list<dist_type::unique>))
  .set_name("static_map_contains_unique_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_timeout(100)          // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(MAX_NOISE)  // Custom noise: 3%. By default: 0.5%.
  .add_float64_axis("Occupancy", OCCUPANCY_RANGE);

NVBENCH_BENCH_TYPES(static_map_contains,
                    NVBENCH_TYPE_AXES(KEY_TYPE_RANGE,
                                      VALUE_TYPE_RANGE,
                                      nvbench::type_list<dist_type::unique>))
  .set_name("static_map_contains_unique_matching_rate")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_timeout(100)          // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(MAX_NOISE)  // Custom noise: 3%. By default: 0.5%.
  .add_float64_axis("MatchingRate", MATCHING_RATE_RANGE);
