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

#include <random>

#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

#include <cuco/static_multimap.cuh>
#include <key_generator.hpp>

/**
 * @brief A benchmark evaluating multi-value query (`count` + `retrieve`) performance:
 * - Total number of insertions: 100'000'000
 * - CG size: 8
 */
template <typename Key, typename Value>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_static_multimap_query(
  nvbench::state& state, nvbench::type_list<Key, Value>)
{
  auto const num_keys      = state.get_int64("NumInputs");
  auto const occupancy     = state.get_float64("Occupancy");
  auto const dist          = state.get_string("Distribution");
  auto const multiplicity  = state.get_int64_or_default("Multiplicity", 8);
  auto const matching_rate = state.get_float64("MatchingRate");

  std::size_t const size = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Key>(dist, h_keys.begin(), h_keys.end(), multiplicity);

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  generate_probe_keys<Key>(matching_rate, h_keys.begin(), h_keys.end());

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  state.add_element_count(num_keys, "NumKeys");

  cuco::static_multimap<Key, Value> map{size, -1, -1};
  map.insert(d_pairs.begin(), d_pairs.end());

  auto const output_size = map.count_outer(d_keys.begin(), d_keys.end());
  thrust::device_vector<cuco::pair_type<Key, Value>> d_results(output_size);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto count = map.count_outer(d_keys.begin(), d_keys.end(), launch.get_stream());
    map.retrieve_outer(d_keys.begin(), d_keys.end(), d_results.data().get(), launch.get_stream());
  });
}

template <typename Key, typename Value>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_static_multimap_query(
  nvbench::state& state, nvbench::type_list<Key, Value>)
{
  state.skip("Key should be the same type as Value.");
}

using key_type   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using value_type = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;

NVBENCH_BENCH_TYPES(nvbench_static_multimap_query, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_name("static_multimap_query_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_int64_axis("Multiplicity",
                  {1, 2, 4, 8, 16, 32, 64, 128, 256})  // only applies to uniform distribution
  .add_string_axis("Distribution", {"UNIFORM"})
  .add_float64_axis("MatchingRate", {0.5});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_query, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_name("static_multimap_query_occupancy")
  .set_type_axes_names({"Key", "Value"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1))
  .add_int64_axis("Multiplicity", {8})  // only applies to uniform distribution
  .add_string_axis("Distribution", {"GAUSSIAN", "GEOMETRIC", "UNIFORM"})
  .add_float64_axis("MatchingRate", {0.5});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_query, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_name("static_multimap_query_matching_rate")
  .set_type_axes_names({"Key", "Value"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_int64_axis("Multiplicity", {8})  // only applies to uniform distribution
  .add_string_axis("Distribution", {"GAUSSIAN", "GEOMETRIC", "UNIFORM"})
  .add_float64_axis("MatchingRate", {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1});
