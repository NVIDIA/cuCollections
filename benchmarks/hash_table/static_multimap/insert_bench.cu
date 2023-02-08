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

#include <key_generator.hpp>

#include <cuco/static_multimap.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

/**
 * @brief A benchmark evaluating multi-value `insert` performance:
 * - Total number of insertions: 100'000'000
 * - CG size: 8
 */
template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_static_multimap_insert(
  nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  using pair_type = cuco::pair_type<Key, Value>;

  auto const num_keys  = state.get_int64("NumInputs");
  auto const occupancy = state.get_float64("Occupancy");

  std::size_t const size = num_keys / occupancy;

  thrust::device_vector<Key> keys(num_keys);

  key_generator gen;
  gen.generate<Dist>(state, thrust::device, keys.begin(), keys.end());

  thrust::device_vector<pair_type> pairs(num_keys);
  thrust::transform(thrust::device,
                    keys.begin(),
                    keys.end(),
                    pairs.begin(),
                    [] __host__ __device__(Key const& key) {
                      return thrust::raw_reference_cast(pair_type(key, 42));
                    });

  state.add_element_count(num_keys);
  state.set_global_memory_rw_bytes(num_keys * sizeof(pair_type));

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               cuco::static_multimap<Key, Value> map{
                 size, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};

               // Use timers to explicitly mark the target region
               timer.start();
               map.insert(pairs.begin(), pairs.end(), launch.get_stream());
               timer.stop();
             });
}

template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_static_multimap_insert(
  nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  state.skip("Key should be the same type as Value.");
}

using key_type   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using value_type = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;

NVBENCH_BENCH_TYPES(nvbench_static_multimap_insert,
                    NVBENCH_TYPE_AXES(key_type, value_type, nvbench::type_list<dist_type::unique>))
  .set_name("staic_multimap_insert_unique_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1));

NVBENCH_BENCH_TYPES(nvbench_static_multimap_insert,
                    NVBENCH_TYPE_AXES(key_type, value_type, nvbench::type_list<dist_type::uniform>))
  .set_name("static_multimap_insert_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_int64_axis("Multiplicity", {1, 2, 4, 8, 16, 32, 64, 128, 256});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_insert,
                    NVBENCH_TYPE_AXES(key_type,
                                      value_type,
                                      nvbench::type_list<dist_type::gaussian>))
  .set_name("static_multimap_insert_gaussian_skew")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("Skew", {0.0001, 0.001, 0.01, 0.1, 1});
