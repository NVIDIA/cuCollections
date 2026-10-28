/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark_defaults.hpp>
#include <benchmark_utils.hpp>

#include <cuco/dynamic_map.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <memory_stats.hpp>

#include <algorithm>

using namespace cuco::benchmark;  // defaults, dist_from_state, memory_stats
using namespace cuco::utility;    // key_generator, distribution

/**
 * @brief A benchmark evaluating `cuco::dynamic_map::erase` performance
 */
template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> dynamic_map_erase(
  nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  using pair_type = cuco::pair<Key, Value>;

  auto const num_keys      = state.get_int64("NumInputs");
  auto const initial_size  = state.get_int64("InitSize");
  auto const matching_rate = state.get_float64("MatchingRate");

  thrust::device_vector<Key> keys(num_keys);

  [[maybe_unused]] key_generator gen{};
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  thrust::device_vector<pair_type> pairs(num_keys);
  thrust::transform(
    keys.begin(), keys.end(), pairs.begin(), [] __device__(auto i) { return pair_type(i, {}); });

  gen.dropout(keys.begin(), keys.end(), matching_rate);

  state.add_element_count(num_keys);

  std::size_t peak_memory_usage = 0;
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      with_tracking_allocator_t<cuco::dynamic_map<Key, Value>> map{
        static_cast<size_t>(initial_size),
        cuco::empty_key<Key>{-1},
        cuco::empty_value<Value>{-1},
        cuco::erased_key<Key>{-2}};
      map.insert(pairs.begin(), pairs.end(), {launch.get_stream()});

      auto const mem_stats_logger = cuco::benchmark::memory_stats_logger();
      timer.start();
      map.erase(keys.begin(), keys.end(), {launch.get_stream()});
      timer.stop();
      peak_memory_usage = std::max(peak_memory_usage, mem_stats_logger.peak_memory_usage());
    });
  state.add_buffer_size(peak_memory_usage, "peak_memory_usage", "peak_memory_usage");
}

template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> dynamic_map_erase(
  nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  state.skip("Key should be the same type as Value.");
}

NVBENCH_BENCH_TYPES(dynamic_map_erase,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      defaults::VALUE_TYPE_RANGE,
                                      nvbench::type_list<distribution::unique>))
  .set_name("dynamic_map_erase_unique_capacity")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .add_int64_axis("NumInputs", defaults::N_RANGE)
  .add_int64_axis("InitSize", {defaults::INITIAL_SIZE})
  .add_float64_axis("MatchingRate", {defaults::MATCHING_RATE});

NVBENCH_BENCH_TYPES(dynamic_map_erase,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      defaults::VALUE_TYPE_RANGE,
                                      nvbench::type_list<distribution::unique>))
  .set_name("dynamic_map_erase_unique_matching_rate")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .add_int64_axis("NumInputs", {defaults::N})
  .add_int64_axis("InitSize", {defaults::INITIAL_SIZE})
  .add_float64_axis("MatchingRate", defaults::MATCHING_RATE_RANGE);
