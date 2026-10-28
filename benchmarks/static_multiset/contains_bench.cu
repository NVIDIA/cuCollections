/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark_defaults.hpp>
#include <benchmark_utils.hpp>

#include <cuco/static_multiset.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>

#include <memory_stats.hpp>

using namespace cuco::benchmark;  // defaults, dist_from_state, memory_stats
using namespace cuco::utility;    // key_generator, distribution

/**
 * @brief A benchmark evaluating `cuco::static_multiset::contains_async` performance
 */
template <typename Key, typename Dist>
void static_multiset_contains(nvbench::state& state, nvbench::type_list<Key, Dist>)
{
  auto const num_keys      = state.get_int64("NumInputs");
  auto const occupancy     = state.get_float64("Occupancy");
  auto const matching_rate = state.get_float64("MatchingRate");

  std::size_t const size = num_keys / occupancy;

  thrust::device_vector<Key> keys(num_keys);

  [[maybe_unused]] key_generator gen{};
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  cuco::static_multiset set{cuco::extent<std::size_t>{size},
                            cuco::empty_key<Key>{-1},
                            {},
                            {},
                            {},
                            {},
                            tracking_allocator<Key>{}};
  set.insert(keys.begin(), keys.end());

  gen.dropout(keys.begin(), keys.end(), matching_rate);

  thrust::device_vector<bool> result(num_keys);

  state.add_element_count(num_keys);

  auto const mem_stats_logger = cuco::benchmark::memory_stats_logger();
  state.exec([&](nvbench::launch& launch) {
    set.contains_async(keys.begin(), keys.end(), result.begin(), {launch.get_stream()});
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH_TYPES(static_multiset_contains,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<distribution::unique>))
  .set_name("static_multiset_constains_unique_capacity")
  .set_type_axes_names({"Key", "Distribution"})
  .add_int64_axis("NumInputs", defaults::N_RANGE_CACHE)
  .add_float64_axis("Occupancy", {defaults::OCCUPANCY})
  .add_float64_axis("MatchingRate", {defaults::MATCHING_RATE});

NVBENCH_BENCH_TYPES(static_multiset_contains,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<distribution::unique>))
  .set_name("static_multiset_contains_unique_occupancy")
  .set_type_axes_names({"Key", "Distribution"})
  .add_int64_axis("NumInputs", {defaults::N})
  .add_float64_axis("Occupancy", defaults::OCCUPANCY_RANGE)
  .add_float64_axis("MatchingRate", {defaults::MATCHING_RATE});

NVBENCH_BENCH_TYPES(static_multiset_contains,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<distribution::unique>))
  .set_name("static_multiset_contains_unique_matching_rate")
  .set_type_axes_names({"Key", "Distribution"})
  .add_int64_axis("NumInputs", {defaults::N})
  .add_float64_axis("Occupancy", {defaults::OCCUPANCY})
  .add_float64_axis("MatchingRate", defaults::MATCHING_RATE_RANGE);
