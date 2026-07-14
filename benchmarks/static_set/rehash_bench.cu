/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark_defaults.hpp>
#include <benchmark_utils.hpp>

#include <cuco/static_set.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>

using namespace cuco::benchmark;  // defaults, dist_from_state
using namespace cuco::utility;    // key_generator, distribution

/**
 * @brief A benchmark evaluating `cuco::static_set::rehash` performance
 */
template <typename Key, typename Dist>
void static_set_rehash(nvbench::state& state, nvbench::type_list<Key, Dist>)
{
  std::size_t const capacity = state.get_int64("Capacity");
  auto const occupancy       = state.get_float64("Occupancy");

  std::size_t const num_keys = capacity * occupancy;

  thrust::device_vector<Key> keys(num_keys);  // slots per second

  [[maybe_unused]] key_generator gen{};
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  state.add_element_count(capacity);

  cuco::static_set<Key> set{capacity, cuco::empty_key<Key>{-1}};

  set.insert(keys.begin(), keys.end());

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { set.rehash({launch.get_stream()}); });
}

NVBENCH_BENCH_TYPES(static_set_rehash,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<distribution::unique>))
  .set_name("static_set_rehash_unique_occupancy")
  .set_type_axes_names({"Key", "Distribution"})
  .add_int64_axis("Capacity", {defaults::N})
  .add_float64_axis("Occupancy", defaults::OCCUPANCY_RANGE);
