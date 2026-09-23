/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../defaults.hpp"
#include "cbf.cuh"

#include <nvbench/nvbench.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

using namespace cuco::benchmark;  // cbf, defaults

/**
 * @brief A benchmark evaluating GPU CBF lookup performance
 */
template <typename Key, nvbench::int32_t NumHashes>
void cbf_contains(nvbench::state& state, nvbench::type_list<Key, nvbench::enum_type<NumHashes>>)
{
  using filter_type = cbf<Key>;

  auto const num_keys            = state.get_int64("NumInputs");
  auto const all_positive_lookup = state.get_int64("AllPositiveLookup") != 0;
  state.add_element_count(num_keys);

  auto const filter_size_mb = state.get_int64("FilterSizeMB");
  std::size_t num_bits      = filter_size_mb * 1024 * 1024 * 8;

  auto const num_build_keys = num_bits / (2 * NumHashes);
  thrust::counting_iterator<Key> key_it{0};
  {
    filter_type filter{num_bits, NumHashes};
    filter.add(key_it, key_it + num_build_keys);

    thrust::device_vector<bool> result(num_keys, false);
    filter.contains(key_it + num_build_keys, key_it + num_build_keys + num_keys, result.begin());

    double const false_positives =
      thrust::count(thrust::device, result.begin(), result.end(), true);
    auto& summary = state.add_summary("FalsePositiveRate");
    summary.set_string("hint", "FPR");
    summary.set_string("short_name", "FPR");
    summary.set_string("description", "False-positive rate of the Bloom filter.");
    summary.set_float64("value", false_positives / static_cast<double>(num_keys));
  }

  filter_type filter{num_bits, NumHashes};
  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end(), 0);
  if (all_positive_lookup) {
    filter.add(keys.begin(), keys.end());
  } else {
    // Reproduce the evaluated workload: inserted keys are positive, while the remaining
    // sequential keys are negative and may exit before evaluating all hash positions.
    filter.add(key_it, key_it + num_build_keys);
  }
  thrust::device_vector<bool> result(num_keys, false);

  state.exec([&](nvbench::launch& launch) {
    filter.contains_async(keys.begin(), keys.end(), result.begin(), launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(cbf_contains,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::uint64_t>,  ///< Key
                                      nvbench::enum_type_list<16>             ///< NumHashes
                                      ))
  .set_name("cbf_contains_unique_size_u64")
  .set_type_axes_names({"Key", "NumHashes"})
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", {32, 1024})
  .add_int64_axis("AllPositiveLookup", {0});
