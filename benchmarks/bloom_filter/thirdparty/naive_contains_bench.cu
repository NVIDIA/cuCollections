/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "../defaults.hpp"
#include "naive_bloom_filter.cuh"

#include <benchmark_defaults.hpp>

#include <nvbench/nvbench.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

using namespace cuco::benchmark;  // defaults

/**
 * @brief A benchmark evaluating `naive_bloom_filter::contains` performance
 */
template <typename Key, nvbench::int32_t NumHashes>
void naive_bloom_filter_contains(nvbench::state& state,
                                 nvbench::type_list<Key, nvbench::enum_type<NumHashes>>)
{
  using filter_type = naive_bloom_filter<Key>;

  auto const num_keys = state.get_int64("NumInputs");
  state.add_element_count(num_keys);

  auto const filter_size_mb = state.get_int64("FilterSizeMB");
  std::size_t num_bits      = filter_size_mb * 1024 * 1024 * 8;
  filter_type filter{num_bits, NumHashes};

  thrust::counting_iterator<Key> key_it(0);

  // insert FPR-optimal number of keys
  auto const num_build_keys = (filter_size_mb * 1024 * 1024 * 8) / (2 * NumHashes);
  filter.add(key_it, key_it + num_build_keys);

  // FPR summary
  thrust::device_vector<bool> result(num_keys, false);
  filter.contains(key_it + num_build_keys, key_it + num_build_keys + num_keys, result.begin());

  double const fp = thrust::count(thrust::device, result.begin(), result.end(), true);

  auto& summ_fpr = state.add_summary("FalsePositiveRate");
  summ_fpr.set_string("hint", "FPR");
  summ_fpr.set_string("short_name", "FPR");
  summ_fpr.set_string("description", "False-positive rate of the bloom filter.");
  summ_fpr.set_float64("value", fp / static_cast<double>(num_keys));

  state.collect_dram_throughput();
  state.collect_l2_hit_rates();

  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end(), 0);

  state.exec([&](nvbench::launch& launch) {
    filter.contains_async(keys.begin(), keys.end(), result.begin(), launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(naive_bloom_filter_contains,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::uint64_t>,  ///< Key
                                      nvbench::enum_type_list<16>             ///< NumHashes
                                      ))
  .set_name("naive_bloom_filter_contains_unique_size_u64")
  .set_type_axes_names({"Key", "NumHashes"})
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_FRONTIER_CACHE);