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

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

using namespace cuco::benchmark;  // defaults

/**
 * @brief A benchmark evaluating `naive_bloom_filter::add` performance
 */
template <typename Key, nvbench::int32_t NumHashes>
void naive_bloom_filter_add(nvbench::state& state,
                            nvbench::type_list<Key, nvbench::enum_type<NumHashes>>)
{
  using filter_type = naive_bloom_filter<Key>;

  auto const num_keys = state.get_int64("NumInputs");
  state.add_element_count(num_keys);

  auto const filter_size_mb  = state.get_int64("FilterSizeMB");
  std::size_t const num_bits = filter_size_mb * 1024 * 1024 * 8;
  filter_type filter{num_bits, NumHashes};

  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(keys.begin(), keys.end(), 0);

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    filter.add_async(keys.begin(), keys.end(), launch.get_stream());
    timer.stop();
    filter.clear_async(launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(naive_bloom_filter_add,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::uint64_t>,  ///< Key
                                      nvbench::enum_type_list<16>             ///< NumHashes
                                      ))
  .set_name("naive_bloom_filter_add_unique_size_u64")
  .set_type_axes_names({"Key", "NumHashes"})
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_FRONTIER_CACHE);
