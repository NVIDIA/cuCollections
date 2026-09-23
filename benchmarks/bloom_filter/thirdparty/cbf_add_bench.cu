/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../defaults.hpp"
#include "cbf.cuh"

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

using namespace cuco::benchmark;  // cbf, defaults

/**
 * @brief A benchmark evaluating GPU CBF construction performance
 */
template <typename Key, nvbench::int32_t NumHashes>
void cbf_add(nvbench::state& state, nvbench::type_list<Key, nvbench::enum_type<NumHashes>>)
{
  using filter_type = cbf<Key>;

  auto const num_keys = state.get_int64("NumInputs");
  state.add_element_count(num_keys);

  auto const filter_size_mb  = state.get_int64("FilterSizeMB");
  std::size_t const num_bits = filter_size_mb * 1024 * 1024 * 8;
  filter_type filter{num_bits, NumHashes};

  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end(), 0);

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    filter.add_async(keys.begin(), keys.end(), launch.get_stream());
    timer.stop();
    filter.clear_async(launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(cbf_add,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::uint64_t>,  ///< Key
                                      nvbench::enum_type_list<16>             ///< NumHashes
                                      ))
  .set_name("cbf_add_unique_size_u64")
  .set_type_axes_names({"Key", "NumHashes"})
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", {32, 1024});
