/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "defaults.hpp"
#include "utils.hpp"

#include <benchmark_defaults.hpp>
#include <benchmark_utils.hpp>

#include <cuco/bloom_filter.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/std/limits>
#include <thrust/device_vector.h>

#include <exception>

using namespace cuco::benchmark;  // defaults, dist_from_state, rebind_hasher_t, add_fpr_summary
using namespace cuco::utility;    // key_generator, distribution

/**
 * @brief A benchmark evaluating `cuco::bloom_filter::contains_async` performance
 */
template <typename Key, typename Hash, typename Word, nvbench::int32_t WordsPerBlock, typename Dist>
void bloom_filter_contains(
  nvbench::state& state,
  nvbench::type_list<Key, Hash, Word, nvbench::enum_type<WordsPerBlock>, Dist>)
{
  // cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 32); // slightly improves peformance if
  // filter block fits into a 32B sector
  using policy_type = cuco::bloom_filter_policy<rebind_hasher_t<Hash, Key>,
                                                Word,
                                                static_cast<std::uint32_t>(WordsPerBlock)>;
  using filter_type =
    cuco::bloom_filter<Key, cuco::extent<size_t>, cuda::thread_scope_device, policy_type>;

  auto const num_keys       = state.get_int64("NumInputs");
  auto const filter_size_mb = state.get_int64("FilterSizeMB");
  auto const pattern_bits   = state.get_int64_or_default("PatternBits", WordsPerBlock);

  try {
    auto const policy = policy_type{static_cast<uint32_t>(pattern_bits)};
  } catch (std::exception const& e) {
    state.skip(e.what());  // skip invalid configurations
  }

  std::size_t const num_sub_filters =
    (filter_size_mb * 1024 * 1024) /
    (sizeof(typename filter_type::word_type) * filter_type::words_per_block);

  thrust::device_vector<Key> keys(num_keys);
  thrust::device_vector<bool> result(num_keys, false);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  state.add_element_count(num_keys);

  filter_type filter{num_sub_filters, {}, {static_cast<uint32_t>(pattern_bits)}};

  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

  add_fpr_summary(state, filter);

  filter.add(keys.begin(), keys.end());

  state.exec([&](nvbench::launch& launch) {
    filter.contains_async(keys.begin(), keys.end(), result.begin(), {launch.get_stream()});
  });
}

NVBENCH_BENCH_TYPES(bloom_filter_contains,
                    NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
                                      nvbench::type_list<defaults::BF_HASH>,
                                      nvbench::type_list<defaults::BF_WORD>,
                                      nvbench::enum_type_list<defaults::BF_WORDS_PER_BLOCK>,
                                      nvbench::type_list<distribution::unique>))
  .set_name("bloom_filter_contains_unique_size")
  .set_type_axes_names({"Key", "Hash", "Word", "WordsPerBlock", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_CACHE);

NVBENCH_BENCH_TYPES(bloom_filter_contains,
                    NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
                                      defaults::HASH_RANGE,
                                      nvbench::type_list<defaults::BF_WORD>,
                                      nvbench::enum_type_list<defaults::BF_WORDS_PER_BLOCK>,
                                      nvbench::type_list<distribution::unique>))
  .set_name("bloom_filter_contains_unique_hash")
  .set_type_axes_names({"Key", "Hash", "Word", "WordsPerBlock", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", {defaults::BF_SIZE_MB});

NVBENCH_BENCH_TYPES(bloom_filter_contains,
                    NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
                                      nvbench::type_list<defaults::BF_HASH>,
                                      nvbench::type_list<nvbench::uint32_t, nvbench::uint64_t>,
                                      nvbench::enum_type_list<1, 2, 4, 8>,
                                      nvbench::type_list<distribution::unique>))
  .set_name("bloom_filter_contains_unique_block_dim")
  .set_type_axes_names({"Key", "Hash", "Word", "WordsPerBlock", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", {defaults::BF_SIZE_MB});