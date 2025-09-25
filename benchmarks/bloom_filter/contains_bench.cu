/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <nvbench/nvbench.cuh>

#include <cuda/std/limits>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <exception>
#include <limits>

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
  using size_type   = std::uint32_t;
  using policy_type = cuco::default_filter_policy<rebind_hasher_t<Hash, Key>,
                                                  Word,
                                                  static_cast<std::uint32_t>(WordsPerBlock)>;
  using filter_type =
    cuco::bloom_filter<Key, cuco::extent<size_type>, cuda::thread_scope_device, policy_type>;

  constexpr auto filter_block_size =
    sizeof(typename filter_type::word_type) * filter_type::words_per_block;

  // if (filter_block_size <= 32) {
  //   cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 32); // slightly improves peformance if
  //   filter block fits into a 32B sector
  // }

  auto const num_keys       = state.get_int64("NumInputs");
  auto const filter_size_mb = state.get_int64("FilterSizeMB");
  auto const pattern_bits   = WordsPerBlock;

  try {
    [[maybe_unused]] auto const policy = policy_type{static_cast<uint32_t>(pattern_bits)};
  } catch (std::exception const& e) {
    state.skip(e.what());  // skip invalid configurations
  }

  std::size_t const num_sub_filters = (filter_size_mb * 1024 * 1024) / filter_block_size;
  std::size_t const num_build_keys  = (filter_size_mb * 1024 * 1024 * 8) / (2 * WordsPerBlock);

  if (num_sub_filters > std::numeric_limits<size_type>::max()) {
    state.skip("num_sub_filters too large for size_type");  // skip invalid configurations
  }

  thrust::device_vector<Key> build_keys(num_build_keys);
  thrust::sequence(build_keys.begin(),
                   build_keys.end(),
                   static_cast<int64_t>(0),
                   static_cast<int64_t>(num_keys / num_build_keys));
  thrust::counting_iterator<Key> keys(0);
  thrust::device_vector<bool> result(num_keys, false);

  state.add_element_count(num_keys);

  filter_type filter{
    static_cast<size_type>(num_sub_filters), {}, {static_cast<uint32_t>(pattern_bits)}};

  state.collect_dram_throughput();
  state.collect_l2_hit_rates();

  add_fpr_summary(state, filter);

  filter.add(build_keys.begin(), build_keys.end());

  state.exec([&](nvbench::launch& launch) {
    filter.contains_async(keys, keys + num_keys, result.begin(), {launch.get_stream()});
  });
}

/**
 * @brief A benchmark evaluating `cuco::bloom_filter::contains_async` performance with
 * `arrow_filter_policy`
 */
template <typename Key, typename Dist>
void arrow_bloom_filter_contains(nvbench::state& state, nvbench::type_list<Key, Dist>)
{
  // cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 32); // slightly improves peformance if
  // filter block fits into a 32B sector
  using size_type   = std::uint32_t;
  using policy_type = cuco::arrow_filter_policy<Key>;
  using filter_type =
    cuco::bloom_filter<Key, cuco::extent<size_type>, cuda::thread_scope_device, policy_type>;

  auto const num_keys       = state.get_int64("NumInputs");
  auto const filter_size_mb = state.get_int64("FilterSizeMB");

  std::size_t const num_sub_filters =
    (filter_size_mb * 1024 * 1024) /
    (sizeof(typename filter_type::word_type) * filter_type::words_per_block);
  std::size_t const num_build_keys =
    (filter_size_mb * 1024 * 1024 * 8) / (2 * policy_type::bits_set_per_block);

  if (num_sub_filters > policy_type::max_filter_blocks) {
    state.skip("bloom filter with arrow policy should have <= 4194304 blocks");  // skip invalid
                                                                                 // configurations
  }

  thrust::device_vector<Key> build_keys(num_build_keys);
  thrust::sequence(build_keys.begin(),
                   build_keys.end(),
                   static_cast<int64_t>(0),
                   static_cast<int64_t>(num_keys / num_build_keys));
  thrust::counting_iterator<Key> keys(0);
  thrust::device_vector<bool> result(num_keys, false);

  state.add_element_count(num_keys);

  filter_type filter{static_cast<size_type>(num_sub_filters)};

  state.collect_dram_throughput();
  state.collect_l2_hit_rates();

  add_fpr_summary(state, filter);

  filter.add(build_keys.begin(), build_keys.end());

  state.exec([&](nvbench::launch& launch) {
    filter.contains_async(keys, keys + num_keys, result.begin(), {launch.get_stream()});
  });
}

/**
 * @brief A benchmark evaluating `cuco::bloom_filter::contains_async` performance with
 * `parametric_filter_policy`
 */
template <typename Key,
          typename Word,
          nvbench::int32_t WordsPerBlock,
          nvbench::int32_t PatternBits,
          nvbench::int32_t AddHorizontalLayout,
          nvbench::int32_t AddVerticalLayout,
          nvbench::int32_t ContainsHorizontalLayout,
          nvbench::int32_t ContainsVerticalLayout,
          typename Dist>
void pfp_bloom_filter_contains(nvbench::state& state,
                               nvbench::type_list<Key,
                                                  Word,
                                                  nvbench::enum_type<WordsPerBlock>,
                                                  nvbench::enum_type<PatternBits>,
                                                  nvbench::enum_type<AddHorizontalLayout>,
                                                  nvbench::enum_type<AddVerticalLayout>,
                                                  nvbench::enum_type<ContainsHorizontalLayout>,
                                                  nvbench::enum_type<ContainsVerticalLayout>,
                                                  Dist>)
{
  // cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 32); // slightly improves peformance if
  // filter block fits into a 32B sector
  using size_type   = std::uint32_t;
  using hasher      = cuco::xxhash_64<Key>;
  using policy_type = cuco::experimental::detail::parametric_filter_policy<hasher,
                                                                           Word,
                                                                           WordsPerBlock,
                                                                           PatternBits,
                                                                           AddHorizontalLayout,
                                                                           AddVerticalLayout,
                                                                           ContainsHorizontalLayout,
                                                                           ContainsVerticalLayout>;
  using filter_type =
    cuco::bloom_filter<Key, cuco::extent<size_type>, cuda::thread_scope_device, policy_type>;

  auto const num_keys       = state.get_int64("NumInputs");
  auto const filter_size_mb = state.get_int64("FilterSizeMB");

  std::size_t const num_sub_filters =
    (filter_size_mb * 1024 * 1024) /
    (sizeof(typename filter_type::word_type) * filter_type::words_per_block);
  auto const num_build_keys = (filter_size_mb * 1024 * 1024 * 8) / (2 * PatternBits);

  if (num_sub_filters > std::numeric_limits<size_type>::max()) {
    // skip invalid configurations
    state.skip("bloom filter with parametric filter policy should not exceed 2^32 - 1 blocks");
  }

  thrust::device_vector<Key> build_keys(num_build_keys);
  thrust::sequence(build_keys.begin(),
                   build_keys.end(),
                   static_cast<int64_t>(0),
                   static_cast<int64_t>(num_keys / num_build_keys));
  thrust::counting_iterator<Key> keys(0);
  thrust::device_vector<bool> result(num_keys, false);

  state.add_element_count(num_keys);

  filter_type filter{static_cast<size_type>(num_sub_filters)};

  state.collect_dram_throughput();
  state.collect_l2_hit_rates();

  add_fpr_summary(state, filter);

  filter.add(build_keys.begin(), build_keys.end());

  state.exec([&](nvbench::launch& launch) {
    filter.contains_async(keys, keys + num_keys, result.begin(), {launch.get_stream()});
  });
}

// NVBENCH_BENCH_TYPES(bloom_filter_contains,
//                     NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
//                                       nvbench::type_list<defaults::BF_HASH>,
//                                       nvbench::type_list<defaults::BF_WORD>,
//                                       nvbench::enum_type_list<defaults::BF_WORDS_PER_BLOCK>,
//                                       nvbench::type_list<distribution::unique>))
//   .set_name("bloom_filter_contains_unique_size")
//   .set_type_axes_names({"Key", "Hash", "Word", "WordsPerBlock", "Distribution"})
//   .set_max_noise(defaults::MAX_NOISE)
//   .add_int64_axis("NumInputs", {defaults::BF_N})
//   .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_CACHE);

// NVBENCH_BENCH_TYPES(bloom_filter_contains,
//                     NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
//                                       defaults::HASH_RANGE,
//                                       nvbench::type_list<defaults::BF_WORD>,
//                                       nvbench::enum_type_list<defaults::BF_WORDS_PER_BLOCK>,
//                                       nvbench::type_list<distribution::unique>))
//   .set_name("bloom_filter_contains_unique_hash")
//   .set_type_axes_names({"Key", "Hash", "Word", "WordsPerBlock", "Distribution"})
//   .set_max_noise(defaults::MAX_NOISE)
//   .add_int64_axis("NumInputs", {defaults::BF_N})
//   .add_int64_axis("FilterSizeMB", {defaults::BF_SIZE_MB});

// NVBENCH_BENCH_TYPES(bloom_filter_contains,
//                     NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
//                                       nvbench::type_list<defaults::BF_HASH>,
//                                       nvbench::type_list<nvbench::uint32_t, nvbench::uint64_t>,
//                                       nvbench::enum_type_list<1, 2, 4, 8>,
//                                       nvbench::type_list<distribution::unique>))
//   .set_name("bloom_filter_contains_unique_block_dim")
//   .set_type_axes_names({"Key", "Hash", "Word", "WordsPerBlock", "Distribution"})
//   .set_max_noise(defaults::MAX_NOISE)
//   .add_int64_axis("NumInputs", {defaults::BF_N})
//   .add_int64_axis("FilterSizeMB", {defaults::BF_SIZE_MB});

// NVBENCH_BENCH_TYPES(arrow_bloom_filter_contains,
//                     NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
//                                       nvbench::type_list<distribution::unique>))
//   .set_name("arrow_bloom_filter_contains_unique_size")
//   .set_type_axes_names({"Key", "Distribution"})
//   .set_max_noise(defaults::MAX_NOISE)
//   .add_int64_axis("NumInputs", {defaults::BF_N})
//   .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_CACHE);

NVBENCH_BENCH_TYPES(pfp_bloom_filter_contains,
                    NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
                                      nvbench::type_list<defaults::BF_WORD>,
                                      nvbench::enum_type_list<defaults::BF_WORDS_PER_BLOCK>,
                                      nvbench::enum_type_list<defaults::BF_PATTERN_BITS>,
                                      nvbench::enum_type_list<8>,  ///< AddHorizontalLayout
                                      nvbench::enum_type_list<1>,  ///< AddVerticalLayout
                                      nvbench::enum_type_list<8>,  ///< ContainsHorizontalLayout
                                      nvbench::enum_type_list<1>,  ///< ContainsVerticalLayout
                                      nvbench::type_list<distribution::unique>))
  .set_name("pfp_bloom_filter_contains_unique_size")
  .set_type_axes_names({"Key",
                        "Word",
                        "WordsPerBlock",
                        "Pattern Bits",
                        "AddHorizontalLayout",
                        "AddVerticalLayout",
                        "ContainsHorizontalLayout",
                        "ContainsVerticalLayout",
                        "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_CACHE);