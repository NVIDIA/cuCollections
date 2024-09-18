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

#include <benchmark_defaults.hpp>
#include <benchmark_utils.hpp>

#include <cuco/bloom_filter.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/std/limits>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

using namespace cuco::benchmark;  // defaults, dist_from_state, rebind_hasher_t
using namespace cuco::utility;    // key_generator, distribution

template <typename FilterType>
void add_fpr_summary(nvbench::state& state, FilterType& filter)
{
  filter.clear();

  auto const num_keys = state.get_int64("NumInputs");

  thrust::device_vector<typename FilterType::key_type> keys(num_keys * 2);
  thrust::sequence(thrust::device, keys.begin(), keys.end(), 1);
  thrust::device_vector<bool> result(num_keys, false);

  auto tp_begin = keys.begin();
  auto tp_end   = tp_begin + num_keys;
  auto tn_begin = tp_end;
  auto tn_end   = keys.end();
  filter.add(tp_begin, tp_end);
  filter.contains(tn_begin, tn_end, result.begin());

  float fp = thrust::count(thrust::device, result.begin(), result.end(), true);

  auto& summ = state.add_summary("FalsePositiveRate");
  summ.set_string("hint", "FPR");
  summ.set_string("short_name", "FPR");
  summ.set_string("description", "False-positive rate of the bloom filter.");
  summ.set_float64("value", fp / num_keys);

  filter.clear();
}

/**
 * @brief A benchmark evaluating `cuco::bloom_filter::add_async` performance
 */
template <typename Key, typename Hash, typename Block, typename Dist>
void bloom_filter_add(nvbench::state& state, nvbench::type_list<Key, Hash, Block, Dist>)
{
  using filter_type =
    cuco::bloom_filter<Key,
                       cuco::extent<size_t>,
                       cuda::thread_scope_device,
                       cuco::bloom_filter_policy<rebind_hasher_t<Hash, Key>, Block>>;

  auto const num_keys       = state.get_int64("NumInputs");
  auto const filter_size_mb = state.get_int64("FilterSizeMB");
  auto const pattern_bits   = state.get_int64("PatternBits");

  if (pattern_bits < filter_type::words_per_block and pattern_bits != defaults::BF_PATTERN_BITS) {
    state.skip("pattern_bits must be at least words_per_block");
  }

  std::size_t const num_sub_filters =
    (filter_size_mb * 1024 * 1024) /
    (sizeof(typename filter_type::word_type) * filter_type::words_per_block);

  thrust::device_vector<Key> keys(num_keys);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  state.add_element_count(num_keys);
  state.add_global_memory_writes<typename filter_type::word_type>(num_keys *
                                                                  filter_type::words_per_block);

  filter_type filter{num_sub_filters, {}, {static_cast<uint32_t>(pattern_bits)}};

  add_fpr_summary(state, filter);

  state.exec([&](nvbench::launch& launch) {
    filter.add_async(keys.begin(), keys.end(), {launch.get_stream()});
  });
}

/**
 * @brief A benchmark evaluating `cuco::bloom_filter::contains_async` performance
 */
template <typename Key, typename Hash, typename Block, typename Dist>
void bloom_filter_contains(nvbench::state& state, nvbench::type_list<Key, Hash, Block, Dist>)
{
  using filter_type =
    cuco::bloom_filter<Key,
                       cuco::extent<size_t>,
                       cuda::thread_scope_device,
                       cuco::bloom_filter_policy<rebind_hasher_t<Hash, Key>, Block>>;

  auto const num_keys       = state.get_int64("NumInputs");
  auto const filter_size_mb = state.get_int64("FilterSizeMB");
  auto const pattern_bits   = state.get_int64("PatternBits");

  if (pattern_bits < filter_type::words_per_block and pattern_bits != defaults::BF_PATTERN_BITS) {
    state.skip("pattern_bits must be at least words_per_block");
  }

  std::size_t const num_sub_filters =
    (filter_size_mb * 1024 * 1024) /
    (sizeof(typename filter_type::word_type) * filter_type::words_per_block);

  thrust::device_vector<Key> keys(num_keys);
  thrust::device_vector<bool> result(num_keys, false);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  state.add_element_count(num_keys);
  state.add_global_memory_reads<typename filter_type::word_type>(num_keys *
                                                                 filter_type::words_per_block);

  filter_type filter{num_sub_filters, {}, {static_cast<uint32_t>(pattern_bits)}};

  add_fpr_summary(state, filter);

  filter.add(keys.begin(), keys.end());

  state.exec([&](nvbench::launch& launch) {
    filter.contains_async(keys.begin(), keys.end(), result.begin(), {launch.get_stream()});
  });
}

NVBENCH_BENCH_TYPES(bloom_filter_add,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<defaults::BF_HASH>,
                                      nvbench::type_list<defaults::BF_BLOCK>,
                                      nvbench::type_list<distribution::unique>))
  .set_name("bloom_filter_add_unique_size")
  .set_type_axes_names({"Key", "Hash", "Block", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_CACHE)
  .add_int64_axis("PatternBits", {defaults::BF_PATTERN_BITS});

NVBENCH_BENCH_TYPES(bloom_filter_add,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      defaults::HASH_RANGE,
                                      nvbench::type_list<defaults::BF_BLOCK>,

                                      nvbench::type_list<distribution::unique>))
  .set_name("bloom_filter_add_unique_hash")
  .set_type_axes_names({"Key", "Hash", "Block", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", {defaults::BF_SIZE_MB})
  .add_int64_axis("PatternBits", {defaults::BF_PATTERN_BITS});

NVBENCH_BENCH_TYPES(bloom_filter_add,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<defaults::BF_HASH>,
                                      nvbench::type_list<cuda::std::array<nvbench::uint32_t, 1>,
                                                         cuda::std::array<nvbench::uint32_t, 2>,
                                                         cuda::std::array<nvbench::uint32_t, 4>,
                                                         cuda::std::array<nvbench::uint32_t, 8>,
                                                         cuda::std::array<nvbench::uint64_t, 1>,
                                                         cuda::std::array<nvbench::uint64_t, 2>,
                                                         cuda::std::array<nvbench::uint64_t, 4>,
                                                         cuda::std::array<nvbench::uint64_t, 8>>,
                                      nvbench::type_list<distribution::unique>))
  .set_name("bloom_filter_add_unique_block_dim")
  .set_type_axes_names({"Key", "Hash", "Block", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", {defaults::BF_SIZE_MB})
  .add_int64_axis("PatternBits", {defaults::BF_PATTERN_BITS});

NVBENCH_BENCH_TYPES(bloom_filter_contains,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<defaults::BF_HASH>,
                                      nvbench::type_list<defaults::BF_BLOCK>,

                                      nvbench::type_list<distribution::unique>))
  .set_name("bloom_filter_contains_unique_size")
  .set_type_axes_names({"Key", "Hash", "Block", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_CACHE)
  .add_int64_axis("PatternBits", {defaults::BF_PATTERN_BITS});

NVBENCH_BENCH_TYPES(bloom_filter_contains,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      defaults::HASH_RANGE,
                                      nvbench::type_list<defaults::BF_BLOCK>,

                                      nvbench::type_list<distribution::unique>))
  .set_name("bloom_filter_contains_unique_hash")
  .set_type_axes_names({"Key", "Hash", "Block", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", {defaults::BF_SIZE_MB})
  .add_int64_axis("PatternBits", {defaults::BF_PATTERN_BITS});

NVBENCH_BENCH_TYPES(bloom_filter_contains,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<defaults::BF_HASH>,
                                      nvbench::type_list<cuda::std::array<nvbench::uint32_t, 1>,
                                                         cuda::std::array<nvbench::uint32_t, 2>,
                                                         cuda::std::array<nvbench::uint32_t, 4>,
                                                         cuda::std::array<nvbench::uint32_t, 8>,
                                                         cuda::std::array<nvbench::uint64_t, 1>,
                                                         cuda::std::array<nvbench::uint64_t, 2>,
                                                         cuda::std::array<nvbench::uint64_t, 4>,
                                                         cuda::std::array<nvbench::uint64_t, 8>>,
                                      nvbench::type_list<distribution::unique>))
  .set_name("bloom_filter_contains_unique_block_dim")
  .set_type_axes_names({"Key", "Hash", "Block", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", {defaults::BF_SIZE_MB})
  .add_int64_axis("PatternBits", {defaults::BF_PATTERN_BITS});

/*
// benchmark outer product of configuration space
NVBENCH_BENCH_TYPES(
  bloom_filter_add,
  NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                    nvbench::type_list<cuco::xxhash_32<char>, cuco::xxhash_64<char>>,
                    nvbench::type_list<cuda::std::array<nvbench::uint32_t, 1>,
                                       cuda::std::array<nvbench::uint32_t, 2>,
                                       cuda::std::array<nvbench::uint32_t, 4>,
                                       cuda::std::array<nvbench::uint32_t, 8>,
                                       cuda::std::array<nvbench::uint64_t, 1>,
                                       cuda::std::array<nvbench::uint64_t, 2>,
                                       cuda::std::array<nvbench::uint64_t, 4>,
                                       cuda::std::array<nvbench::uint64_t, 8>>,
                    nvbench::type_list<distribution::unique>))
  .set_name("bloom_filter_add_unique_product")
  .set_type_axes_names({"Key", "Hash", "Block", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_CACHE)
  .add_int64_axis("PatternBits", {1, 2, 4, 6, 8, 10});

NVBENCH_BENCH_TYPES(
  bloom_filter_contains,
  NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                    nvbench::type_list<cuco::xxhash_32<char>, cuco::xxhash_64<char>>,
                    nvbench::type_list<cuda::std::array<nvbench::uint32_t, 1>,
                                       cuda::std::array<nvbench::uint32_t, 2>,
                                       cuda::std::array<nvbench::uint32_t, 4>,
                                       cuda::std::array<nvbench::uint32_t, 8>,
                                       cuda::std::array<nvbench::uint64_t, 1>,
                                       cuda::std::array<nvbench::uint64_t, 2>,
                                       cuda::std::array<nvbench::uint64_t, 4>,
                                       cuda::std::array<nvbench::uint64_t, 8>>,
                    nvbench::type_list<distribution::unique>))
  .set_name("bloom_filter_contains_unique_product")
  .set_type_axes_names({"Key", "Hash", "Block", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_CACHE)
  .add_int64_axis("PatternBits", {1, 2, 4, 6, 8, 10});
*/
