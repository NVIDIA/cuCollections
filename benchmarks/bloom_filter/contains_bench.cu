/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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
#include "nvbench/state.cuh"
#include "utils.hpp"

#include <benchmark_defaults.hpp>
#include <benchmark_utils.hpp>

#include <cuco/bloom_filter.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/iterator>
#include <cuda/std/limits>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

// REMOVE sort keys by hash
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <exception>
#include <limits>

using namespace cuco::benchmark;  // defaults, dist_from_state, rebind_hasher_t, add_fpr_summary
using namespace cuco::utility;    // key_generator, distribution

/**
 * @brief Implementation of `cuco::bloom_filter::contains_async` with
 * `parametric_filter_policy`
 */
template <bool ExcludeIO,
          typename Key,
          typename Word,
          nvbench::int32_t BlockBits,
          nvbench::int32_t PatternBits,
          nvbench::int32_t HorizontalLayout,
          nvbench::int32_t VerticalLayout>
void pfp_bloom_filter_contains_impl(nvbench::state& state,
                                    nvbench::type_list<Key,
                                                       Word,
                                                       nvbench::enum_type<BlockBits>,
                                                       nvbench::enum_type<PatternBits>,
                                                       nvbench::enum_type<HorizontalLayout>,
                                                       nvbench::enum_type<VerticalLayout>>)
{
  auto constexpr words_per_block       = BlockBits / cuda::std::numeric_limits<Word>::digits;
  auto constexpr pattern_bits_per_word = PatternBits / words_per_block;

  // Check for a valid configuration
  if constexpr ((not cuda::std::has_single_bit(static_cast<uint32_t>(BlockBits))) or
                (words_per_block == 0)) {
    state.skip("Invalid filter block size");
  } else if constexpr (HorizontalLayout * VerticalLayout > words_per_block) {
    state.skip("Invalid vectorization layout");  // TODO check if this is correct
  } else if constexpr ((pattern_bits_per_word <= 0) or
                       (pattern_bits_per_word > cuda::std::numeric_limits<Word>::digits) or
                       (pattern_bits_per_word * words_per_block > 64)) {
    state.skip("Invalid pattern bits per word");
  } else {
    using size_type                      = std::uint32_t;
    using hasher                         = cuco::xxhash_64<Key>;
    auto constexpr add_vertical_layout   = 1;
    auto constexpr add_horizontal_layout = words_per_block;
    using policy_type                    = cuco::parametric_filter_policy<hasher,
                                                                          Word,
                                                                          words_per_block,
                                                                          PatternBits,
                                                                          add_horizontal_layout,
                                                                          add_vertical_layout,
                                                                          HorizontalLayout,
                                                                          VerticalLayout>;
    using filter_type =
      cuco::bloom_filter<Key, cuco::extent<size_type>, cuda::thread_scope_device, policy_type>;

    auto const num_keys       = state.get_int64("NumInputs");
    auto const filter_size_mb = state.get_int64("FilterSizeMB");

    std::size_t const num_sub_filters =
      (filter_size_mb * 1024 * 1024) /
      (sizeof(typename filter_type::word_type) * filter_type::words_per_block);

    if (num_sub_filters > policy_type::max_filter_blocks) {
      // skip invalid configurations
      state.skip("num_sub_filters exceeds max_filter_blocks");
    }

    state.add_element_count(num_keys);

    filter_type filter{static_cast<size_type>(num_sub_filters)};

    thrust::counting_iterator<Key> key_it(0);

    // insert FPR-optimal number of keys
    auto const num_build_keys = (filter_size_mb * 1024 * 1024 * 8) / (2 * PatternBits);
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

    if constexpr (ExcludeIO) {
      state.add_global_memory_reads<Word>(num_keys * words_per_block);
      // state.collect_dram_throughput();

      auto result_it = make_lazy_discard_iterator(result.begin());

      state.exec([&](nvbench::launch& launch) {
        filter.contains_async(key_it, key_it + num_keys, result_it, {launch.get_stream()});
      });
    } else {
      thrust::device_vector<Key> keys(num_keys);
      thrust::sequence(thrust::device, keys.begin(), keys.end(), 0);

      // sort keys by hash
      // thrust::device_vector<uint64_t> hashes(num_keys);
      // thrust::transform(thrust::device, keys.begin(), keys.end(), hashes.begin(), hasher());
      // thrust::sort_by_key(thrust::device, hashes.begin(), hashes.end(), keys.begin());

      state.add_global_memory_reads<char>(num_keys *
                                          ((words_per_block * sizeof(Word)) + sizeof(Key)));
      state.add_global_memory_writes<char>(num_keys * sizeof(bool));
      // state.collect_dram_throughput();

      state.exec([&](nvbench::launch& launch) {
        filter.contains_async(keys.begin(), keys.end(), result.begin(), {launch.get_stream()});
      });
    }
  }
}

/**
 * @brief A benchmark evaluating `cuco::bloom_filter::contains_async` performance with
 * `parametric_filter_policy` with IO
 */
template <typename Key,
          typename Word,
          nvbench::int32_t BlockBits,
          nvbench::int32_t PatternBitsPerWord,
          nvbench::int32_t HorizontalLayout,
          nvbench::int32_t VerticalLayout>
void pfp_bloom_filter_contains(nvbench::state& state,
                               nvbench::type_list<Key,
                                                  Word,
                                                  nvbench::enum_type<BlockBits>,
                                                  nvbench::enum_type<PatternBitsPerWord>,
                                                  nvbench::enum_type<HorizontalLayout>,
                                                  nvbench::enum_type<VerticalLayout>> type_list)
{
  constexpr bool exclude_io = false;
  pfp_bloom_filter_contains_impl<exclude_io>(state, type_list);
}

/**
 * @brief A benchmark evaluating `cuco::bloom_filter::contains_async` performance with
 * `parametric_filter_policy` without IO
 */
template <typename Key,
          typename Word,
          nvbench::int32_t BlockBits,
          nvbench::int32_t PatternBitsPerWord,
          nvbench::int32_t HorizontalLayout,
          nvbench::int32_t VerticalLayout>
void pfp_bloom_filter_contains_exclude_io(
  nvbench::state& state,
  nvbench::type_list<Key,
                     Word,
                     nvbench::enum_type<BlockBits>,
                     nvbench::enum_type<PatternBitsPerWord>,
                     nvbench::enum_type<HorizontalLayout>,
                     nvbench::enum_type<VerticalLayout>> type_list)
{
  constexpr bool exclude_io = true;
  pfp_bloom_filter_contains_impl<exclude_io>(state, type_list);
}

/**
 * @brief A benchmark evaluating `cuco::bloom_filter::contains_async` performance with
 * `parametric_filter_policy` with cache sectorization
 */
template <typename Key,
          typename Word,
          nvbench::int32_t BlockBits,
          nvbench::int32_t PatternBits,
          nvbench::int32_t GroupsPerBlock,
          nvbench::int32_t HorizontalLayout>
void pfp_bloom_filter_contains_csbf(nvbench::state& state,
                                    nvbench::type_list<Key,
                                                       Word,
                                                       nvbench::enum_type<BlockBits>,
                                                       nvbench::enum_type<PatternBits>,
                                                       nvbench::enum_type<GroupsPerBlock>,
                                                       nvbench::enum_type<HorizontalLayout>>)
{
  auto constexpr words_per_block = BlockBits / cuda::std::numeric_limits<Word>::digits;
  auto constexpr words_per_group = words_per_block / GroupsPerBlock;
  auto constexpr VerticalLayout  = words_per_block / HorizontalLayout;

  if constexpr (words_per_group == 0) {
    state.skip("Invalid GroupsPerBlock");
  } else if constexpr ((HorizontalLayout * VerticalLayout != words_per_block) or
                       (VerticalLayout < words_per_group)) {
    state.skip("Invalid vectorization layout");
  } else {
    using size_type                      = std::uint32_t;
    using hasher                         = cuco::xxhash_64<Key>;
    auto constexpr add_horizontal_layout = GroupsPerBlock;
    auto constexpr add_vertical_layout   = words_per_group;

    using policy_type = cuco::parametric_filter_policy<hasher,
                                                       Word,
                                                       words_per_block,
                                                       PatternBits,
                                                       add_horizontal_layout,
                                                       add_vertical_layout,
                                                       HorizontalLayout,
                                                       VerticalLayout,
                                                       GroupsPerBlock>;
    using filter_type =
      cuco::bloom_filter<Key, cuco::extent<size_type>, cuda::thread_scope_device, policy_type>;

    auto const num_keys       = state.get_int64("NumInputs");
    auto const filter_size_mb = state.get_int64("FilterSizeMB");

    std::size_t const num_sub_filters =
      (filter_size_mb * 1024 * 1024) /
      (sizeof(typename filter_type::word_type) * filter_type::words_per_block);

    if (num_sub_filters > policy_type::max_filter_blocks) {
      // skip invalid configurations
      state.skip("num_sub_filters exceeds max_filter_blocks");
    }

    state.add_element_count(num_keys);

    filter_type filter{static_cast<size_type>(num_sub_filters)};

    thrust::counting_iterator<Key> key_it(0);

    // insert FPR-optimal number of keys
    auto const num_build_keys = (filter_size_mb * 1024 * 1024 * 8) / (2 * PatternBits);
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

    state.add_global_memory_reads<char>(num_keys *
                                        ((words_per_block * sizeof(Word)) + sizeof(Key)));
    state.add_global_memory_writes<char>(num_keys * sizeof(bool));
    // state.collect_dram_throughput();

    state.exec([&](nvbench::launch& launch) {
      filter.contains_async(keys.begin(), keys.end(), result.begin(), {launch.get_stream()});
    });
  }
}

NVBENCH_BENCH_TYPES(
  pfp_bloom_filter_contains,
  NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
                    nvbench::type_list<nvbench::uint64_t>,             ///< Word
                    nvbench::enum_type_list<64, 128, 256, 512, 1024>,  ///< BlockBits
                    nvbench::enum_type_list<16>,                       ///< PatternBits
                    nvbench::enum_type_list<1, 2, 4, 8, 16>,           /// <HorizontalLayout
                    nvbench::enum_type_list<1, 2, 4, 8, 16>            ///< VerticalLayout
                    ))
  .set_name("pfp_bloom_filter_contains_unique_size_u64")
  .set_type_axes_names(
    {"Key", "Word", "BlockBits", "PatternBits", "HorizontalLayout", "VerticalLayout"})
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_FRONTIER_CACHE);

NVBENCH_BENCH_TYPES(pfp_bloom_filter_contains_csbf,
                    NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
                                      nvbench::type_list<nvbench::uint64_t>,
                                      nvbench::enum_type_list<128, 256, 512, 1024>,
                                      nvbench::enum_type_list<16>,
                                      nvbench::enum_type_list<2, 4, 8>,
                                      nvbench::enum_type_list<1, 2, 4, 8>))
  .set_name("pfp_bloom_filter_contains_csbf_unique_size_u64")
  .set_type_axes_names(
    {"Key", "Word", "BlockBits", "PatternBits", "GroupsPerBlock", "HorizontalLayout"})
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_FRONTIER_CACHE);