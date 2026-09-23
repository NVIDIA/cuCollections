/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "defaults.hpp"

#include <benchmark_defaults.hpp>

#include <cuco/bloom_filter.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/std/limits>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <cstdint>

using namespace cuco::benchmark;  // defaults

template <typename Key,
          nvbench::int32_t WordBytes,
          nvbench::int32_t BlockBits,
          nvbench::int32_t PatternBits,
          nvbench::int32_t GroupsPerBlock,
          nvbench::int32_t HorizontalLayout>
void bloom_filter_contains_csbf(nvbench::state& state,
                                nvbench::type_list<Key,
                                                   nvbench::enum_type<WordBytes>,
                                                   nvbench::enum_type<BlockBits>,
                                                   nvbench::enum_type<PatternBits>,
                                                   nvbench::enum_type<GroupsPerBlock>,
                                                   nvbench::enum_type<HorizontalLayout>>)
{
  auto constexpr word_bits       = WordBytes * cuda::std::numeric_limits<unsigned char>::digits;
  auto constexpr words_per_block = BlockBits / word_bits;
  auto constexpr words_per_group = words_per_block / (GroupsPerBlock == 0 ? 1 : GroupsPerBlock);
  auto constexpr vertical_layout = words_per_block / (HorizontalLayout == 0 ? 1 : HorizontalLayout);

  if constexpr ((not cuda::std::has_single_bit(static_cast<uint32_t>(BlockBits))) ||
                words_per_block == 0) {
    state.skip("Invalid filter block size");
  } else if constexpr (GroupsPerBlock == 0 || GroupsPerBlock > words_per_block ||
                       words_per_block % GroupsPerBlock != 0) {
    state.skip("Invalid cache-sectorization layout");
  } else if constexpr (HorizontalLayout == 0 ||
                       HorizontalLayout * vertical_layout != words_per_block ||
                       vertical_layout % words_per_group != 0) {
    state.skip("Invalid vectorization layout");
  } else if constexpr (PatternBits % GroupsPerBlock != 0 ||
                       PatternBits / GroupsPerBlock > word_bits) {
    state.skip("Invalid pattern bits per group");
  } else {
    using size_type                      = std::uint32_t;
    auto constexpr add_horizontal_layout = GroupsPerBlock;
    auto constexpr add_vertical_layout   = words_per_group;
    using policy_type                    = cuco::bloom_filter_policy<Key,
                                                                     cuco::xxhash_64<Key>,
                                                                     WordBytes,
                                                                     words_per_block,
                                                                     PatternBits,
                                                                     add_horizontal_layout,
                                                                     add_vertical_layout,
                                                                     HorizontalLayout,
                                                                     vertical_layout,
                                                                     false,
                                                                     false,
                                                                     false,
                                                                     GroupsPerBlock>;
    using filter_type =
      cuco::bloom_filter<Key, cuco::extent<size_type>, cuda::thread_scope_device, policy_type>;

    auto const num_keys       = state.get_int64("NumInputs");
    auto const filter_size_mb = state.get_int64("FilterSizeMB");
    std::size_t const num_blocks =
      (filter_size_mb * 1024 * 1024) /
      (sizeof(typename filter_type::word_type) * filter_type::words_per_block);

    if (num_blocks > policy_type::max_filter_blocks) {
      state.skip("num_blocks exceeds max_filter_blocks");
      return;
    }

    state.add_element_count(num_keys);

    filter_type filter{static_cast<size_type>(num_blocks)};
    thrust::counting_iterator<Key> key_it(0);

    auto const num_build_keys = (filter_size_mb * 1024 * 1024 * 8) / (2 * PatternBits);
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

    thrust::device_vector<Key> keys(num_keys);
    thrust::sequence(thrust::device, keys.begin(), keys.end(), 0);

    state.add_global_memory_reads<char>(num_keys * ((words_per_block * WordBytes) + sizeof(Key)));
    state.add_global_memory_writes<char>(num_keys * sizeof(bool));

    state.exec([&](nvbench::launch& launch) {
      filter.contains_async(keys.begin(), keys.end(), result.begin(), {launch.get_stream()});
    });
  }
}

NVBENCH_BENCH_TYPES(bloom_filter_contains_csbf,
                    NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
                                      nvbench::enum_type_list<8>,                    ///< WordBytes
                                      nvbench::enum_type_list<128, 256, 512, 1024>,  ///< BlockBits
                                      nvbench::enum_type_list<16>,             ///< PatternBits
                                      nvbench::enum_type_list<2, 4, 8, 16>,    ///< GroupsPerBlock
                                      nvbench::enum_type_list<1, 2, 4, 8, 16>  ///< HorizontalLayout
                                      ))
  .set_name("bloom_filter_contains_csbf_unique_size_u64")
  .set_type_axes_names(
    {"Key", "WordBytes", "BlockBits", "PatternBits", "GroupsPerBlock", "HorizontalLayout"})
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_CACHE);
