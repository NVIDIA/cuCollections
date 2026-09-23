/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "defaults.hpp"

#include <benchmark_defaults.hpp>

#include <cuco/bloom_filter.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/std/limits>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cstdint>

using namespace cuco::benchmark;  // defaults

template <typename Key,
          nvbench::int32_t WordBytes,
          nvbench::int32_t BlockBits,
          nvbench::int32_t PatternBits,
          nvbench::int32_t GroupsPerBlock,
          nvbench::int32_t HorizontalLayout>
void bloom_filter_add_csbf(nvbench::state& state,
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
    using size_type                           = std::uint32_t;
    auto constexpr contains_horizontal_layout = GroupsPerBlock;
    auto constexpr contains_vertical_layout   = words_per_group;
    using policy_type                         = cuco::bloom_filter_policy<Key,
                                                                          cuco::xxhash_64<Key>,
                                                                          WordBytes,
                                                                          words_per_block,
                                                                          PatternBits,
                                                                          HorizontalLayout,
                                                                          vertical_layout,
                                                                          contains_horizontal_layout,
                                                                          contains_vertical_layout,
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
    }

    state.add_element_count(num_keys);

    filter_type filter{static_cast<size_type>(num_blocks)};
    thrust::device_vector<Key> keys(num_keys);
    thrust::sequence(thrust::device, keys.begin(), keys.end(), 0);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      filter.add_async(keys.begin(), keys.end(), {launch.get_stream()});
      timer.stop();
      filter.clear_async({launch.get_stream()});
    });
  }
}

NVBENCH_BENCH_TYPES(bloom_filter_add_csbf,
                    NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
                                      nvbench::enum_type_list<8>,                    ///< WordBytes
                                      nvbench::enum_type_list<128, 256, 512, 1024>,  ///< BlockBits
                                      nvbench::enum_type_list<16>,             ///< PatternBits
                                      nvbench::enum_type_list<2, 4, 8, 16>,    ///< GroupsPerBlock
                                      nvbench::enum_type_list<1, 2, 4, 8, 16>  ///< HorizontalLayout
                                      ))
  .set_name("bloom_filter_add_csbf_unique_size_u64")
  .set_type_axes_names(
    {"Key", "WordBytes", "BlockBits", "PatternBits", "GroupsPerBlock", "HorizontalLayout"})
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_CACHE);
