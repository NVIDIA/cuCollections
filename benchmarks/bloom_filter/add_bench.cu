/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "defaults.hpp"

#include <benchmark_defaults.hpp>
#include <benchmark_utils.hpp>

#include <cuco/bloom_filter.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/std/limits>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cstdint>

using namespace cuco::benchmark;  // defaults, dist_from_state, rebind_hasher_t
using namespace cuco::utility;    // key_generator, distribution

/**
 * @brief A benchmark evaluating `cuco::bloom_filter::add_async` performance
 */
template <typename Key,
          nvbench::int32_t WordBytes,
          nvbench::int32_t BlockBits,
          nvbench::int32_t PatternBits,
          nvbench::int32_t HorizontalLayout,
          nvbench::int32_t VerticalLayout>
void bloom_filter_add(nvbench::state& state,
                      nvbench::type_list<Key,
                                         nvbench::enum_type<WordBytes>,
                                         nvbench::enum_type<BlockBits>,
                                         nvbench::enum_type<PatternBits>,
                                         nvbench::enum_type<HorizontalLayout>,
                                         nvbench::enum_type<VerticalLayout>>)
{
  auto constexpr word_bits       = WordBytes * cuda::std::numeric_limits<unsigned char>::digits;
  auto constexpr words_per_block = BlockBits / word_bits;
  auto constexpr pattern_bits_per_word = PatternBits / words_per_block;

  // Check for a valid configuration
  if constexpr ((not cuda::std::has_single_bit(static_cast<uint32_t>(BlockBits))) or
                (words_per_block == 0)) {
    state.skip("Invalid filter block size");
  } else if constexpr (HorizontalLayout * VerticalLayout != words_per_block) {
    state.skip("Invalid vectorization layout");
  } else if constexpr ((pattern_bits_per_word <= 0) or (pattern_bits_per_word > word_bits) or
                       (pattern_bits_per_word * words_per_block > 64)) {
    state.skip("Invalid pattern bits per word");
  } else {
    using size_type                           = std::uint32_t;
    auto constexpr contains_vertical_layout   = words_per_block;
    auto constexpr contains_horizontal_layout = 1;
    using policy_type                         = cuco::bloom_filter_policy<Key,
                                                                          cuco::xxhash_64<Key>,
                                                                          WordBytes,
                                                                          words_per_block,
                                                                          PatternBits,
                                                                          HorizontalLayout,
                                                                          VerticalLayout,
                                                                          contains_horizontal_layout,
                                                                          contains_vertical_layout,
                                                                          false,
                                                                          false,
                                                                          false>;
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

// Default benchmark: single layout matching default `cuco::bloom_filter_policy`.
NVBENCH_BENCH_TYPES(bloom_filter_add,
                    NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
                                      nvbench::enum_type_list<4>,    ///< WordBytes
                                      nvbench::enum_type_list<256>,  ///< BlockBits
                                      nvbench::enum_type_list<8>,    ///< PatternBits
                                      nvbench::enum_type_list<8>,    ///< HorizontalLayout
                                      nvbench::enum_type_list<1>     ///< VerticalLayout
                                      ))
  .set_name("bloom_filter_add_unique_size")
  .set_type_axes_names(
    {"Key", "WordBytes", "BlockBits", "PatternBits", "HorizontalLayout", "VerticalLayout"})
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_CACHE);

// Exhaustive sweep across block sizes and vectorization layouts. Uncomment for performance
// tuning / paper-style characterization; not run by default because the matrix is large.
// NVBENCH_BENCH_TYPES(
//   bloom_filter_add,
//   NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
//                     nvbench::enum_type_list<8, 4>,                    ///< WordBytes
//                     nvbench::enum_type_list<64, 128, 256, 512, 1024>, ///< BlockBits
//                     nvbench::enum_type_list<8, 16>,                   ///< PatternBits
//                     nvbench::enum_type_list<1, 2, 4, 8, 16>,          ///< HorizontalLayout
//                     nvbench::enum_type_list<1, 2, 4, 8, 16>           ///< VerticalLayout
//                     ))
//   .set_name("bloom_filter_add_full_sweep_u64")
//   .set_type_axes_names(
//     {"Key", "WordBytes", "BlockBits", "PatternBits", "HorizontalLayout", "VerticalLayout"})
//   .add_int64_axis("NumInputs", {defaults::BF_N})
//   .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_CACHE);
