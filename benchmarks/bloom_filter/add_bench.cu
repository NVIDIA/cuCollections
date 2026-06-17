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
#include "utils.hpp"

#include <benchmark_defaults.hpp>
#include <benchmark_utils.hpp>

#include <cuco/bloom_filter.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/iterator>
#include <cuda/std/limits>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cstdint>
#include <exception>
#include <limits>

using namespace cuco::benchmark;  // defaults, dist_from_state, rebind_hasher_t, add_fpr_summary
using namespace cuco::utility;    // key_generator, distribution

/**
 * @brief Implementation of `cuco::bloom_filter::add_async`
 */
template <bool ExcludeIO,
          typename Key,
          typename Word,
          nvbench::int32_t BlockBits,
          nvbench::int32_t PatternBits,
          nvbench::int32_t HorizontalLayout,
          nvbench::int32_t VerticalLayout>
void bloom_filter_add_impl(nvbench::state& state,
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
  } else if constexpr (HorizontalLayout * VerticalLayout != words_per_block) {
    state.skip("Invalid vectorization layout");
  } else if constexpr ((pattern_bits_per_word <= 0) or
                       (pattern_bits_per_word > cuda::std::numeric_limits<Word>::digits) or
                       (pattern_bits_per_word * words_per_block > 64)) {
    state.skip("Invalid pattern bits per word");
  } else {
    using size_type                           = std::uint32_t;
    using hasher                              = cuco::xxhash_64<Key>;
    auto constexpr contains_vertical_layout   = words_per_block;
    auto constexpr contains_horizontal_layout = 1;
    using policy_type                         = cuco::parametric_filter_policy<hasher,
                                                                               Word,
                                                                               words_per_block,
                                                                               PatternBits,
                                                                               HorizontalLayout,
                                                                               VerticalLayout,
                                                                               contains_horizontal_layout,
                                                                               contains_vertical_layout>;
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

    if constexpr (ExcludeIO) {
      state.add_global_memory_writes<Word>(num_keys * words_per_block);

      cuda::counting_iterator<Key> keys(0);

      state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        timer.start();
        filter.add_async(keys, keys + num_keys, {launch.get_stream()});
        timer.stop();
        filter.clear_async({launch.get_stream()});
      });
    } else {
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
}

/**
 * @brief A benchmark evaluating `cuco::bloom_filter::add_async` performance with IO
 */
template <typename Key,
          typename Word,
          nvbench::int32_t BlockBits,
          nvbench::int32_t PatternBits,
          nvbench::int32_t HorizontalLayout,
          nvbench::int32_t VerticalLayout>
void bloom_filter_add(nvbench::state& state,
                      nvbench::type_list<Key,
                                         Word,
                                         nvbench::enum_type<BlockBits>,
                                         nvbench::enum_type<PatternBits>,
                                         nvbench::enum_type<HorizontalLayout>,
                                         nvbench::enum_type<VerticalLayout>> type_list)
{
  constexpr bool exclude_io = false;
  bloom_filter_add_impl<exclude_io>(state, type_list);
}

/**
 * @brief A benchmark evaluating `cuco::bloom_filter::add_async` performance without IO
 */
template <typename Key,
          typename Word,
          nvbench::int32_t BlockBits,
          nvbench::int32_t PatternBits,
          nvbench::int32_t HorizontalLayout,
          nvbench::int32_t VerticalLayout>
void bloom_filter_add_exclude_io(nvbench::state& state,
                                 nvbench::type_list<Key,
                                                    Word,
                                                    nvbench::enum_type<BlockBits>,
                                                    nvbench::enum_type<PatternBits>,
                                                    nvbench::enum_type<HorizontalLayout>,
                                                    nvbench::enum_type<VerticalLayout>> type_list)
{
  constexpr bool exclude_io = true;
  bloom_filter_add_impl<exclude_io>(state, type_list);
}

NVBENCH_BENCH_TYPES(
  bloom_filter_add,
  NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
                    nvbench::type_list<nvbench::uint64_t>,             ///< Word
                    nvbench::enum_type_list<64, 128, 256, 512, 1024>,  ///< BlockBits
                    nvbench::enum_type_list<16>,                       ///< PatternBits
                    nvbench::enum_type_list<1, 2, 4, 8, 16>,           ///< HorizontalLayout
                    nvbench::enum_type_list<1, 2, 4>                   ///< VerticalLayout
                    ))
  .set_name("bloom_filter_add_unique_size_u64")
  .set_type_axes_names(
    {"Key", "Word", "BlockBits", "PatternBits", "HorizontalLayout", "VerticalLayout"})
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_FRONTIER_CACHE);

NVBENCH_BENCH_TYPES(
  bloom_filter_add_exclude_io,
  NVBENCH_TYPE_AXES(nvbench::type_list<defaults::BF_KEY>,
                    nvbench::type_list<nvbench::uint64_t>,             ///< Word
                    nvbench::enum_type_list<64, 128, 256, 512, 1024>,  ///< BlockBits
                    nvbench::enum_type_list<16>,                       ///< PatternBits
                    nvbench::enum_type_list<1, 2, 4, 8, 16>,           ///< HorizontalLayout
                    nvbench::enum_type_list<1, 2, 4>                   ///< VerticalLayout
                    ))
  .set_name("bloom_filter_add_exclude_io_unique_size_u64")
  .set_type_axes_names(
    {"Key", "Word", "BlockBits", "PatternBits", "HorizontalLayout", "VerticalLayout"})
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_FRONTIER_CACHE);