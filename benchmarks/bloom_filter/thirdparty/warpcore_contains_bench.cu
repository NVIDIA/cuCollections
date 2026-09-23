/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../defaults.hpp"

#include <nvbench/nvbench.cuh>

#include <cuda/std/limits>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <warpcore/bloom_filter.cuh>

#include <cstdint>

using namespace cuco::benchmark;  // defaults

/**
 * @brief A benchmark evaluating `warpcore::BloomFilter::retrieve` performance
 */
template <typename Key, typename Word, nvbench::int32_t BlockBits, nvbench::int32_t PatternBits>
void warpcore_bloom_filter_contains(
  nvbench::state& state,
  nvbench::type_list<Key, Word, nvbench::enum_type<BlockBits>, nvbench::enum_type<PatternBits>>)
{
  auto constexpr words_per_block       = BlockBits / cuda::std::numeric_limits<Word>::digits;
  auto constexpr pattern_bits_per_word = PatternBits / words_per_block;

  if constexpr ((not cuda::std::has_single_bit(static_cast<std::uint32_t>(BlockBits))) or
                (words_per_block == 0)) {
    state.skip("Invalid filter block size");
  } else if constexpr ((pattern_bits_per_word <= 0) or
                       (pattern_bits_per_word > cuda::std::numeric_limits<Word>::digits)) {
    state.skip("Invalid pattern bits per word");
  } else {
    using filter_type =
      warpcore::BloomFilter<Key, warpcore::defaults::hasher_t<Key>, Word, words_per_block>;

    auto const num_keys = state.get_int64("NumInputs");
    state.add_element_count(num_keys);

    auto const filter_size_mb = state.get_int64("FilterSizeMB");
    std::size_t num_bits      = filter_size_mb * 1024 * 1024 * 8;

    {
      filter_type filter{num_bits, PatternBits};
      auto const num_build_keys = num_bits / (2 * PatternBits);
      thrust::device_vector<Key> build_keys(num_build_keys);
      thrust::sequence(thrust::device, build_keys.begin(), build_keys.end(), 0);
      filter.insert(thrust::raw_pointer_cast(build_keys.data()), num_build_keys);

      thrust::device_vector<bool> result(num_keys, false);
      thrust::device_vector<Key> fpr_keys(num_keys);
      thrust::sequence(thrust::device, fpr_keys.begin(), fpr_keys.end(), num_build_keys);
      filter.retrieve(thrust::raw_pointer_cast(fpr_keys.data()),
                      num_keys,
                      thrust::raw_pointer_cast(result.data()));

      double const false_positives =
        thrust::count(thrust::device, result.begin(), result.end(), true);
      auto& summary = state.add_summary("FalsePositiveRate");
      summary.set_string("hint", "FPR");
      summary.set_string("short_name", "FPR");
      summary.set_string("description", "False-positive rate of the Bloom filter.");
      summary.set_float64("value", false_positives / static_cast<double>(num_keys));
    }

    filter_type filter{num_bits, PatternBits};
    thrust::device_vector<Key> keys(num_keys);
    thrust::sequence(thrust::device, keys.begin(), keys.end(), 0);
    filter.insert(thrust::raw_pointer_cast(keys.data()), num_keys);
    thrust::device_vector<bool> result(num_keys, false);

    state.exec([&](nvbench::launch& launch) {
      filter.retrieve(thrust::raw_pointer_cast(keys.data()),
                      num_keys,
                      thrust::raw_pointer_cast(result.data()),
                      launch.get_stream());
    });
  }
}

NVBENCH_BENCH_TYPES(
  warpcore_bloom_filter_contains,
  NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::uint64_t>,             ///< Key
                    nvbench::type_list<nvbench::uint64_t>,             ///< Word
                    nvbench::enum_type_list<64, 128, 256, 512, 1024>,  ///< BlockBits
                    nvbench::enum_type_list<16>                        ///< PatternBits
                    ))
  .set_name("warpcore_bloom_filter_contains_unique_size_u64")
  .set_type_axes_names({"Key", "Word", "BlockBits", "PatternBits"})
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", {32, 1024});
