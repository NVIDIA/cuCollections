/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "../defaults.hpp"

#include <benchmark_defaults.hpp>

#include <nvbench/nvbench.cuh>

#include <cuda/std/limits>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <warpcore/bloom_filter.cuh>

using namespace cuco::benchmark;  // defaults

/**
 * @brief A benchmark evaluating `warpcore::BloomFilter::retrieve` performance
 */
template <typename Key,
          typename Word,
          nvbench::int32_t BlockBits,
          nvbench::int32_t PatternBitsPerWord>
void warpcore_bloom_filter_contains(
  nvbench::state& state,
  nvbench::
    type_list<Key, Word, nvbench::enum_type<BlockBits>, nvbench::enum_type<PatternBitsPerWord>>)
{
  auto constexpr words_per_block = BlockBits / cuda::std::numeric_limits<Word>::digits;

  // Check for a valid configuration
  if constexpr ((not cuda::std::has_single_bit(static_cast<uint32_t>(BlockBits))) or
                (words_per_block == 0)) {
    state.skip("Invalid filter block size");
  } else if constexpr ((PatternBitsPerWord <= 0) or
                       (PatternBitsPerWord > cuda::std::numeric_limits<Word>::digits)) {
    state.skip("Invalid pattern bits per word");
  } else {
    using size_type             = std::uint32_t;
    auto constexpr pattern_bits = words_per_block * PatternBitsPerWord;
    using filter_type =
      warpcore::BloomFilter<Key, warpcore::defaults::hasher_t<Key>, Word, words_per_block>;

    auto const num_keys = state.get_int64("NumInputs");
    state.add_element_count(num_keys);

    auto const filter_size_mb = state.get_int64("FilterSizeMB");
    std::size_t num_bits      = filter_size_mb * 1024 * 1024 * 8;
    filter_type filter{num_bits, pattern_bits};

    thrust::counting_iterator<Key> key_it(0);

    // insert FPR-optimal number of keys
    auto const num_build_keys = (filter_size_mb * 1024 * 1024 * 8) / (2 * pattern_bits);
    {
      thrust::device_vector<Key> build_keys(num_build_keys);
      thrust::sequence(build_keys.begin(), build_keys.end(), 0);
      filter.insert(thrust::raw_pointer_cast(build_keys.data()), num_build_keys);
    }

    // FPR summary
    thrust::device_vector<bool> result(num_keys, false);
    {
      thrust::device_vector<Key> fpr_keys(num_keys - num_build_keys);
      thrust::sequence(fpr_keys.begin(), fpr_keys.end(), num_build_keys);

      filter.retrieve(thrust::raw_pointer_cast(fpr_keys.data()),
                      num_keys - num_build_keys,
                      thrust::raw_pointer_cast(result.data()));
      double const fp = thrust::count(thrust::device, result.begin(), result.end(), true);

      auto& summ_fpr = state.add_summary("FalsePositiveRate");
      summ_fpr.set_string("hint", "FPR");
      summ_fpr.set_string("short_name", "FPR");
      summ_fpr.set_string("description", "False-positive rate of the bloom filter.");
      summ_fpr.set_float64("value", fp / static_cast<double>(num_keys));

      auto& summ_k = state.add_summary("PatternBits");
      summ_k.set_string("hint", "K");
      summ_k.set_string("short_name", "K");
      summ_k.set_string("description", "Cardinality of a key's bit pattern.");
      summ_k.set_int64("value", pattern_bits);
    }

    thrust::device_vector<Key> keys(num_keys);
    thrust::sequence(keys.begin(), keys.end(), 0);

    state.exec([&](nvbench::launch& launch) {
      filter.retrieve(thrust::raw_pointer_cast(keys.data()),
                      num_keys,
                      thrust::raw_pointer_cast(result.data()),
                      {launch.get_stream()});
    });
  }
}

NVBENCH_BENCH_TYPES(
  warpcore_bloom_filter_contains,
  NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::uint64_t>,           ///< Key
                    nvbench::type_list<nvbench::uint64_t>,           ///< Word
                    nvbench::enum_type_list<32, 64, 128, 256, 512>,  ///< BlockBits
                    nvbench::enum_type_list<1, 16>                   ///< PatternBitsPerWord
                    ))
  .set_name("warpcore_bloom_filter_contains_unique_size_u64")
  .set_type_axes_names({"Key", "Word", "BlockBits", "PatternBitsPerWord"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("NumInputs", {defaults::BF_N})
  .add_int64_axis("FilterSizeMB", defaults::BF_SIZE_MB_RANGE_CACHE);
