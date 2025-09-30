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

#define BITWISE_COMPARE

#include <cuco/bloom_filter.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/fast_int.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <iostream>

int main()
{
  using key_type = int;
  using hasher   = cuco::xxhash_64<key_type>;

  // Generate the data.
  size_t constexpr num_build_keys = 1'000'000;
  size_t constexpr num_probe_keys = 1'000'000'000;
  thrust::device_vector<key_type> build_keys(num_build_keys);
  thrust::device_vector<key_type> probe_keys(num_probe_keys);
  thrust::sequence(build_keys.begin(), build_keys.end(), 0, 2);  // even build keys
  thrust::sequence(probe_keys.begin(), probe_keys.end(), 1, 2);  // odd probe keys
  thrust::device_vector<bool> output_flags(num_probe_keys, false);

  //===----------Parametric Filter Policy----------===//
  /**
   * CURRENT CONFIGURATION:
   * - Block Size: 256b
   * - Sector Size: 32b
   * - k: 8
   * - Add Horizontal Layout: 8
   * - Add Vertical Layout: 1
   * - Add Loop Iterations: 1
   * - Contains Horizontal Layout: 1
   * - Contains Vertical Layout: 8
   * - Contains Loop Iterations: 1
   */
  // using word_type                               = uint32_t;
  // uint32_t constexpr words_per_block            = 8;
  // uint32_t constexpr pattern_bits               = 8;
  // uint32_t constexpr add_horizontal_layout      = 8;
  // uint32_t constexpr add_vertical_layout        = 1;
  // uint32_t constexpr contains_horizontal_layout = 1;
  // uint32_t constexpr contains_vertical_layout   = 8;
  // using policy_t = cuco::experimental::detail::parametric_filter_policy<hasher,
  //                                                                       word_type,
  //                                                                       words_per_block,
  //                                                                       pattern_bits,
  //                                                                       add_horizontal_layout,
  //                                                                       add_vertical_layout,
  //                                                                       contains_horizontal_layout,
  //                                                                       contains_vertical_layout>;

  using policy_t = cuco::experimental::arrow_filter_policy<key_type, cuco::xxhash_64>;

  using filter_t = cuco::
    bloom_filter<key_type, cuco::utility::fast_int<uint32_t>, cuda::thread_scope_device, policy_t>;

  // Create the filter.
  uint32_t constexpr bits_per_key = 2 * policy_t::pattern_bits;  // ~50% LF
  uint32_t constexpr num_blocks =
    cuda::ceil_div(num_build_keys * bits_per_key,
                   policy_t::words_per_block * sizeof(typename policy_t::word_type) * 8);
  filter_t filter(cuco::utility::fast_int<uint32_t>{num_blocks});
  std::cout << "Filter size (bytes): "
            << filter.block_extent().value() * policy_t::words_per_block *
                 sizeof(typename policy_t::word_type)
            << "\n";

  // Build
  filter.add(build_keys.begin(), build_keys.end());

  // Probe
  filter.contains(probe_keys.begin(), probe_keys.end(), output_flags.begin());

  // Calcuate the FPR
  auto const num_fps = thrust::count(output_flags.begin(), output_flags.end(), true);
  auto const fp_rate = float(num_fps) / float(num_probe_keys);
  std::cout << "Num FPs=" << num_fps << "\n";
  std::cout << "FPR=" << fp_rate << "\n";

  //===----------Arrow Filter Policy----------===//
  using arrow_filter_t = cuco::bloom_filter<key_type,
                                            cuco::extent<size_t>,
                                            cuda::thread_scope_device,
                                            cuco::arrow_filter_policy<key_type, cuco::xxhash_64>>;
  thrust::device_vector<bool> output_flags_arrow(num_probe_keys, false);

  // Create the Arrow filter.
  arrow_filter_t arrow_filter(num_blocks);
  std::cout << "Filter size (bytes): "
            << arrow_filter.block_extent() * arrow_filter_t::words_per_block *
                 sizeof(typename arrow_filter_t::word_type)
            << "\n";

  // Build
  arrow_filter.add(build_keys.begin(), build_keys.end());

  // Probe
  arrow_filter.contains(probe_keys.begin(), probe_keys.end(), output_flags_arrow.begin());

  // Calcuate the FPR
  auto const num_fps_arrow =
    thrust::count(output_flags_arrow.begin(), output_flags_arrow.end(), true);
  auto const fp_rate_arrow = float(num_fps_arrow) / float(num_probe_keys);
  std::cout << "Arrow Num FPs=" << num_fps_arrow << "\n";
  std::cout << "Arrow FPR=" << fp_rate_arrow << "\n";

#ifdef BITWISE_COMPARE
#include <thrust/mismatch.h>
  if (num_fps != num_fps_arrow) {
    std::cout << "Mismatch in number of false positives between policies!\n";
    return -1;
  }
  auto const mismatch_iter = thrust::mismatch(
    thrust::device,
    arrow_filter.data(),
    arrow_filter.data() + arrow_filter.block_extent() * arrow_filter_t::words_per_block,
    filter.data());
  if (mismatch_iter.first !=
      arrow_filter.data() + arrow_filter.block_extent() * arrow_filter_t::words_per_block) {
    auto const mismatch_index = thrust::distance(arrow_filter.data(), mismatch_iter.first);
    std::cout << "Mismatch at index: " << mismatch_index << "\n";
    return -1;
  }
  std::cout << "Output bitwise match between policies!\n";
#endif

  return 0;
}