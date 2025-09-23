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

#define DEBUG_FPS

#include <cuco/bloom_filter.cuh>
#include <cuco/hash_functions.cuh>

#include <cub/cub.cuh>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <iostream>

int main()
{
  using key_type = int;
  using hasher   = cuco::xxhash_64<key_type>;
  /**
   * CURRENT CONFIGURATION:
   * - Block Size: 256b
   * - Sector Size: 32b
   * - k: 8
   * - Add Horizontal Layout: 8
   * - Add Vertical Layout: 1
   * - Add Loop Count: 1
   * - Contains Horizontal Layout: 1
   * - Contains Vertical Layout: 8
   * - Contains Loop Count: 1
   */
  using word_type                               = uint32_t;
  uint32_t constexpr words_per_block            = 8;
  uint32_t constexpr pattern_bits               = 8;
  uint32_t constexpr add_horizontal_layout      = 8;
  uint32_t constexpr add_vertical_layout        = 1;
  uint32_t constexpr contains_horizontal_layout = 8;
  uint32_t constexpr contains_vertical_layout   = 1;
  using policy_t = cuco::experimental::detail::parametric_filter_policy<hasher,
                                                                        word_type,
                                                                        words_per_block,
                                                                        pattern_bits,
                                                                        add_horizontal_layout,
                                                                        add_vertical_layout,
                                                                        contains_horizontal_layout,
                                                                        contains_vertical_layout>;
  using filter_t =
    cuco::bloom_filter<key_type, cuco::extent<size_t>, cuda::thread_scope_device, policy_t>;

  // Create the filter.
  size_t constexpr num_build_keys = 100;
  size_t constexpr bits_per_key   = 2 * pattern_bits;
  size_t constexpr num_blocks =
    cuda::ceil_div(num_build_keys * bits_per_key, sizeof(word_type) * 8);
  filter_t filter(num_blocks);

  // Generate the data.
  size_t constexpr num_probe_keys = 2 * num_build_keys;
  thrust::device_vector<key_type> build_keys(num_build_keys);
  thrust::device_vector<key_type> probe_keys(num_probe_keys);
  thrust::sequence(build_keys.begin(), build_keys.end(), 0);
  thrust::sequence(probe_keys.begin(), probe_keys.end(), 0);
  thrust::device_vector<bool> tp_result(num_build_keys, false);  // ground truth positives
  thrust::device_vector<bool> tn_result(num_build_keys, false);  // ground truth negatives

  // Insert the build keys
  filter.add(build_keys.begin(), build_keys.end());
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "Add done.\n";

  // Probe the filter
  filter.contains(probe_keys.begin(), probe_keys.begin() + num_build_keys, tp_result.begin());
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  filter.contains(probe_keys.begin() + num_build_keys, probe_keys.end(), tn_result.begin());
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "Contains done.\n";

  /// DEBUG ///
  // Ensure no false negatives
  auto const num_fns = thrust::count(tp_result.begin(), tp_result.end(), false);
  if (num_fns != 0) {
    std::cout << "Error: False negatives detected: " << num_fns << "\n";
    // Show the false negatives
    for (size_t i = 0; i < num_build_keys; ++i) {
      if (!tp_result[i]) { std::cout << "False negative: " << probe_keys[i] << "\n"; }
    }
  } else {
    std::cout << "No false negatives detected.\n";
  }
  /// DEBUG ///

  // Calcuate the FPR
  auto const num_fps = thrust::count(tn_result.begin(), tn_result.end(), true);
#ifdef DEBUG_FPS
  for (size_t i = 0; i < num_build_keys; ++i) {
    if (tn_result[i]) { std::cout << "False positive: " << probe_keys[num_build_keys + i] << "\n"; }
  }
#endif
  auto const fp_rate = float(num_fps) / float(num_build_keys);
  std::cout << "FPR=" << fp_rate << "\n";

  return 0;
}