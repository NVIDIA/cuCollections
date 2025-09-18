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

#include <cuco/bloom_filter.cuh>
#include <cuco/hash_functions.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <iostream>

// TODO: Hook up user interface with new kernels.

int main()
{
  using key_type                                = int;
  using hasher                                  = cuco::xxhash_64<key_type>;
  using word_type                               = uint64_t;
  uint32_t constexpr words_per_block            = 1;
  uint32_t constexpr pattern_bits               = 8;
  uint32_t constexpr add_horizontal_layout      = 1;
  uint32_t constexpr add_vertical_layout        = 1;
  uint32_t constexpr contains_horizontal_layout = 1;
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
  constexpr size_t num_keys     = 100;
  constexpr size_t bits_per_key = 2 * words_per_block;
  constexpr size_t num_blocks   = cuda::ceil_div(num_keys * bits_per_key, sizeof(word_type) * 8);
  filter_t filter(num_blocks);

  // Generate the data.
  thrust::device_vector<key_type> build_keys(num_keys);
  thrust::device_vector<key_type> probe_keys(num_keys);
  thrust::sequence(build_keys.begin(), build_keys.end(), 0, 2);  // even build keys
  thrust::sequence(probe_keys.begin(), probe_keys.end(), 1, 2);  // odd probe keys
  thrust::device_vector<bool> output_flags(num_keys, false);

  auto stream = cuda::stream_ref();  // default stream

  // Insert the build keys
  filter.add_async(build_keys.begin(), build_keys.end(), stream);

  // Probe the filter
  filter.contains_async(probe_keys.begin(), probe_keys.end(), output_flags.begin(), stream);
  stream.wait();

  // Calcuate the FPR
  float fp_rate =
    float(thrust::count(output_flags.begin(), output_flags.end(), true)) / float(num_keys);

  std::cout << "FPR=" << fp_rate << "\n";

  return 0;
}