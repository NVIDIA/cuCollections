/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cuco/detail/cache_residency_control.cuh>

#include <thrust/device_vector.h>

#include <iostream>

int main(void)
{
  int const num_keys   = 10'000'000;
  int const num_bits   = 300'000'000;  // 37 MB; fits in the L2 of an A100
  int const num_hashes = 2;            // sufficient for small filters

  // Spawn a 37MB filter and 2-bit patterns for each key.
  cuco::bloom_filter<int> filter{num_bits, num_hashes};

  // Create a CUDA stream in which this operation is performed.
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  thrust::device_vector<int> keys(num_keys);
  thrust::sequence(keys.begin(), keys.end(), 1);
  thrust::device_vector<bool> contains(num_keys);

  // Insert all keys and subsequently query them against the filter; measure runtime
  cudaEvent_t gmem_start, gmem_stop;
  cudaEventCreate(&gmem_start);
  cudaEventCreate(&gmem_stop);

  cudaEventRecord(gmem_start, stream);
  filter.insert(keys.begin(), keys.end(), stream);
  filter.contains(keys.begin(), keys.end(), contains.begin(), stream);
  cudaEventRecord(gmem_stop, stream);
  cudaStreamSynchronize(stream);

  float gmem_delta;
  cudaEventElapsedTime(&gmem_delta, gmem_start, gmem_stop);
  std::cout << "Insert+query filter in global memory: " << gmem_delta << "ms\n";

  // Re-initialize the filter, i.e., set all bits to zero
  filter.initialize(stream);
  cudaStreamSynchronize(stream);

  // Make the filter persistent in the GPU's L2 cache
  cuco::register_l2_persistence(
    stream, filter.get_slots(), filter.get_slots() + filter.get_num_slots());

  // Insert all keys and subsequently query them against the filter; measure runtime
  cudaEvent_t l2_start, l2_stop;
  cudaEventCreate(&l2_start);
  cudaEventCreate(&l2_stop);

  cudaEventRecord(l2_start, stream);
  filter.insert(keys.begin(), keys.end(), stream);
  filter.contains(keys.begin(), keys.end(), contains.begin(), stream);
  cudaEventRecord(l2_stop, stream);
  cudaStreamSynchronize(stream);

  float l2_delta;
  cudaEventElapsedTime(&l2_delta, l2_start, l2_stop);
  std::cout << "Insert+query filter in L2: " << l2_delta << "ms\n";

  // Flush the L2 so it can be used for other tasks
  cuco::unregister_l2_persistence(stream);

  return 0;
}
