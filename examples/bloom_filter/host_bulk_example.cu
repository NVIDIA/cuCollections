/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <iostream>

int main(void)
{
  // Generate 10'000 keys and insert the first 5'000 into the filter.
  int const num_keys = 10'000;
  int const num_tp   = num_keys * 0.5;
  int const num_tn   = num_keys - num_tp;

  // Spawn a filter with 1'000'000 bits and 6-bit patterns for each key.
  cuco::bloom_filter<int> filter{num_tp * 10, 6};

  thrust::device_vector<int> keys(num_keys);
  thrust::sequence(keys.begin(), keys.end(), 1);

  auto tp_begin = keys.begin();
  auto tp_end   = tp_begin + num_tp;
  auto tn_begin = tp_end;
  auto tn_end   = keys.end();

  // Insert the first half of the keys.
  filter.insert(tp_begin, tp_end);

  thrust::device_vector<bool> tp_result(num_tp, false);
  thrust::device_vector<bool> tn_result(num_keys - num_tp, false);

  // Query the filter for the previously inserted keys.
  // This should result in a true-positive rate of TPR=1.
  filter.contains(tp_begin, tp_end, tp_result.begin());

  // Query the filter for the keys that are not present in the filter.
  // Since bloom filters are probalistic data structures, the filter
  // exhibits a false-positive rate FPR>0 depending on the number of bits in
  // the filter and the number of hashes used per key.
  filter.contains(tn_begin, tn_end, tn_result.begin());

  float tp_rate =
    float(thrust::count(thrust::device, tp_result.begin(), tp_result.end(), true)) / float(num_tp);
  float fp_rate =
    float(thrust::count(thrust::device, tn_result.begin(), tn_result.end(), true)) / float(num_tn);

  std::cout << "TPR=" << tp_rate << " FPR=" << fp_rate << std::endl;

  return 0;
}
