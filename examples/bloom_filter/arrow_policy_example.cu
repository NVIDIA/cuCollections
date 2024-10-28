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

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <iostream>

/**
 * @file arrow_policy_example.cu
 * @brief Demonstrates usage of an arrow-compatible bloom filter
 *
 * In addition to the default policy aimed at achieving the speed of light 
 * performance on the device, `cuCollections` offers an `arrow_filter_policy`
 * that allows users to easily create a bloom filter that mimics the behavior
 * of the bloom filter defined in Apache Arrow:
 * https://github.com/apache/arrow/blob/be1dcdb96b030639c0b56955c4c62f9d6b03f473/cpp/src/parquet/bloom_filter.cc#L219-L230.
 *
 * @note This example is for demonstration purposes only. It is not intended to show the most
 * performant way to do the example algorithm.
 */
 
int main(void)
{
  int constexpr num_keys    = 10'000;          ///< Generate 10'000 keys
  int constexpr num_tp      = num_keys * 0.5;  ///< Insert the first half keys into the filter.
  int constexpr num_tn      = num_keys - num_tp;
  int constexpr sub_filters = 200;  ///< 200 sub-filters per bloom filter

  // key type for bloom filter
  using key_type = int;

  // We will use the Arrow filter policy for bloom filter fingerprint generation
  using policy_type = cuco::arrow_filter_policy<key_type>;
  // Bloom filter type with Arrow filter policy
  using filter_type =
    cuco::bloom_filter<key_type, cuco::extent<size_t>, cuda::thread_scope_device, policy_type>;

  // Spawn a bloom filter with arrow policy and 200 sub-filters.
  filter_type filter{sub_filters};

  std::cout << "Bulk insert into bloom filter with Arrow fingerprint generation policy: "
            << std::endl;

  thrust::device_vector<key_type> keys(num_keys);
  thrust::sequence(keys.begin(), keys.end(), 1);

  auto tp_begin = keys.begin();
  auto tp_end   = tp_begin + num_tp;
  auto tn_begin = tp_end;
  auto tn_end   = keys.end();

  // Insert the first half of the keys.
  filter.add(tp_begin, tp_end);

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
