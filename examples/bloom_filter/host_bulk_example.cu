/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/bloom_filter.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <iostream>

int main(void)
{
  int constexpr num_keys    = 10'000;          ///< Generate 10'000 keys
  int constexpr num_tp      = num_keys * 0.5;  ///< Insert the first half keys into the filter.
  int constexpr num_tn      = num_keys - num_tp;
  int constexpr sub_filters = 200;  ///< 200 sub-filters per bloom filter

  // key type for bloom filter
  using key_type = int;

  // Spawn a bloom filter with default policy and 200 sub-filters.
  cuco::bloom_filter<key_type> filter{sub_filters};

  std::cout << "Bulk insert into bloom filter with default fingerprint generation policy: "
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