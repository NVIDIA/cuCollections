/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/hash_functions.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cstdint>

namespace cuco::benchmark {

template <typename FilterType>
void add_fpr_summary(nvbench::state& state, FilterType& filter)
{
  filter.clear();

  auto const num_keys = state.get_int64("NumInputs");

  thrust::device_vector<typename FilterType::key_type> keys(num_keys * 2);
  thrust::sequence(thrust::device, keys.begin(), keys.end(), 1);
  thrust::device_vector<bool> result(num_keys, false);

  auto tp_begin = keys.begin();
  auto tp_end   = tp_begin + num_keys;
  auto tn_begin = tp_end;
  auto tn_end   = keys.end();
  filter.add(tp_begin, tp_end);
  filter.contains(tn_begin, tn_end, result.begin());

  float fp = thrust::count(thrust::device, result.begin(), result.end(), true);

  auto& summ = state.add_summary("FalsePositiveRate");
  summ.set_string("hint", "FPR");
  summ.set_string("short_name", "FPR");
  summ.set_string("description", "False-positive rate of the bloom filter.");
  summ.set_float64("value", fp / num_keys);

  filter.clear();
}

}  // namespace cuco::benchmark