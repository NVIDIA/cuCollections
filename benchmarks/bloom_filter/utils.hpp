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

#pragma once

#include <nvbench/nvbench.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

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