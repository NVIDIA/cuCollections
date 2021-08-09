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

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <cuco/bloom_filter.cuh>
#include <nvbench/nvbench.cuh>

/**
 * @brief A benchmark evaluating insert performance.
 */
template <typename Key, typename Slot>
void nvbench_cuco_bloom_filter_insert(nvbench::state& state, nvbench::type_list<Key, Slot>)
{
  using filter_type =
    cuco::bloom_filter<Key, cuda::thread_scope_device, cuco::cuda_allocator<char>, Slot>;

  auto const num_keys   = state.get_int64("NumInputs");
  auto const num_bits   = state.get_int64("NumBits");
  auto const num_hashes = state.get_int64("NumHashes");

  thrust::device_vector<Key> keys(num_keys * 2);
  thrust::sequence(keys.begin(), keys.end(), 1);

  auto tp_begin = keys.begin();
  auto tp_end   = tp_begin + num_keys;

  {  // determine false-positive rate
    auto tn_begin = tp_end;
    auto tn_end   = keys.end();

    filter_type filter(num_bits, num_hashes);
    filter.insert(tp_begin, tp_end);

    thrust::device_vector<bool> result(num_keys, false);
    filter.contains(tn_begin, tn_end, result.begin());

    float fp = thrust::count(thrust::device, result.begin(), result.end(), true);

    auto& summ = state.add_summary("False-Positive Rate");
    summ.set_string("hint", "FPR");
    summ.set_string("short_name", "FPR");
    summ.set_string("description", "False-positive rate of the bloom filter.");
    summ.set_float64("value", float(fp) / num_keys);
  }

  state.add_element_count(num_keys);
  state.add_global_memory_writes<Key>(num_keys);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               filter_type filter(num_bits, num_hashes);

               timer.start();
               filter.insert(tp_begin, tp_end, launch.get_stream());
               timer.stop();
             });
}

/**
 * @brief A benchmark evaluating insert performance.
 */
template <typename Key, typename Slot>
void nvbench_cuco_bloom_filter_contains(nvbench::state& state, nvbench::type_list<Key, Slot>)
{
  using filter_type =
    cuco::bloom_filter<Key, cuda::thread_scope_device, cuco::cuda_allocator<char>, Slot>;

  auto const num_keys   = state.get_int64("NumInputs");
  auto const num_bits   = state.get_int64("NumBits");
  auto const num_hashes = state.get_int64("NumHashes");

  thrust::device_vector<Key> keys(num_keys * 2);
  thrust::sequence(thrust::device, keys.begin(), keys.end(), 1);

  auto tp_begin = keys.begin();
  auto tp_end   = tp_begin + (num_keys);

  auto tn_begin = tp_end;
  auto tn_end   = keys.end();

  filter_type filter(num_bits, num_hashes);
  filter.insert(tp_begin, tp_end);

  // determine false-positive rate
  thrust::device_vector<bool> result(num_keys, false);
  filter.contains(tn_begin, tn_end, result.begin());

  float fp = thrust::count(thrust::device, result.begin(), result.end(), true);

  auto& summ = state.add_summary("False-Positive Rate");
  summ.set_string("hint", "FPR");
  summ.set_string("short_name", "FPR");
  summ.set_string("description", "False-positive rate of the bloom filter.");
  summ.set_float64("value", float(fp) / num_keys);

  state.add_element_count(num_keys);
  state.add_global_memory_reads<Key>(num_keys);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               filter.contains(tp_begin, tp_end, result.begin(), launch.get_stream());
               timer.stop();
             });
}

// type parameter dimensions for benchmark
using key_type_range  = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using slot_type_range = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;

// benchmark setups

NVBENCH_BENCH_TYPES(nvbench_cuco_bloom_filter_insert,
                    NVBENCH_TYPE_AXES(key_type_range, slot_type_range))
  .set_name("nvbench_cuco_bloom_filter_insert")
  .set_type_axes_names({"Key", "Slot"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of keys
  .add_int64_axis("NumBits", {10'000'000'000, 100'000'000'000})  //, 100'000'000'000})
  .add_int64_axis("NumHashes", nvbench::range(2, 10, 2));

NVBENCH_BENCH_TYPES(nvbench_cuco_bloom_filter_contains,
                    NVBENCH_TYPE_AXES(key_type_range, slot_type_range))
  .set_name("nvbench_cuco_bloom_filter_contains")
  .set_type_axes_names({"Key", "Slot"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of keys
  .add_int64_axis("NumBits", {10'000'000'000, 100'000'000'000})
  .add_int64_axis("NumHashes", nvbench::range(6, 12, 2));
