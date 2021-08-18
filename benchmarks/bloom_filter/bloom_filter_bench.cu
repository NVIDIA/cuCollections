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

#include <cuda_runtime_api.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <cuco/bloom_filter.cuh>
#include <nvbench/nvbench.cuh>

enum class filter_op { INSERT, CONTAINS };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  // Enum type:
  filter_op,
  // Callable to generate input strings:
  // Short identifier used for tables, command-line args, etc.
  // Used when context is available to figure out the enum type.
  [](filter_op op) {
    switch (op) {
      case filter_op::INSERT: return "INSERT";
      case filter_op::CONTAINS: return "CONTAINS";
      default: return "ERROR";
    }
  },
  // Callable to generate descriptions:
  // If non-empty, these are used in `--list` to describe values.
  // Used when context may not be available to figure out the type from the
  // input string.
  // Just use `[](auto) { return std::string{}; }` if you don't want these.
  [](auto) { return std::string{}; })

enum class filter_scope { GMEM, L2 };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  // Enum type:
  filter_scope,
  // Callable to generate input strings:
  // Short identifier used for tables, command-line args, etc.
  // Used when context is available to figure out the enum type.
  [](filter_scope s) {
    switch (s) {
      case filter_scope::GMEM: return "GMEM";
      case filter_scope::L2: return "L2";
      default: return "ERROR";
    }
  },
  // Callable to generate descriptions:
  // If non-empty, these are used in `--list` to describe values.
  // Used when context may not be available to figure out the type from the
  // input string.
  // Just use `[](auto) { return std::string{}; }` if you don't want these.
  [](auto) { return std::string{}; })

template <typename Key, typename Slot, filter_op Op, filter_scope Scope>
void nvbench_cuco_bloom_filter(
  nvbench::state& state,
  nvbench::type_list<Key, Slot, nvbench::enum_type<Op>, nvbench::enum_type<Scope>>)
{
  using filter_type =
    cuco::bloom_filter<Key, cuda::thread_scope_device, cuco::cuda_allocator<char>, Slot>;

  auto const num_keys   = state.get_int64("NumInputs");
  auto const num_bits   = state.get_int64("NumBits");
  auto const num_hashes = state.get_int64("NumHashes");

  thrust::device_vector<Key> keys(num_keys * 2);
  thrust::sequence(keys.begin(), keys.end(), 1);
  thrust::device_vector<bool> result(num_keys, false);

  auto tp_begin = keys.begin();
  auto tp_end   = tp_begin + num_keys;

  {
    filter_type filter(num_bits, num_hashes);

    // check if the filter fits into L2
    if constexpr (Scope == filter_scope::L2) {
      int device_id;
      cudaGetDevice(&device_id);
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, device_id);

      auto const filter_bytes = filter.get_num_bits() / CHAR_BIT;

      if (filter_bytes > prop.persistingL2CacheMaxSize) {
        state.skip("Filter does not fit into L2.");
        return;
      } else {
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, filter_bytes);
      }
    }

    // determine false-positive rate
    auto tn_begin = tp_end;
    auto tn_end   = keys.end();

    filter.insert(tp_begin, tp_end);
    filter.contains(tn_begin, tn_end, result.begin());

    float fp = thrust::count(thrust::device, result.begin(), result.end(), true);

    auto& summ = state.add_summary("False-Positive Rate");
    summ.set_string("hint", "FPR");
    summ.set_string("short_name", "FPR");
    summ.set_string("description", "False-positive rate of the bloom filter.");
    summ.set_float64("value", float(fp) / num_keys);
  }

  state.add_element_count(num_keys);

  if constexpr (Scope == filter_scope::GMEM) {
    state.exec(
      nvbench::exec_tag::sync | nvbench::exec_tag::timer,
      [&](nvbench::launch& launch, auto& timer) {
        filter_type filter(num_bits, num_hashes);

        if constexpr (Op == filter_op::INSERT) {
          timer.start();
          filter.insert(tp_begin, tp_end, launch.get_stream());
          timer.stop();
        } else if constexpr (Op == filter_op::CONTAINS) {
          filter.insert(tp_begin, tp_end, launch.get_stream());
          thrust::fill(
            thrust::cuda::par.on(launch.get_stream()), result.begin(), result.end(), false);

          timer.start();
          filter.contains(tp_begin, tp_end, result.begin(), launch.get_stream());
          timer.stop();
        } else {
          state.skip("Invaild operation type.");
        }
      });
  } else if constexpr (Scope == filter_scope::L2) {
    state.exec(
      nvbench::exec_tag::sync | nvbench::exec_tag::timer,
      [&](nvbench::launch& launch, auto& timer) {
        filter_type filter(num_bits, num_hashes);

        cudaStreamAttrValue stream_attribute;
        stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(filter.get_slots());
        stream_attribute.accessPolicyWindow.num_bytes = filter.get_num_bits() / CHAR_BIT;
        stream_attribute.accessPolicyWindow.hitRatio  = 0.6;  // TODO find proper value
        stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(
          launch.get_stream(), cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

        filter.initialize(launch.get_stream());

        if constexpr (Op == filter_op::INSERT) {
          timer.start();
          filter.insert(tp_begin, tp_end, launch.get_stream());
          timer.stop();
        } else if constexpr (Op == filter_op::CONTAINS) {
          filter.insert(tp_begin, tp_end, launch.get_stream());
          thrust::fill(
            thrust::cuda::par.on(launch.get_stream()), result.begin(), result.end(), false);

          timer.start();
          filter.contains(tp_begin, tp_end, result.begin(), launch.get_stream());
          timer.stop();
        } else {
          state.skip("Invaild operation type.");
        }

        stream_attribute.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(
          launch.get_stream(), cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

        cudaCtxResetPersistingL2Cache();
      });
  } else {
    state.skip("Invalid scope.");
  }
}

// type parameter dimensions for benchmark
using key_type_range  = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using slot_type_range = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using op_range        = nvbench::enum_type_list<filter_op::INSERT, filter_op::CONTAINS>;
using scope_range     = nvbench::enum_type_list<filter_scope::L2, filter_scope::GMEM>;

// benchmark setups
NVBENCH_BENCH_TYPES(nvbench_cuco_bloom_filter,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t, nvbench::int64_t>,
                                      op_range,
                                      scope_range))
  .set_name("cuco_bloom_filter_scope")
  .set_type_axes_names({"KeyType", "SlotType", "Operation", "Scope"})
  .set_max_noise(3)  // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis(
    "NumInputs",
    {10'000'000, 20'000'000, 30'000'000, 40'000'000, 50'000'000})  // Total number of keys
  .add_int64_axis(
    "NumBits",
    {200'000'000, 250'000'000, 300'000'000, 350'000'000, 400'000'000})  //, 100'000'000'000})
  .add_int64_axis("NumHashes", nvbench::range(2, 8, 2));

NVBENCH_BENCH_TYPES(nvbench_cuco_bloom_filter,
                    NVBENCH_TYPE_AXES(key_type_range,
                                      slot_type_range,
                                      op_range,
                                      nvbench::enum_type_list<filter_scope::GMEM>))
  .set_name("cuco_bloom_filter_gmem")
  .set_type_axes_names({"KeyType", "SlotType", "Operation", "Scope"})
  .set_max_noise(3)                                           // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000, 1'000'000'000})  // Total number of keys
  .add_int64_axis("NumBits",
                  {10'000'000,
                   100'000'000,
                   1'000'000'000,
                   10'000'000'000,
                   20'000'000'000,
                   30'000'000'000,
                   40'000'000'000,
                   50'000'000'000,
                   60'000'000'000,
                   70'000'000'000,
                   80'000'000'000,
                   90'000'000'000,
                   100'000'000'000})
  .add_int64_axis("NumHashes", nvbench::range(2, 8, 2));