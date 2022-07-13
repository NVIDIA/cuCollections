/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cuda_runtime_api.h>
#include <nvbench/nvbench.cuh>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

enum class filter_op { INSERT, CONTAINS };
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  filter_op,
  [](filter_op op) {
    switch (op) {
      case filter_op::INSERT: return "INSERT";
      case filter_op::CONTAINS: return "CONTAINS";
      default: return "ERROR";
    }
  },
  [](auto) { return std::string{}; })

enum class filter_scope { GMEM, L2 };
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  filter_scope,
  [](filter_scope s) {
    switch (s) {
      case filter_scope::GMEM: return "GMEM";
      case filter_scope::L2: return "L2";
      default: return "ERROR";
    }
  },
  [](auto) { return std::string{}; })

enum class data_scope { GMEM, REGISTERS };
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  data_scope,
  [](data_scope s) {
    switch (s) {
      case data_scope::GMEM: return "GMEM";
      case data_scope::REGISTERS: return "REGISTERS";
      default: return "ERROR";
    }
  },
  [](auto) { return std::string{}; })

template <typename Key, typename Slot>
void add_fpr_summary(nvbench::state& state)
{
  using filter_type =
    cuco::bloom_filter<Key, cuda::thread_scope_device, cuco::cuda_allocator<char>, Slot>;

  auto const num_keys   = state.get_int64("NumInputs");
  auto const num_bits   = state.get_int64("NumBits");
  auto const num_hashes = state.get_int64("NumHashes");

  thrust::device_vector<Key> keys(num_keys * 2);
  thrust::sequence(thrust::device, keys.begin(), keys.end(), 1);
  thrust::device_vector<bool> result(num_keys, false);

  auto tp_begin = keys.begin();
  auto tp_end   = tp_begin + num_keys;
  auto tn_begin = tp_end;
  auto tn_end   = keys.end();

  filter_type filter(num_bits, num_hashes);
  filter.insert(tp_begin, tp_end);
  filter.contains(tn_begin, tn_end, result.begin());

  float fp = thrust::count(thrust::device, result.begin(), result.end(), true);

  auto& summ = state.add_summary("False-Positive Rate");
  summ.set_string("hint", "FPR");
  summ.set_string("short_name", "FPR");
  summ.set_string("description", "False-positive rate of the bloom filter.");
  summ.set_float64("value", fp / num_keys);
}

template <typename MutableFilterView>
__global__ void insert_from_regs_to_gmem_kernel(MutableFilterView mutable_view,
                                                std::size_t num_keys)
{
  using key_type = typename MutableFilterView::key_type;

  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_keys;
       i += blockDim.x * gridDim.x) {
    mutable_view.insert(key_type(i + 1));
  }
}

template <typename Key, typename Slot, filter_op Op, filter_scope FilterScope, data_scope DataScope>
void nvbench_cuco_bloom_filter(nvbench::state& state,
                               nvbench::type_list<Key,
                                                  Slot,
                                                  nvbench::enum_type<Op>,
                                                  nvbench::enum_type<FilterScope>,
                                                  nvbench::enum_type<DataScope>>)
{
  // using thread_scope = std::conditional<(FilterScope == filter_scope::SMEM),
  // cuda::thread_scope_block, cuda::thread_scope_device>;
  using filter_type =
    cuco::bloom_filter<Key, cuda::thread_scope_device, cuco::cuda_allocator<char>, Slot>;
  using mutable_view_type = typename filter_type::device_mutable_view;
  using view_type         = typename filter_type::device_view;

  std::size_t constexpr block_size = 128;

  auto const num_keys   = state.get_int64("NumInputs");
  auto const num_bits   = state.get_int64("NumBits");
  auto const num_hashes = state.get_int64("NumHashes");

  state.add_element_count(num_keys);

  add_fpr_summary<Key, Slot>(state);

  [[maybe_unused]] nvbench::float64_t l2_hit_rate;
  [[maybe_unused]] nvbench::float64_t l2_carve_out;
  [[maybe_unused]] thrust::device_vector<Key> keys;
  [[maybe_unused]] thrust::device_vector<bool> result;

  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);

  if constexpr (DataScope == data_scope::GMEM) {
    keys.resize(num_keys);
    thrust::sequence(thrust::device, keys.begin(), keys.end(), 1);

    if constexpr (Op == filter_op::CONTAINS) { result.resize(num_keys); }
  }

  if constexpr (FilterScope == filter_scope::L2) {
    filter_type filter(num_bits, num_hashes);
    auto const filter_bytes = filter.get_num_bits() / CHAR_BIT;

    if (filter_bytes > prop.accessPolicyMaxWindowSize) {
      state.skip("Filter size exceeds maximum access policy window size.");
      return;
    }

    l2_hit_rate  = state.get_float64("L2HitRate");
    l2_carve_out = state.get_float64("L2CarveOut");

    if (l2_hit_rate <= 0.0 or l2_hit_rate > 1.0) {
      state.skip("L2 hit ratio must be in (0,1].");
      return;
    }

    if (l2_carve_out <= 0.0 or l2_carve_out > 1.0) {
      state.skip("L2 carve out must be in (0,1].");
      return;
    }

    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                       l2_carve_out * prop.persistingL2CacheMaxSize);
  }

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      filter_type filter(num_bits, num_hashes);

      if constexpr (FilterScope == filter_scope::L2) {
        // Stream level attributes data structure
        cudaStreamAttrValue stream_attribute;
        // Global Memory data pointer
        stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(filter.get_slots());
        // Number of bytes for persistence access.
        stream_attribute.accessPolicyWindow.num_bytes = filter.get_num_bits() / CHAR_BIT;
        // Hint for cache hit ratio
        stream_attribute.accessPolicyWindow.hitRatio = l2_hit_rate;
        // Type of access property on cache hit
        stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        // Type of access property on cache miss.
        stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        // Set the attributes to a CUDA stream of type cudaStream_t
        cudaStreamSetAttribute(
          launch.get_stream(), cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
      }

      if constexpr (Op == filter_op::INSERT) {
        if constexpr (DataScope == data_scope::GMEM) {
          timer.start();
          filter.insert(keys.begin(), keys.end(), launch.get_stream());
          timer.stop();
        } else if constexpr (DataScope == data_scope::REGISTERS) {
          auto mutable_view = filter.get_device_mutable_view();

          std::size_t const grid_size = cuco::detail::get_grid_size(
            insert_from_regs_to_gmem_kernel<mutable_view_type>, block_size);

          timer.start();
          insert_from_regs_to_gmem_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
            mutable_view, num_keys);
          timer.stop();
        } else {
          state.skip("Invalid data scope.");
          return;
        }
      } else if constexpr (Op == filter_op::CONTAINS) {
        if constexpr (DataScope == data_scope::GMEM) {
          filter.insert(keys.begin(), keys.end(), launch.get_stream());
          thrust::fill(
            thrust::cuda::par.on(launch.get_stream()), result.begin(), result.end(), false);
          timer.start();
          filter.contains(keys.begin(), keys.end(), result.begin(), launch.get_stream());
          timer.stop();
        } else {
          state.skip("Invalid data scope.");
          return;
        }
      } else {
        state.skip("Invalid filter operation.");
        return;
      }

      if constexpr (FilterScope == filter_scope::L2) {
        cudaStreamAttrValue stream_attribute;
        // Setting the window size to 0 disable it
        stream_attribute.accessPolicyWindow.num_bytes = 0;
        // Overwrite the access policy attribute to a CUDA Stream
        cudaStreamSetAttribute(
          launch.get_stream(), cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
        // Remove any persistent lines in L2
        cudaCtxResetPersistingL2Cache();
      }
    });
}

// type parameter dimensions for benchmark
using key_type_range     = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using slot_type_range    = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using op_range           = nvbench::enum_type_list<filter_op::INSERT, filter_op::CONTAINS>;
using filter_scope_range = nvbench::enum_type_list<filter_scope::L2, filter_scope::GMEM>;
using data_scope_range   = nvbench::enum_type_list<data_scope::GMEM, data_scope::REGISTERS>;

NVBENCH_BENCH_TYPES(nvbench_cuco_bloom_filter,
                    NVBENCH_TYPE_AXES(key_type_range,
                                      nvbench::type_list<nvbench::int32_t>,
                                      op_range,
                                      nvbench::enum_type_list<filter_scope::GMEM>,
                                      data_scope_range))
  .set_name("cuco_bloom_filter_gmem")
  .set_type_axes_names({"KeyType", "SlotType", "Operation", "FilterScope", "DataScope"})
  .set_max_noise(3)                                        // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000, 10'000'000})  // Total number of keys
  .add_int64_axis("NumBits",
                  {100'000'000,
                   200'000'000,
                   250'000'000,
                   300'000'000,
                   350'000'000,
                   400'000'000,
                   450'000'000,
                   500'000'000,
                   1'000'000'000,
                   5'000'000'000,
                   10'000'000'000,
                   50'000'000'000})  //, 100'000'000'000})
  .add_int64_axis("NumHashes", {2});

NVBENCH_BENCH_TYPES(nvbench_cuco_bloom_filter,
                    NVBENCH_TYPE_AXES(key_type_range,
                                      nvbench::type_list<nvbench::int32_t>,
                                      op_range,
                                      nvbench::enum_type_list<filter_scope::L2>,
                                      data_scope_range))
  .set_name("cuco_bloom_filter_L2")
  .set_type_axes_names({"KeyType", "SlotType", "Operation", "FilterScope", "DataScope"})
  .set_max_noise(3)                                        // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000, 10'000'000})  // Total number of keys
  .add_int64_axis("NumBits",
                  {100'000'000,
                   200'000'000,
                   250'000'000,
                   300'000'000,
                   350'000'000,
                   400'000'000,
                   450'000'000,
                   500'000'000,
                   1'000'000'000,
                   5'000'000'000,
                   10'000'000'000,
                   50'000'000'000})
  .add_int64_axis("NumHashes", {2})
  .add_float64_axis("L2HitRate", nvbench::range(0.2, 1.0, 0.2))
  .add_float64_axis("L2CarveOut", nvbench::range(0.2, 1.0, 0.2));