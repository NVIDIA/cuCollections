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
#include <cuco/detail/cache_residency_control.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/std/atomic>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cstddef>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

static constexpr nvbench::int64_t block_size = 256;
static constexpr nvbench::int64_t stride     = 4;

enum class FilterOperation { INSERT, CONTAINS };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  FilterOperation,
  [](FilterOperation op) {
    switch (op) {
      case FilterOperation::INSERT: return "INSERT";
      case FilterOperation::CONTAINS: return "CONTAINS";
      default: return "ERROR";
    }
  },
  [](FilterOperation op) {
    switch (op) {
      case FilterOperation::INSERT: return "FilterOperation::INSERT";
      case FilterOperation::CONTAINS: return "FilterOperation::CONTAINS";
      default: return "ERROR";
    }
  })

enum class FilterScope { GMEM, L2 };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  FilterScope,
  [](FilterScope s) {
    switch (s) {
      case FilterScope::GMEM: return "GMEM";
      case FilterScope::L2: return "L2";
      default: return "ERROR";
    }
  },
  [](FilterScope s) {
    switch (s) {
      case FilterScope::GMEM: return "FilterScope::GMEM";
      case FilterScope::L2: return "FilterScope::L2";
      default: return "ERROR";
    }
  })

enum class DataScope { GMEM, REGS };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  DataScope,
  [](DataScope s) {
    switch (s) {
      case DataScope::GMEM: return "GMEM";
      case DataScope::REGS: return "REGS";
      default: return "ERROR";
    }
  },
  [](DataScope s) {
    switch (s) {
      case DataScope::GMEM: return "DataScope::GMEM";
      case DataScope::REGS: return "DataScope::REGS";
      default: return "ERROR";
    }
  })

template <typename Key, typename Slot>
void add_size_summary(nvbench::state& state)
{
  using filter_type =
    cuco::bloom_filter<Key, cuda::thread_scope_device, cuco::cuda_allocator<char>, Slot>;

  auto const num_keys   = state.get_int64("NumInputs");
  auto const num_bits   = state.get_int64("NumBits");
  auto const num_hashes = state.get_int64("NumHashes");

  filter_type filter(num_bits, num_hashes);

  auto& summ = state.add_summary("nv/filter/size/mb");
  summ.set_string("hint", "FilterMB");
  summ.set_string("short_name", "FilterMB");
  summ.set_string("description", "Size of the Bloom filter in MB.");
  summ.set_float64("value", filter.get_num_slots() * sizeof(Slot) / 1000 / 1000);
}

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

  auto& summ = state.add_summary("nv/filter/fpr");
  summ.set_string("hint", "FPR");
  summ.set_string("short_name", "FPR");
  summ.set_string("description", "False-positive rate of the bloom filter.");
  summ.set_float64("value", fp / num_keys);
}

template <nvbench::int64_t BLOCK_SIZE, typename Filter, typename InputIt>
__global__ void __launch_bounds__(BLOCK_SIZE)
  insert_kernel(Filter mutable_view, InputIt first, InputIt last)
{
  std::size_t tid = block_size * blockIdx.x + threadIdx.x;
  auto it         = first + tid;

  while (it < last) {
    mutable_view.insert(*it);
    it += gridDim.x * BLOCK_SIZE;
  }
}

template <nvbench::int64_t BLOCK_SIZE, typename Filter, typename InputIt, typename OutputIt>
__global__ void __launch_bounds__(BLOCK_SIZE)
  contains_kernel(Filter view, InputIt first, InputIt last, OutputIt results)
{
  std::size_t tid = block_size * blockIdx.x + threadIdx.x;

  while ((first + tid) < last) {
    *(results + tid) = view.contains(*(first + tid));
    tid += gridDim.x * BLOCK_SIZE;
  }
}

template <nvbench::int64_t BLOCK_SIZE, typename Filter>
__global__ void __launch_bounds__(BLOCK_SIZE)
  insert_kernel(Filter mutable_view, nvbench::int64_t num_keys)
{
  using key_type = typename Filter::key_type;

  auto g = cg::this_grid();

  for (key_type key = g.thread_rank(); key < num_keys; key += g.size()) {
    mutable_view.insert(key);
  }
}

template <nvbench::int64_t BLOCK_SIZE, typename Filter>
__global__ void __launch_bounds__(BLOCK_SIZE)
  contains_kernel(Filter view, nvbench::int64_t num_keys)
{
  using key_type = typename Filter::key_type;

  auto g = cg::this_grid();

  for (key_type key = g.thread_rank(); key < num_keys; key += g.size()) {
    volatile bool contains = view.contains(key);
  }
}

template <typename Key, typename Slot, FilterOperation Op, FilterScope FScope, DataScope DScope>
void nvbench_cuco_bloom_filter(nvbench::state& state,
                               nvbench::type_list<Key,
                                                  Slot,
                                                  nvbench::enum_type<Op>,
                                                  nvbench::enum_type<FScope>,
                                                  nvbench::enum_type<DScope>>)
{
  auto num_keys   = state.get_int64("NumInputs");
  auto num_bits   = state.get_int64("NumBits");
  auto num_hashes = state.get_int64("NumHashes");

  [[maybe_unused]] thrust::device_vector<Key> keys;
  [[maybe_unused]] thrust::device_vector<bool> results;

  if constexpr (DScope == DataScope::GMEM) {
    keys.resize(num_keys);
    thrust::sequence(thrust::device, keys.begin(), keys.end(), 1);

    if constexpr (Op == FilterOperation::CONTAINS) { results.resize(num_keys); }
  }

  using filter_type =
    cuco::bloom_filter<Key, cuda::thread_scope_device, cuco::cuda_allocator<char>, Slot>;

  filter_type filter(num_bits, num_hashes);
  auto mutable_view           = filter.get_device_mutable_view();
  auto view                   = filter.get_device_view();
  std::size_t const grid_size = SDIV(num_keys, stride * block_size);

  state.add_element_count(num_keys);
  state.add_global_memory_writes<Slot>(num_keys);

  add_fpr_summary<Key, Slot>(state);
  add_size_summary<Key, Slot>(state);

  if constexpr (Op == FilterOperation::CONTAINS) {
    insert_kernel<block_size><<<grid_size, block_size>>>(mutable_view, num_keys);
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  if constexpr (FScope == FilterScope::L2)
    cuco::register_l2_persistence(
      stream, filter.get_slots(), filter.get_slots() + filter.get_num_slots());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

  state.exec([&](nvbench::launch& launch) {
    if constexpr (Op == FilterOperation::INSERT) {
      filter.initialize(launch.get_stream());
      if constexpr (DScope == DataScope::GMEM) {
        insert_kernel<block_size><<<grid_size, block_size, 0, launch.get_stream()>>>(
          mutable_view, keys.begin(), keys.end());
      }
      if constexpr (DScope == DataScope::REGS) {
        insert_kernel<block_size>
          <<<grid_size, block_size, 0, launch.get_stream()>>>(mutable_view, num_keys);
      }
    }
    if constexpr (Op == FilterOperation::CONTAINS) {
      if constexpr (DScope == DataScope::GMEM) {
        contains_kernel<block_size><<<grid_size, block_size, 0, launch.get_stream()>>>(
          view, keys.begin(), keys.end(), results.begin());
      }
      if constexpr (DScope == DataScope::REGS) {
        contains_kernel<block_size>
          <<<grid_size, block_size, 0, launch.get_stream()>>>(view, num_keys);
      }
    }
  });

  if constexpr (FScope == FilterScope::L2) cuco::unregister_l2_persistence(stream);
}

using key_type_range  = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using slot_type_range = nvbench::type_list<nvbench::int32_t, nvbench::uint64_t>;
using op_range        = nvbench::enum_type_list<FilterOperation::INSERT, FilterOperation::CONTAINS>;
using filter_scope_range = nvbench::enum_type_list<FilterScope::GMEM, FilterScope::L2>;
using data_scope_range   = nvbench::enum_type_list<DataScope::GMEM, DataScope::REGS>;

// A100 L2 = 40MB ~ 330'000'000 bits
// smem = 48kb ~ 390'0000 bits
// 1GB ~ 8'500'000'000 bits
// 4GB ~ 34'000'000'000 bits

NVBENCH_BENCH_TYPES(nvbench_cuco_bloom_filter,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      op_range,
                                      filter_scope_range,
                                      data_scope_range))
  .set_name("cuco_bloom_filter_l2")
  .set_type_axes_names({"KeyType", "SlotType", "FilterOperation", "FilterScope", "DataScope"})
  .set_max_noise(3)
  .add_int64_axis("NumInputs", {10'000'000, 100'000'000})
  .add_int64_axis("NumBits", {300'000'000})
  .add_int64_axis("NumHashes", {2});

NVBENCH_BENCH_TYPES(nvbench_cuco_bloom_filter,
                    NVBENCH_TYPE_AXES(key_type_range,
                                      slot_type_range,
                                      op_range,
                                      nvbench::enum_type_list<FilterScope::GMEM>,
                                      data_scope_range))
  .set_name("cuco_bloom_filter_gmem")
  .set_type_axes_names({"KeyType", "SlotType", "FilterOperation", "FilterScope", "DataScope"})
  .set_max_noise(3)
  .add_int64_axis("NumInputs", {1'000'000'000, 100'000'000})
  .add_int64_axis("NumBits", {8'500'000'000, 34'000'000'000})
  .add_int64_axis("NumHashes", {6});