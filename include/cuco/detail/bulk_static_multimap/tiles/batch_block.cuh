/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "shared.cuh"

#include <cstddef>

namespace cuco::detail::bulk_tiles {
namespace batch_detail {
template <typename Record, typename Offset, typename Counter, typename Bucket>
__host__ __device__ const TileBuildStorage<Record, Offset, Counter, Bucket>& table(
  TileBuildStorage<Record, Offset, Counter, Bucket> const& storage)
{
  return storage;
}
template <typename Record, typename Offset, typename Counter, typename Bucket>
__host__ __device__ const TileBuildStorage<Record, Offset, Counter, Bucket>& table(
  TileLrbBuildStorage<Record, Offset, Counter, Bucket> const& storage)
{
  return storage.table;
}

template <typename T>
__host__ __device__ T* advance(T* base, int64_t index, int64_t stride)
{
  return base ? base + index * stride : nullptr;
}

template <typename T>
__host__ bool slab_fits(int64_t count, int64_t stride)
{
  if (count < 0 || stride < 0) return false;
  constexpr uint64_t limit = uint64_t(std::numeric_limits<ptrdiff_t>::max()) / sizeof(T);
  return !count || uint64_t(stride) <= limit / uint64_t(count);
}

template <typename Record, typename Offset, typename Counter, typename Bucket>
__host__ bool uniform_fits(TileBuildStorage<Record, Offset, Counter, Bucket> const& storage,
                           int64_t count)
{
  return slab_fits<Record>(count, storage.record_count) &&
         slab_fits<Record>(count, storage.record_capacity) &&
         slab_fits<Offset>(count, storage.offset_capacity) &&
         slab_fits<Counter>(count, storage.counter_capacity);
}
template <typename Record, typename Offset, typename Counter, typename Bucket>
__host__ bool uniform_fits(TileLrbBuildStorage<Record, Offset, Counter, Bucket> const& storage,
                           int64_t count)
{
  return uniform_fits(storage.table, count) && slab_fits<Record>(count, storage.reorder_capacity) &&
         slab_fits<Bucket>(count, storage.hash_capacity) &&
         slab_fits<Counter>(count, storage.bin_counter_capacity) &&
         slab_fits<Offset>(count, storage.bin_offset_capacity);
}

template <typename Record, typename Offset, typename Counter, typename Bucket>
__host__ __device__ TileBuildStorage<Record, Offset, Counter, Bucket> uniform_at(
  TileBuildStorage<Record, Offset, Counter, Bucket> storage, int64_t index)
{
  storage.input    = advance(storage.input, index, storage.record_count);
  storage.records  = advance(storage.records, index, storage.record_capacity);
  storage.offsets  = advance(storage.offsets, index, storage.offset_capacity);
  storage.counters = advance(storage.counters, index, storage.counter_capacity);
  return storage;
}
template <typename Record, typename Offset, typename Counter, typename Bucket>
__host__ __device__ TileLrbBuildStorage<Record, Offset, Counter, Bucket> uniform_at(
  TileLrbBuildStorage<Record, Offset, Counter, Bucket> storage, int64_t index)
{
  storage.table         = uniform_at(storage.table, index);
  storage.reordered     = advance(storage.reordered, index, storage.reorder_capacity);
  storage.bucket_hashes = advance(storage.bucket_hashes, index, storage.hash_capacity);
  storage.bin_counters  = advance(storage.bin_counters, index, storage.bin_counter_capacity);
  storage.bin_offsets   = advance(storage.bin_offsets, index, storage.bin_offset_capacity);
  return storage;
}
}  // namespace batch_detail

// Descriptors and their pointed-to arrays must be device accessible and live through completion.
template <typename Storage>
struct DeviceTableBatch {
  using storage_type         = Storage;
  Storage const* descriptors = nullptr;
  int64_t table_count        = 0;
  __host__ bool valid() const
  {
    return table_count >= 0 && (!table_count || descriptors) &&
           batch_detail::slab_fits<Storage>(table_count, 1);
  }
  __device__ Storage operator[](int64_t index) const { return descriptors[index]; }
};

// Each array is a separate packed slab; capacities specify the per-table strides.
template <typename Storage>
struct UniformTableBatch {
  using storage_type = Storage;
  Storage first{};
  int64_t table_count = 0;
  __host__ bool valid() const
  {
    return table_count >= 0 && (!table_count || (first.validate() == TileBuildStatus::success &&
                                                 batch_detail::uniform_fits(first, table_count)));
  }
  __host__ __device__ Storage operator[](int64_t index) const
  {
    return batch_detail::uniform_at(first, index);
  }
};

enum class TileBatchMemory { global, shared_metadata, shared_output };
struct TileBatchOptions {
  TileBatchMemory memory = TileBatchMemory::global;
  size_t shared_bytes    = 0;
  // Zero uses CUDA's one-dimensional grid limit; smaller values permit bounded grids.
  int max_blocks = 0;
};

namespace batch_detail {
template <typename Batch, typename = void>
struct has_resolve : std::false_type {};
template <typename Batch>
struct has_resolve<Batch, std::void_t<decltype(std::declval<Batch>().resolve())>> : std::true_type {
};
template <typename Batch>
__device__ auto resolve(Batch batch)
{
  if constexpr (has_resolve<Batch>::value)
    return batch.resolve();
  else
    return batch;
}
template <typename Batch, typename = void>
struct has_output_index : std::false_type {};
template <typename Batch>
struct has_output_index<Batch, std::void_t<decltype(std::declval<Batch>().output_index(int64_t{}))>>
  : std::true_type {};
template <typename Batch>
__device__ int64_t output_index(Batch const& batch, int64_t index)
{
  if constexpr (has_output_index<Batch>::value)
    return batch.output_index(index);
  else
    return index;
}

template <typename S>
struct is_lrb_storage : std::false_type {};
template <typename R, typename O, typename C, typename B>
struct is_lrb_storage<TileLrbBuildStorage<R, O, C, B>> : std::true_type {};
template <typename S>
__host__ __device__ bool needs_record_lrb(S const& storage)
{
  if constexpr (is_lrb_storage<S>::value)
    return storage.table.record_count &&
           !(storage.bucket_span > 0 && storage.table.bucket_count <= storage.bucket_span);
  else
    return false;
}

template <typename Kernel>
__host__ cudaError_t configure_shared(Kernel kernel, size_t bytes)
{
  if (bytes > size_t(std::numeric_limits<int>::max())) return cudaErrorInvalidValue;
  cudaFuncAttributes attributes{};
  auto error = cudaFuncGetAttributes(&attributes, kernel);
  if (error != cudaSuccess) return error;
  int device, limit;
  error = cudaGetDevice(&device);
  if (error != cudaSuccess) return error;
  error = cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (error != cudaSuccess) return error;
  if (attributes.sharedSizeBytes > size_t(limit) ||
      bytes > size_t(limit) - attributes.sharedSizeBytes)
    return cudaErrorInvalidConfiguration;
  if (bytes <= size_t(attributes.maxDynamicSharedSizeBytes)) return cudaSuccess;
  // Always opt in to the same device limit, so concurrent callers never lower it.
  return cudaFuncSetAttribute(
    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, limit - int(attributes.sharedSizeBytes));
}

template <typename Batch, typename Status, typename View>
__host__ bool valid_outputs(Batch const& batch, Status* statuses, View* views)
{
  return batch.valid() && (!batch.table_count || statuses) &&
         slab_fits<Status>(batch.table_count, 1) &&
         (!views || slab_fits<View>(batch.table_count, 1));
}

template <int Threads,
          typename Record,
          typename Hash,
          typename Offset,
          typename Counter,
          typename Batch,
          typename Policy,
          TileBatchMemory Memory,
          typename Bucket>
struct BlockArguments {
  Batch batch;
  TileBuildStatus* statuses;
  TileCsrView<Record, Hash, Offset, Bucket>* views;
  size_t shared_bytes;
  Hash hash;
};

template <int Threads,
          typename Record,
          typename Hash,
          typename Offset,
          typename Counter,
          typename Batch,
          typename Policy,
          TileBatchMemory Memory,
          auto StaticBuckets = 0,
          typename Bucket    = uint32_t>
__global__ void block_build(
  BlockArguments<Threads, Record, Hash, Offset, Counter, Batch, Policy, Memory, Bucket> args)
{
  using Block = BlockHashGraph<Threads, Record, Hash, Offset, Counter, StaticBuckets, Bucket>;
  using Shared =
    SharedBlockHashGraph<Threads, Record, Hash, Offset, Counter, StaticBuckets, Bucket>;
  constexpr size_t storage_alignment =
    alignof(Record) > alignof(Offset)
      ? (alignof(Record) > alignof(Counter) ? alignof(Record) : alignof(Counter))
      : (alignof(Offset) > alignof(Counter) ? alignof(Offset) : alignof(Counter));
  constexpr size_t alignment =
    storage_alignment > alignof(Bucket) ? storage_alignment : alignof(Bucket);
  __shared__ typename Block::TempStorage temp;
  extern __shared__ __align__(alignment) unsigned char workspace[];
  auto const batch = resolve(args.batch);
  for (int64_t index = blockIdx.x; index < batch.table_count;) {
    auto const storage        = batch[index];
    int64_t const destination = output_index(batch, index);
    Policy policy{};
    TileBuildStatus result;
    if constexpr (Memory == TileBatchMemory::global) {
      if constexpr (Policy::shared_buckets) {
        policy.bucket_cache    = reinterpret_cast<Bucket*>(workspace);
        policy.bucket_capacity = int64_t(args.shared_bytes / sizeof(Bucket));
      }
      result = Block(temp).build(storage, args.hash, {}, policy);
    } else {
      result = Shared(temp).build(storage,
                                  workspace,
                                  args.shared_bytes,
                                  Memory == TileBatchMemory::shared_output,
                                  args.hash,
                                  {},
                                  policy);
    }
    if (!threadIdx.x) {
      args.statuses[destination] = result;
      if (args.views)
        args.views[destination] =
          result == TileBuildStatus::success
            ? table(storage).view(args.hash)
            : TileCsrView<Record, Hash, Offset, Bucket>{nullptr, nullptr, Offset{0}, args.hash};
    }
    if (batch.table_count - index <= gridDim.x) break;
    __syncthreads();
    index += gridDim.x;
  }
}

template <typename Action>
__host__ cudaError_t with_memory(TileBatchMemory memory, Action action)
{
  switch (memory) {
    case TileBatchMemory::global:
      return action(std::integral_constant<TileBatchMemory, TileBatchMemory::global>{});
    case TileBatchMemory::shared_metadata:
      return action(std::integral_constant<TileBatchMemory, TileBatchMemory::shared_metadata>{});
    case TileBatchMemory::shared_output:
      return action(std::integral_constant<TileBatchMemory, TileBatchMemory::shared_output>{});
  }
  return cudaErrorInvalidValue;
}
}  // namespace batch_detail

template <int Threads,
          typename Record    = KeyRecord,
          typename Hash      = cuco::default_hash_function<RecordKey<Record>>,
          typename Offset    = int32_t,
          typename Counter   = Offset,
          auto StaticBuckets = 0,
          typename Bucket    = uint32_t>
class BatchedBlockHashGraph {
  static_assert(Threads > 0 && Threads <= 1024);
  static_assert(detail::valid_static_buckets<StaticBuckets, Offset, Bucket>,
                "Static buckets must be zero or a representable power of two");
  using Storage    = TileBuildStorage<Record, Offset, Counter, Bucket>;
  using LrbStorage = TileLrbBuildStorage<Record, Offset, Counter, Bucket>;
  using View       = TileCsrView<Record, Hash, Offset, Bucket>;
  template <typename Batch>
  __host__ static void check_type()
  {
    static_assert(std::is_same_v<typename Batch::storage_type, Storage> ||
                    std::is_same_v<typename Batch::storage_type, LrbStorage>,
                  "Batch storage must match builder types");
  }
  template <typename Batch>
  __host__ static bool matches_buckets(Batch const&)
  {
    return true;
  }
  template <typename S>
  __host__ static bool matches_buckets(UniformTableBatch<S> const& batch)
  {
    return !StaticBuckets || !batch.table_count ||
           batch_detail::table(batch.first).bucket_count == StaticBuckets;
  }
  template <typename Policy>
  __host__ static bool valid_options(TileBatchOptions options)
  {
    static_assert(std::is_same_v<typename Policy::bucket_type, Bucket>,
                  "Bucket cache policy must match builder bucket type");
    return options.max_blocks >= 0 &&
           options.shared_bytes <= size_t(std::numeric_limits<int>::max()) &&
           (options.memory == TileBatchMemory::global ||
            options.memory == TileBatchMemory::shared_metadata ||
            options.memory == TileBatchMemory::shared_output) &&
           (!Policy::shared_pad_period || options.memory == TileBatchMemory::shared_output);
  }

 public:
  template <typename Policy = TileBuildPolicy<0, false, 0, 0, Bucket>>
  __host__ static size_t workspace_bytes(int64_t records,
                                         int64_t buckets,
                                         TileBatchMemory memory,
                                         Policy policy = {})
  {
    static_assert(std::is_same_v<typename Policy::bucket_type, Bucket>,
                  "Bucket cache policy must match builder bucket type");
    if ((StaticBuckets && buckets != StaticBuckets) ||
        !detail::valid_sizes<Offset, Counter, Bucket>(records, buckets) ||
        uint64_t(records) > std::numeric_limits<size_t>::max() / sizeof(Record))
      return 0;
    if (memory == TileBatchMemory::global) {
      if constexpr (Policy::shared_pad_period) return 0;
      return policy.cache_bytes(records, Threads);
    }
    if (memory != TileBatchMemory::shared_metadata && memory != TileBatchMemory::shared_output)
      return 0;
    return SharedBlockHashGraph<Threads, Record, Hash, Offset, Counter, StaticBuckets, Bucket>::
      workspace_bytes(records, buckets, memory == TileBatchMemory::shared_output, policy);
  }

  // Configure once outside graph capture; no allocation or per-table device reads.
  template <typename Batch, typename Policy = TileBuildPolicy<0, false, 0, 0, Bucket>>
  __host__ static cudaError_t configure(Batch batch, TileBatchOptions options = {}, Policy = {})
  {
    check_type<Batch>();
    if (!batch.valid() || !matches_buckets(batch) || !valid_options<Policy>(options))
      return cudaErrorInvalidValue;
    if (!batch.table_count) return cudaSuccess;
    return batch_detail::with_memory(options.memory, [&](auto memory) {
      return batch_detail::configure_shared(batch_detail::block_build<Threads,
                                                                      Record,
                                                                      Hash,
                                                                      Offset,
                                                                      Counter,
                                                                      Batch,
                                                                      Policy,
                                                                      decltype(memory)::value,
                                                                      StaticBuckets,
                                                                      Bucket>,
                                            options.shared_bytes);
    });
  }

  // Launch only. Descriptor failures are reported independently in device statuses.
  template <typename Batch, typename Policy = TileBuildPolicy<0, false, 0, 0, Bucket>>
  __host__ static cudaError_t build(Batch batch,
                                    TileBuildStatus* statuses,
                                    TileBatchOptions options = {},
                                    cudaStream_t stream      = nullptr,
                                    Hash hash                = {},
                                    Policy                   = {},
                                    View* views              = nullptr)
  {
    check_type<Batch>();
    if (!batch_detail::valid_outputs(batch, statuses, views) || !matches_buckets(batch) ||
        !valid_options<Policy>(options))
      return cudaErrorInvalidValue;
    if (!batch.table_count) return cudaSuccess;
    int const limit = options.max_blocks ? options.max_blocks : std::numeric_limits<int>::max();
    unsigned const blocks = unsigned(batch.table_count < limit ? batch.table_count : limit);
    return batch_detail::with_memory(options.memory, [&](auto memory) {
      using Arguments = batch_detail::BlockArguments<Threads,
                                                     Record,
                                                     Hash,
                                                     Offset,
                                                     Counter,
                                                     Batch,
                                                     Policy,
                                                     decltype(memory)::value,
                                                     Bucket>;
      Arguments arguments{batch, statuses, views, options.shared_bytes, hash};
      void* parameters[]{&arguments};
      return cudaLaunchKernel(batch_detail::block_build<Threads,
                                                        Record,
                                                        Hash,
                                                        Offset,
                                                        Counter,
                                                        Batch,
                                                        Policy,
                                                        decltype(memory)::value,
                                                        StaticBuckets,
                                                        Bucket>,
                              dim3(blocks),
                              dim3(Threads),
                              parameters,
                              options.shared_bytes,
                              stream);
    });
  }
};
}  // namespace cuco::detail::bulk_tiles
