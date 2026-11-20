/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "batch_block.cuh"

#include <cuda/std/limits>

#include <cooperative_groups.h>

namespace cuco::detail::bulk_tiles {
namespace cooperative_detail {
template <typename R, typename O, typename C, typename Bucket>
__host__ __device__ __forceinline__ TileBuildStatus
validate_storage(TileBuildStorage<R, O, C, Bucket> const& storage, int threads, Bucket buckets)
{
  if (!detail::valid_sizes<O, C, Bucket>(storage.record_count, storage.bucket_count) ||
      storage.bucket_count != buckets || storage.record_count > threads ||
      uint64_t(storage.record_count) > cuda::std::numeric_limits<size_t>::max() / sizeof(R))
    return TileBuildStatus::invalid_size;
  if (storage.record_capacity < storage.record_count || storage.offset_capacity <= buckets)
    return TileBuildStatus::insufficient_capacity;
  if (!storage.offsets || (storage.record_count && (!storage.input || !storage.records ||
                                                    storage.input == storage.records)))
    return TileBuildStatus::invalid_storage;
  return TileBuildStatus::success;
}
}  // namespace cooperative_detail

// One record per lane; every member supplies identical storage and hash metadata.
template <int Threads,
          auto Buckets,
          typename Record  = KeyRecord,
          typename Hash    = cuco::default_hash_function<RecordKey<Record>>,
          typename Offset  = int32_t,
          typename Counter = Offset,
          typename Bucket  = uint32_t>
class CooperativeTileHashGraph {
  static_assert(Threads == 8 || Threads == 16 || Threads == 32);
  static_assert(Buckets && !(Buckets & (Buckets - 1)),
                "Bucket count must be a positive power of two");
  static_assert(detail::valid_static_buckets<Buckets, Offset, Bucket>);
  using Storage = TileBuildStorage<Record, Offset, Counter, Bucket>;

 public:
  __host__ __device__ static TileBuildStatus validate(Storage const& storage)
  {
    return cooperative_detail::validate_storage(storage, Threads, Bucket(Buckets));
  }

  template <typename Parent>
  __device__ static TileBuildStatus build(
    cooperative_groups::thread_block_tile<Threads, Parent> const& group,
    Storage const& storage,
    Hash hash = {})
  {
    auto const status = validate(storage);
    if (status != TileBuildStatus::success) return status;
    int const n = int(storage.record_count), lane = int(group.thread_rank());
    if (!__isGlobal(storage.offsets) ||
        (n && (!__isGlobal(storage.input) || !__isGlobal(storage.records))))
      return TileBuildStatus::invalid_storage;
    if constexpr (Buckets == 1) {
      if (lane < 2) storage.offsets[lane] = lane ? Offset(n) : Offset{0};
      if (lane < n) storage.records[lane] = storage.input[lane];
      group.sync();
      return TileBuildStatus::success;
    } else {
      if (!n) {
        using Index = detail::index_type<Offset>;
        for (Index b = lane; b < Index(Buckets);) {
          storage.offsets[b] = 0;
          if (!detail::advance_index(b, Index(Buckets), Index(Threads))) break;
        }
        if (!lane) storage.offsets[Buckets] = 0;
        group.sync();
        return TileBuildStatus::success;
      }
      bool const active   = lane < n;
      Record const record = storage.input[active ? lane : 0];
      asm volatile("" ::: "memory");
      Bucket const bucket =
        active ? detail::reduce_hash<Bucket>(hash(record.first), Bucket(Buckets)) : 0;
      unsigned const preceding = (1u << lane) - 1;
      Offset prefix = 0, position = 0;
// Ballots are relative to this subgroup, including groups sharing a warp.
#pragma unroll 8
      for (Bucket b = 0; b < Bucket(Buckets); ++b) {
        unsigned const members = group.ballot(active && bucket == b);
        if (unsigned(lane) == b % Threads) storage.offsets[b] = prefix;
        if (active && bucket == b) position = prefix + Offset(__popc(members & preceding));
        prefix += Offset(__popc(members));
      }
      if (!lane) storage.offsets[Buckets] = Offset(n);
      if (active) storage.records[position] = record;
      group.sync();
      return TileBuildStatus::success;
    }
  }
};

struct CooperativeBatchOptions {
  int max_blocks = 0;
};
namespace cooperative_detail {
template <typename Batch>
struct is_uniform : std::false_type {};
template <typename Storage>
struct is_uniform<UniformTableBatch<Storage>> : std::true_type {};
template <typename Batch>
__host__ bool valid(Batch const& batch)
{
  return batch.valid();
}
template <typename R, typename O, typename C, typename Bucket>
__host__ bool valid(UniformTableBatch<TileBuildStorage<R, O, C, Bucket>> const& batch)
{
  auto const& first = batch.first;
  return batch.table_count >= 0 &&
         (!batch.table_count ||
          (batch_detail::slab_fits<R>(batch.table_count, first.record_count) &&
           batch_detail::slab_fits<R>(batch.table_count, first.record_capacity) &&
           batch_detail::slab_fits<O>(batch.table_count, first.offset_capacity)));
}
template <typename Batch>
__device__ auto at(Batch const& batch, int64_t index)
{
  return batch[index];
}
template <typename R, typename O, typename C, typename Bucket>
__device__ auto at(UniformTableBatch<TileBuildStorage<R, O, C, Bucket>> const& batch, int64_t index)
{
  auto storage    = batch.first;
  storage.input   = batch_detail::advance(storage.input, index, storage.record_count);
  storage.records = batch_detail::advance(storage.records, index, storage.record_capacity);
  storage.offsets = batch_detail::advance(storage.offsets, index, storage.offset_capacity);
  return storage;
}
template <typename Batch, typename Hash, typename View>
struct Arguments {
  Batch batch;
  TileBuildStatus* statuses;
  View* views;
  Hash hash;
};
template <int Threads,
          auto Buckets,
          int BlockThreads,
          typename R,
          typename H,
          typename O,
          typename C,
          typename Batch,
          typename Bucket>
__global__ void build(Arguments<Batch, H, TileCsrView<R, H, O, Bucket>> args)
{
  constexpr int groups = BlockThreads / Threads;
  auto const tile =
    cooperative_groups::tiled_partition<Threads>(cooperative_groups::this_thread_block());
  auto const batch     = batch_detail::resolve(args.batch);
  int64_t const stride = int64_t(gridDim.x) * groups;
  for (int64_t index = int64_t(blockIdx.x) * groups + threadIdx.x / Threads;
       index < batch.table_count;) {
    auto const storage = at(batch, index);
    auto const status  = CooperativeTileHashGraph<Threads, Buckets, R, H, O, C, Bucket>::build(
      tile, storage, args.hash);
    if (!tile.thread_rank()) {
      auto const destination     = batch_detail::output_index(batch, index);
      args.statuses[destination] = status;
      if (args.views)
        args.views[destination] =
          status == TileBuildStatus::success
            ? storage.view(args.hash)
            : TileCsrView<R, H, O, Bucket>{nullptr, nullptr, O{0}, args.hash};
    }
    if (batch.table_count - index <= stride) break;
    index += stride;
  }
}
}  // namespace cooperative_detail

template <int Threads,
          auto Buckets,
          int BlockThreads = 128,
          typename Record  = KeyRecord,
          typename Hash    = cuco::default_hash_function<RecordKey<Record>>,
          typename Offset  = int32_t,
          typename Counter = Offset,
          typename Bucket  = uint32_t>
class BatchedCooperativeHashGraph {
  static_assert(BlockThreads > 0 && BlockThreads <= 1024 && !(BlockThreads % 32));
  static_assert(BlockThreads % Threads == 0);
  using Storage = TileBuildStorage<Record, Offset, Counter, Bucket>;
  using Builder = CooperativeTileHashGraph<Threads, Buckets, Record, Hash, Offset, Counter, Bucket>;
  using View    = TileCsrView<Record, Hash, Offset, Bucket>;
  template <typename Batch>
  __host__ static bool valid(Batch batch, CooperativeBatchOptions options)
  {
    static_assert(std::is_same_v<typename Batch::storage_type, Storage>);
    if (options.max_blocks < 0 || !cooperative_detail::valid(batch)) return false;
    if constexpr (cooperative_detail::is_uniform<Batch>::value)
      if (batch.table_count && Builder::validate(batch.first) != TileBuildStatus::success)
        return false;
    return true;
  }

 public:
  template <typename Batch>
  __host__ static cudaError_t configure(Batch batch, CooperativeBatchOptions options = {})
  {
    if (!valid(batch, options)) return cudaErrorInvalidValue;
    if (!batch.table_count) return cudaSuccess;
    auto kernel = cooperative_detail::
      build<Threads, Buckets, BlockThreads, Record, Hash, Offset, Counter, Batch, Bucket>;
    cudaFuncAttributes attributes{};
    auto error = cudaFuncGetAttributes(&attributes, kernel);
    if (error != cudaSuccess) return error;
    if (BlockThreads > attributes.maxThreadsPerBlock) return cudaErrorInvalidConfiguration;
    int capacity = 0;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&capacity, kernel, BlockThreads, 0);
    if (error != cudaSuccess) return error;
    return capacity ? cudaSuccess : cudaErrorInvalidConfiguration;
  }
  template <typename Batch>
  __host__ static cudaError_t build(Batch batch,
                                    TileBuildStatus* statuses,
                                    CooperativeBatchOptions options = {},
                                    cudaStream_t stream             = nullptr,
                                    Hash hash                       = {},
                                    View* views                     = nullptr)
  {
    if (!valid(batch, options) || (batch.table_count && !statuses) ||
        !batch_detail::slab_fits<TileBuildStatus>(batch.table_count, 1) ||
        (views && !batch_detail::slab_fits<View>(batch.table_count, 1)))
      return cudaErrorInvalidValue;
    if (!batch.table_count) return cudaSuccess;
    constexpr int groups = BlockThreads / Threads;
    int64_t const needed = 1 + (batch.table_count - 1) / groups;
    int const limit =
      options.max_blocks ? options.max_blocks : cuda::std::numeric_limits<int>::max();
    unsigned const blocks = unsigned(needed < limit ? needed : limit);
    cooperative_detail::Arguments<Batch, Hash, View> args{batch, statuses, views, hash};
    void* parameters[]{&args};
    return cudaLaunchKernel(
      cooperative_detail::
        build<Threads, Buckets, BlockThreads, Record, Hash, Offset, Counter, Batch, Bucket>,
      dim3(blocks),
      dim3(BlockThreads),
      parameters,
      0,
      stream);
  }
};
}  // namespace cuco::detail::bulk_tiles
