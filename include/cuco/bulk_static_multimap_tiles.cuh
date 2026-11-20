/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuco/detail/bulk_static_multimap/tiles/scheduler.cuh>
#include <cuco/detail/bulk_static_multimap/tiles/device_build.cuh>
#include <cuda/stream>

namespace cuco {
/**
 * @brief Caller-owned optimized bulk multimap construction APIs.
 *
 * Adapted from the HashGraph tile algorithms at 444c92c. All records are cuco::pair<Key,T>.
 * Input/output/workspace arrays must be disjoint and live through build completion. Build is
 * collective within the selected subgroup, block or cluster. Query refs require successful
 * construction and external ordering; keys, equality and hash must agree.
 */
template <class Key, class T, class Offset = uint32_t, class Counter = Offset, class Bucket = uint32_t>
using bulk_static_multimap_storage = detail::bulk_tiles::TileBuildStorage<pair<Key,T>,Offset,Counter,Bucket>;

template <class Key, class T, class Offset = uint32_t, class Counter = Offset, class Bucket = uint32_t>
using bulk_static_multimap_lrb_storage = detail::bulk_tiles::TileLrbBuildStorage<pair<Key,T>,Offset,Counter,Bucket>;

template <class Key, class T, class Hash = default_hash_function<Key>, class Offset = uint32_t, class Bucket = uint32_t>
using bulk_static_multimap_retained_storage = detail::bulk_tiles::TileCsrView<pair<Key,T>,Hash,Offset,Bucket>;

template <int RegisterItems = 0, bool SharedIds = false, int PrefetchItems = 0,
          int SharedPadPeriod = 0, class Bucket = uint32_t>
using bulk_static_multimap_block_policy = detail::bulk_tiles::TileBuildPolicy<RegisterItems,SharedIds,PrefetchItems,SharedPadPeriod,Bucket>;

template <int PrefetchItems = 0, bool CacheIds = false, bool SharedBins = false, class Bucket = uint32_t>
using bulk_static_multimap_cluster_policy = detail::bulk_tiles::ClusterBuildPolicy<PrefetchItems,CacheIds,SharedBins,Bucket>;

template <class Key, class T, int Threads = 128, class Hash = default_hash_function<Key>,
          class Offset = uint32_t, class Counter = Offset, auto StaticBuckets = 0, class Bucket = uint32_t>
using bulk_static_multimap_block_builder = detail::bulk_tiles::BlockHashGraph<Threads,pair<Key,T>,Hash,Offset,Counter,StaticBuckets,Bucket>;

template <class Key, class T, int Threads = 128, class Hash = default_hash_function<Key>,
          class Offset = uint32_t, class Counter = Offset, auto StaticBuckets = 0, class Bucket = uint32_t>
using bulk_static_multimap_shared_builder = detail::bulk_tiles::SharedBlockHashGraph<Threads,pair<Key,T>,Hash,Offset,Counter,StaticBuckets,Bucket>;

template <class Key, class T, int Threads = 128, class Hash = default_hash_function<Key>,
          class Offset = uint32_t, class Counter = Offset, class Bucket = uint32_t>
using bulk_static_multimap_cluster_builder = detail::bulk_tiles::ClusterHashGraph<Threads,pair<Key,T>,Hash,Offset,Counter,Bucket>;

template <class Key, class T, int Threads, auto Buckets, class Hash = default_hash_function<Key>,
          class Offset = uint32_t, class Counter = Offset, class Bucket = uint32_t>
using bulk_static_multimap_subgroup_builder = detail::bulk_tiles::CooperativeTileHashGraph<Threads,Buckets,pair<Key,T>,Hash,Offset,Counter,Bucket>;

template <class Storage>
using bulk_static_multimap_descriptor_batch = detail::bulk_tiles::DeviceTableBatch<Storage>;
template <class Storage>
using bulk_static_multimap_uniform_batch = detail::bulk_tiles::UniformTableBatch<Storage>;
using bulk_static_multimap_memory = detail::bulk_tiles::TileBatchMemory;
using bulk_static_multimap_batch_options = detail::bulk_tiles::TileBatchOptions;
using bulk_static_multimap_subgroup_options = detail::bulk_tiles::CooperativeBatchOptions;
template <class Bucket = uint32_t>
using bulk_static_multimap_bucket_cache = detail::bulk_tiles::TypedTableBucketCache<Bucket>;
template <class Bucket = uint32_t>
using bulk_static_multimap_bucket_caches = detail::bulk_tiles::TypedBatchBucketCaches<Bucket>;
template <class Bucket = uint32_t>
using bulk_static_multimap_cluster_options = detail::bulk_tiles::TypedClusterBatchOptions<Bucket>;
template <class Bucket = uint32_t>
using bulk_static_multimap_schedule_options = detail::bulk_tiles::TypedTableScheduleOptions<Bucket>;
using bulk_static_multimap_schedule_kind = detail::bulk_tiles::TableScheduleKind;
using bulk_static_multimap_device_policy = detail::bulk_tiles::SingleBuildPolicy;
using bulk_static_multimap_device_workspace = detail::bulk_tiles::SingleBuildWorkspace;

template <class Record, class Hash, class Offset, class Bucket, class KeyEqual, class... Operators>
__host__ __device__ auto make_bulk_static_multimap_ref(
  detail::bulk_tiles::TileCsrView<Record,Hash,Offset,Bucket> storage, KeyEqual equal, Operators...)
{
  return bulk_static_multimap_ref<typename Record::first_type,typename Record::second_type,
    Offset,KeyEqual,Hash,Operators...>{storage.records,storage.offsets,storage.size,storage.buckets,equal,storage.hash};
}

template <class Key, class T, int Threads = 128, class Hash = default_hash_function<Key>,
          class Offset = uint32_t, class Counter = Offset, auto StaticBuckets = 0, class Bucket = uint32_t>
class bulk_static_multimap_block_batch {
  using Engine = detail::bulk_tiles::BatchedBlockHashGraph<Threads,pair<Key,T>,Hash,Offset,Counter,StaticBuckets,Bucket>;
  using View = bulk_static_multimap_retained_storage<Key,T,Hash,Offset,Bucket>;
 public:
  template <class Policy = bulk_static_multimap_block_policy<0,false,0,0,Bucket>>
  static size_t workspace_bytes(int64_t n,int64_t b,bulk_static_multimap_memory memory,Policy policy = {})
  { return Engine::workspace_bytes(n,b,memory,policy); }
  template <class Batch,class Policy = bulk_static_multimap_block_policy<0,false,0,0,Bucket>>
  static cudaError_t configure(Batch batch,bulk_static_multimap_batch_options options = {},Policy policy = {})
  { return Engine::configure(batch,options,policy); }
  template <class Batch,class Policy = bulk_static_multimap_block_policy<0,false,0,0,Bucket>>
  static cudaError_t build_async(Batch batch,bulk_build_status* status,
    bulk_static_multimap_batch_options options = {},cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}},
    Hash hash = {},Policy policy = {},View* views = nullptr)
  { return Engine::build(batch,status,options,stream.get(),hash,policy,views); }
};

template <class Key,class T,int Threads,auto Buckets,int BlockThreads = 128,
          class Hash = default_hash_function<Key>,class Offset = uint32_t,class Counter = Offset,class Bucket = uint32_t>
class bulk_static_multimap_subgroup_batch {
  using Engine = detail::bulk_tiles::BatchedCooperativeHashGraph<Threads,Buckets,BlockThreads,pair<Key,T>,Hash,Offset,Counter,Bucket>;
  using View = bulk_static_multimap_retained_storage<Key,T,Hash,Offset,Bucket>;
 public:
  template <class Batch>
  static cudaError_t configure(Batch batch,bulk_static_multimap_subgroup_options options = {})
  { return Engine::configure(batch,options); }
  template <class Batch>
  static cudaError_t build_async(Batch batch,bulk_build_status* status,
    bulk_static_multimap_subgroup_options options = {},cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}},
    Hash hash = {},View* views = nullptr)
  { return Engine::build(batch,status,options,stream.get(),hash,views); }
};

template <class Key,class T,int Threads = 128,class Hash = default_hash_function<Key>,
          class Offset = uint32_t,class Counter = Offset,class Bucket = uint32_t>
class bulk_static_multimap_cluster_batch {
  using Engine = detail::bulk_tiles::BatchedClusterHashGraph<Threads,pair<Key,T>,Hash,Offset,Counter,Bucket>;
  using View = bulk_static_multimap_retained_storage<Key,T,Hash,Offset,Bucket>;
 public:
  template <class Policy = bulk_static_multimap_cluster_policy<0,false,false,Bucket>>
  static size_t workspace_bytes(int64_t bins,int blocks,Policy policy = {})
  { return Engine::workspace_bytes(bins,blocks,policy); }
  template <class Batch,class Policy = bulk_static_multimap_cluster_policy<0,false,false,Bucket>>
  static cudaError_t configure(Batch batch,bulk_static_multimap_cluster_options<Bucket> options = {},
                               Policy policy = {},int* active_clusters = nullptr)
  { return Engine::configure(batch,options,policy,active_clusters); }
  template <class Batch,class Policy = bulk_static_multimap_cluster_policy<0,false,false,Bucket>>
  static cudaError_t build_async(Batch batch,bulk_build_status* status,
    bulk_static_multimap_cluster_options<Bucket> options = {},cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}},
    Hash hash = {},Policy policy = {},View* views = nullptr)
  { return Engine::build(batch,status,options,stream.get(),hash,policy,views); }
};

template <class Key,class T,class Hash = default_hash_function<Key>,class Offset = uint32_t,
          class Counter = Offset,int Items = 8,class Bucket = uint32_t>
class bulk_static_multimap_scheduler {
  using Engine = detail::bulk_tiles::ScheduledHashGraph<pair<Key,T>,Hash,Offset,Counter,Items,Bucket>;
  using View = bulk_static_multimap_retained_storage<Key,T,Hash,Offset,Bucket>;
 public:
  template <class Batch> using configuration = typename Engine::template Configuration<Batch>;
  static constexpr int class_count = Engine::class_count;
  static constexpr int range_count = Engine::range_count;
  static constexpr auto class_from_key(uint32_t key) { return Engine::class_from_key(key); }
  static constexpr auto range_from_key(uint32_t key) { return Engine::range_from_key(key); }
  static constexpr auto record_lrb_from_key(uint32_t key) { return Engine::record_lrb_from_key(key); }
  static constexpr auto bucket_count(int64_t n,int64_t target = 8) { return Engine::bucket_count(n,target); }
  template <class Batch>
  static cudaError_t configure(Batch batch,configuration<Batch>& config,
                               bulk_static_multimap_schedule_options<Bucket> options = {})
  { return Engine::configure(batch,config,options); }
  template <class Batch>
  static size_t workspace_bytes(configuration<Batch> const& config) { return Engine::workspace_bytes(config); }
  template <class Batch>
  static auto classification_keys(configuration<Batch> const& config,void* scratch)
  { return Engine::classification_keys(config,scratch); }
  template <class Batch>
  static auto ranges(configuration<Batch> const& config,void* scratch) { return Engine::ranges(config,scratch); }
  template <class Batch>
  static cudaError_t build_async(configuration<Batch> const& config,Batch batch,bulk_build_status* status,
    void* scratch,size_t bytes,cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}},Hash hash = {},View* views = nullptr)
  { return Engine::build(config,batch,status,scratch,bytes,stream.get(),hash,views); }
};

template <class Key,class T,class Offset,class Counter,class Bucket>
cudaError_t bulk_static_multimap_device_workspace_bytes(
  bulk_static_multimap_lrb_storage<Key,T,Offset,Counter,Bucket> const& storage,size_t* bytes)
{ return detail::bulk_tiles::single_lrb_workspace_bytes(storage,bytes); }

template <class Key,class T,class Offset,class Counter,class Bucket,class Hash = default_hash_function<Key>>
cudaError_t bulk_static_multimap_device_build_async(
  bulk_static_multimap_lrb_storage<Key,T,Offset,Counter,Bucket> const& storage,
  bulk_static_multimap_device_workspace workspace,bulk_static_multimap_device_policy policy = {},
  cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}},Hash hash = {})
{ return detail::bulk_tiles::single_lrb_build(storage,workspace,policy,stream.get(),nullptr,hash); }
}  // namespace cuco
