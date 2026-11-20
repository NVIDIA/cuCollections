/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "block.cuh"
#include <cooperative_groups.h>

namespace cuco::detail::bulk_tiles {

template <int PrefetchItems = 0, bool CacheBuckets = false, bool SharedLrbBins = false, typename Bucket = uint32_t>
struct ClusterBuildPolicy {
    static_assert(PrefetchItems >= 0);
    static constexpr int prefetch_items = PrefetchItems;
    static constexpr bool cache_buckets = CacheBuckets;
    using bucket_type = Bucket;
    Bucket* bucket_cache = nullptr;
    int64_t bucket_capacity = 0;
    void* shared_workspace = nullptr;
    size_t workspace_capacity = 0;
    static constexpr bool shared_lrb_bins = SharedLrbBins;
    template <typename Offset = int32_t, typename Counter = Offset>
    __host__ __device__ static size_t workspace_bytes(int64_t bins, int blocks) {
        if constexpr (!SharedLrbBins) return 0;
        if (!detail::valid_sizes<Offset, Counter, Bucket>(0, bins) || blocks <= 0) return 0;
        static_assert(detail::storage_integer<Offset> && detail::storage_integer<Counter>);
        const size_t slots = size_t((1 + (bins - 1) / blocks));
        size_t bytes = 0, start;
        if (!detail::reserve_region(bytes, slots, sizeof(Counter), alignof(Counter), start) ||
            !detail::reserve_region(bytes, slots, sizeof(Offset), alignof(Offset), start)) return 0;
        return bytes;
    }
};

// Collective over a complete 1D cluster; descriptors must be cluster-uniform.
// All table arrays are global and disjoint. Each block supplies shared temp.
template <int BlockThreads, typename Record = KeyRecord, typename Hash = cuco::default_hash_function<RecordKey<Record>>,
          typename Offset = int32_t, typename Counter = Offset, typename Bucket = uint32_t>
class ClusterHashGraph {
    static_assert(BlockThreads > 0 && BlockThreads <= 1024);
    using Block = BlockHashGraph<BlockThreads, Record, Hash, Offset, Counter, 0, Bucket>;
    using Index = detail::index_type<Offset>;

public:
    struct TempStorage {
        typename Block::TempStorage block;
        Offset total, base;
    };

    __device__ explicit ClusterHashGraph(TempStorage& temp) : temp_(temp) {}

    template <typename Observer = TileBuildNoObserver, int PrefetchItems = 0, bool CacheBuckets = false, bool SharedLrbBins = false>
    __device__ TileBuildStatus build(const TileBuildStorage<Record, Offset, Counter, Bucket>& storage,
                                     Hash hash = {}, Observer observer = {}, ClusterBuildPolicy<PrefetchItems, CacheBuckets, SharedLrbBins, Bucket> policy = {}) {
        return run(storage, hash, observer, policy, false);
    }

private:
    // Compile-time slots keep bounded prefetch windows in registers.
    template <typename F, size_t... Items>
    __device__ __forceinline__ static void prefetch_each(F f, std::index_sequence<Items...>) {
        (f(std::integral_constant<size_t, Items>{}), ...);
    }

    template <typename Observer, int PrefetchItems, bool CacheBuckets, bool SharedLrbBins>
    __device__ TileBuildStatus run(const TileBuildStorage<Record, Offset, Counter, Bucket>& storage, Hash hash, Observer observer,
                                   ClusterBuildPolicy<PrefetchItems, CacheBuckets, SharedLrbBins, Bucket> policy, bool cached_input) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        if (blockDim.x != BlockThreads || blockDim.y != 1 || blockDim.z != 1)
            return TileBuildStatus::invalid_block;
        const auto cluster = cooperative_groups::this_cluster();
        if (cluster.dim_blocks().y != 1 || cluster.dim_blocks().z != 1)
            return TileBuildStatus::invalid_cluster;
        const auto status = storage.validate();
        if (status != TileBuildStatus::success) return status;
        if (!__isGlobal(storage.offsets) || !__isGlobal(storage.counters) ||
            (storage.record_count && (!__isGlobal(storage.input) || !__isGlobal(storage.records))))
            return TileBuildStatus::invalid_storage;
        const auto cache_status = validate_cache(storage, policy);
        if (cache_status != TileBuildStatus::success) return cache_status;
        if constexpr (!CacheBuckets)
            if (cluster.num_blocks() == 1)
                return Block(temp_.block).build(storage, hash, observer, TileBuildPolicy<0, false, PrefetchItems, 0, Bucket>{});

        const int lane = threadIdx.x;
        const int rank = cluster.block_rank(), blocks = cluster.num_blocks();
        const int thread = rank * BlockThreads + lane, stride = blocks * BlockThreads;
        const Index n = Index(storage.record_count), b = Index(storage.bucket_count);
        const Bucket buckets = static_cast<Bucket>(b);
        cluster.sync();
        observer.mark(0);
        for (Index i = thread; i < b;) {
            storage.counters[i] = 0;
            if (!detail::advance_index(i, b, Index(stride))) break;
        }
        cluster.sync();
        observer.mark(1);
        if constexpr (PrefetchItems) {
            for (Index base = thread; base < n;) {
                RecordKey<Record> keys[PrefetchItems];
                Bucket ids[CacheBuckets ? PrefetchItems : 1];
                prefetch_each([&](auto slot) {
                    constexpr int item = decltype(slot)::value;
                    const Index delta = Index(item) * Index(stride);
                    if (delta < n - base) {
                        const Index i = base + delta;
                        if constexpr (CacheBuckets) {
                            if (cached_input) ids[item] = policy.bucket_cache[i];
                            else keys[item] = storage.input[i].first;
                        } else keys[item] = storage.input[i].first;
                    }
                }, std::make_index_sequence<PrefetchItems>{});
                // Keep independent loads ahead of their dependent atomics.
                asm volatile("" ::: "memory");
                prefetch_each([&](auto slot) {
                    constexpr int item = decltype(slot)::value;
                    const Index delta = Index(item) * Index(stride);
                    if (delta < n - base) {
                        Bucket bucket;
                        if constexpr (CacheBuckets) {
                            if (cached_input) bucket = ids[item];
                            else {
                                bucket = detail::reduce_hash<Bucket>(hash(keys[item]), buckets);
                                policy.bucket_cache[base + delta] = bucket;
                            }
                        } else bucket = detail::reduce_hash<Bucket>(hash(keys[item]), buckets);
                        detail::atomic_add(storage.counters + bucket);
                    }
                }, std::make_index_sequence<PrefetchItems>{});
                if (!detail::advance_index(base, n, Index(stride) * PrefetchItems)) break;
            }
        } else
            for (Index i = thread; i < n;) {
                Bucket bucket;
                if constexpr (CacheBuckets) {
                    if (cached_input) bucket = policy.bucket_cache[i];
                    else {
                        bucket = detail::reduce_hash<Bucket>(hash(storage.input[i].first), buckets);
                        policy.bucket_cache[i] = bucket;
                    }
                } else bucket = detail::reduce_hash<Bucket>(hash(storage.input[i].first), buckets);
                detail::atomic_add(storage.counters + bucket);
                if (!detail::advance_index(i, n, Index(stride))) break;
            }
        cluster.sync();
        observer.mark(2);

        scan_counts(storage.counters, storage.offsets, b, n);
        observer.mark(3);

        if constexpr (PrefetchItems) {
            for (Index base = thread; base < n;) {
                Record records[PrefetchItems];
                Bucket ids[CacheBuckets ? PrefetchItems : 1];
                prefetch_each([&](auto slot) {
                    constexpr int item = decltype(slot)::value;
                    const Index delta = Index(item) * Index(stride);
                    if (delta < n - base) {
                        const Index i = base + delta;
                        records[item] = storage.input[i];
                        if constexpr (CacheBuckets) ids[item] = policy.bucket_cache[i];
                    }
                }, std::make_index_sequence<PrefetchItems>{});
                asm volatile("" ::: "memory");
                prefetch_each([&](auto slot) {
                    constexpr int item = decltype(slot)::value;
                    if (Index(item) * Index(stride) < n - base) {
                        Bucket bucket;
                        if constexpr (CacheBuckets) bucket = ids[item];
                        else bucket = detail::reduce_hash<Bucket>(hash(records[item].first), buckets);
                        const Offset position = storage.offsets[bucket] + Offset(detail::atomic_add(storage.counters + bucket));
                        storage.records[position] = records[item];
                    }
                }, std::make_index_sequence<PrefetchItems>{});
                if (!detail::advance_index(base, n, Index(stride) * PrefetchItems)) break;
            }
        } else
            for (Index i = thread; i < n;) {
                const auto record = storage.input[i];
                Bucket bucket;
                if constexpr (CacheBuckets) bucket = policy.bucket_cache[i];
                else bucket = detail::reduce_hash<Bucket>(hash(record.first), buckets);
                const Offset position = storage.offsets[bucket] + Offset(detail::atomic_add(storage.counters + bucket));
                storage.records[position] = record;
                if (!detail::advance_index(i, n, Index(stride))) break;
            }
        // Also protects remote shared reads before temp can be reused or blocks exit.
        cluster.sync();
        observer.mark(4);
        return TileBuildStatus::success;
#else
        return TileBuildStatus::unsupported_architecture;
#endif
    }

public:
    template <typename Observer = TileBuildNoObserver, int PrefetchItems = 0, bool CacheBuckets = false, bool SharedLrbBins = false>
    __device__ TileBuildStatus build(const TileLrbBuildStorage<Record, Offset, Counter, Bucket>& storage,
                                     Hash hash = {}, Observer observer = {}, ClusterBuildPolicy<PrefetchItems, CacheBuckets, SharedLrbBins, Bucket> policy = {}) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        if (blockDim.x != BlockThreads || blockDim.y != 1 || blockDim.z != 1)
            return TileBuildStatus::invalid_block;
        const auto cluster = cooperative_groups::this_cluster();
        if (cluster.dim_blocks().y != 1 || cluster.dim_blocks().z != 1)
            return TileBuildStatus::invalid_cluster;
        const auto status = storage.validate();
        if (status != TileBuildStatus::success) return status;
        const auto& table = storage.table;
        if (!__isGlobal(table.offsets) || !__isGlobal(table.counters) ||
            !__isGlobal(storage.bin_counters) || !__isGlobal(storage.bin_offsets) ||
            (table.record_count && (!__isGlobal(table.input) || !__isGlobal(table.records) ||
                !__isGlobal(storage.reordered) || !__isGlobal(storage.bucket_hashes))))
            return TileBuildStatus::invalid_storage;
        const auto cache_status = validate_cache(table, policy);
        if (cache_status != TileBuildStatus::success) return cache_status;
        if constexpr (SharedLrbBins) {
            if (!policy.shared_workspace || !__isShared(policy.shared_workspace) ||
                reinterpret_cast<uintptr_t>(policy.shared_workspace) % (alignof(Counter) > alignof(Offset) ? alignof(Counter) : alignof(Offset)))
                return TileBuildStatus::invalid_storage;
            const size_t needed = policy.template workspace_bytes<Offset, Counter>(storage.bin_count(), cluster.num_blocks());
            if (!needed || policy.workspace_capacity < needed) return TileBuildStatus::insufficient_capacity;
        }
        if constexpr (CacheBuckets)
            if (table.record_count && (policy.bucket_cache == storage.bucket_hashes ||
                policy.bucket_cache == reinterpret_cast<const void*>(storage.reordered) ||
                policy.bucket_cache == reinterpret_cast<const void*>(storage.bin_counters) ||
                policy.bucket_cache == reinterpret_cast<const void*>(storage.bin_offsets)))
                return TileBuildStatus::invalid_storage;
        if constexpr (!CacheBuckets && !SharedLrbBins)
            if (cluster.num_blocks() == 1)
                return Block(temp_.block).build(storage, hash, observer, TileBuildPolicy<0, false, PrefetchItems, 0, Bucket>{});
        const int rank = cluster.block_rank(), blocks = cluster.num_blocks();
        const int thread = rank * BlockThreads + threadIdx.x, stride = blocks * BlockThreads;
        const Index bins = Index(storage.bin_count()), n = Index(table.record_count);
        Index slots = 0;
        Counter* shared_counts = nullptr;
        Offset* shared_offsets = nullptr;
        if constexpr (SharedLrbBins) {
            slots = (1 + (bins - 1) / blocks);
            shared_counts = static_cast<Counter*>(policy.shared_workspace);
            shared_offsets = reinterpret_cast<Offset*>(static_cast<unsigned char*>(policy.shared_workspace) +
                detail::align_bytes(size_t(slots) * sizeof(Counter), alignof(Offset)));
        }
        cluster.sync();
        observer.mark(0);
        if constexpr (SharedLrbBins) {
            for (Index i = threadIdx.x; i < slots;) {
                shared_counts[i] = 0;
                if (!detail::advance_index(i, slots, Index(BlockThreads))) break;
            }
        } else
            for (Index i = thread; i < bins;) {
                storage.bin_counters[i] = 0;
                if (!detail::advance_index(i, bins, Index(stride))) break;
            }
        cluster.sync();
        observer.mark(1);
        for (Index i = thread; i < n;) {
            const Bucket bucket = detail::reduce_hash<Bucket>(hash(table.input[i].first), Bucket(table.bucket_count));
            storage.bucket_hashes[i] = bucket;
            const Bucket bin = bucket / Bucket(storage.bucket_span);
            if constexpr (SharedLrbBins) {
                auto* owner_counts = cluster.map_shared_rank(shared_counts, bin % blocks);
                detail::atomic_add(owner_counts + bin / blocks);
            } else detail::atomic_add(storage.bin_counters + bin);
            if (!detail::advance_index(i, n, Index(stride))) break;
        }
        cluster.sync();
        if constexpr (SharedLrbBins) {
            for (Index i = threadIdx.x; i < slots;) {
                const bool present = Index(rank) < bins && i <= (bins - 1 - Index(rank)) / Index(blocks);
                const Index bin = present ? Index(rank) + i * Index(blocks) : 0;
                if (present) storage.bin_counters[bin] = shared_counts[i];
                if (!detail::advance_index(i, slots, Index(BlockThreads))) break;
            }
            cluster.sync();
        }
        observer.mark(2);
        scan_counts(storage.bin_counters, storage.bin_offsets, bins, n);
        if constexpr (SharedLrbBins) {
            for (Index i = threadIdx.x; i < slots;) {
                const bool present = Index(rank) < bins && i <= (bins - 1 - Index(rank)) / Index(blocks);
                const Index bin = present ? Index(rank) + i * Index(blocks) : 0;
                shared_counts[i] = 0;
                if (present) shared_offsets[i] = storage.bin_offsets[bin];
                if (!detail::advance_index(i, slots, Index(BlockThreads))) break;
            }
            cluster.sync();
        }
        observer.mark(3);
        for (Index i = thread; i < n;) {
            const Bucket bucket = storage.bucket_hashes[i];
            const Bucket bin = bucket / Bucket(storage.bucket_span);
            Offset position;
            if constexpr (SharedLrbBins) {
                auto* owner_counts = cluster.map_shared_rank(shared_counts, bin % blocks);
                const auto* owner_offsets = cluster.map_shared_rank(shared_offsets, bin % blocks);
                position = owner_offsets[bin / blocks] + Offset(detail::atomic_add(owner_counts + bin / blocks));
            } else position = storage.bin_offsets[bin] + Offset(detail::atomic_add(storage.bin_counters + bin));
            storage.reordered[position] = table.input[i];
            if constexpr (CacheBuckets) policy.bucket_cache[position] = bucket;
            if (!detail::advance_index(i, n, Index(stride))) break;
        }
        cluster.sync();
        if constexpr (SharedLrbBins) {
            for (Index i = threadIdx.x; i < slots;) {
                const bool present = Index(rank) < bins && i <= (bins - 1 - Index(rank)) / Index(blocks);
                const Index bin = present ? Index(rank) + i * Index(blocks) : 0;
                if (present) storage.bin_counters[bin] = shared_counts[i];
                if (!detail::advance_index(i, slots, Index(BlockThreads))) break;
            }
            cluster.sync();
        }
        auto reordered = table;
        reordered.input = storage.reordered;
        return run(reordered, hash, detail::LrbBucketObserver<Observer>{observer}, policy, CacheBuckets);
#else
        return TileBuildStatus::unsupported_architecture;
#endif
    }

private:
    template <int PrefetchItems, bool CacheBuckets, bool SharedLrbBins>
    __device__ static TileBuildStatus validate_cache(const TileBuildStorage<Record, Offset, Counter, Bucket>& storage,
                                                     ClusterBuildPolicy<PrefetchItems, CacheBuckets, SharedLrbBins, Bucket> policy) {
        if constexpr (CacheBuckets) {
            if (policy.bucket_capacity < storage.record_count) return TileBuildStatus::insufficient_capacity;
            if (storage.record_count && (!policy.bucket_cache || !__isGlobal(policy.bucket_cache) ||
                reinterpret_cast<uintptr_t>(policy.bucket_cache) % alignof(Bucket) ||
                policy.bucket_cache == reinterpret_cast<const void*>(storage.input) ||
                policy.bucket_cache == reinterpret_cast<const void*>(storage.records) ||
                policy.bucket_cache == reinterpret_cast<const void*>(storage.counters) ||
                policy.bucket_cache == reinterpret_cast<const void*>(storage.offsets)))
                return TileBuildStatus::invalid_storage;
        }
        return TileBuildStatus::success;
    }

    __device__ void scan_counts(Counter* counters, Offset* offsets, Index b, Index n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        const auto cluster = cooperative_groups::this_cluster();
        const int lane = threadIdx.x, rank = cluster.block_rank(), blocks = cluster.num_blocks();
        const int thread = rank * BlockThreads + lane;
        const Index begin = detail::partition_begin(b, rank, blocks), end = detail::partition_begin(b, rank + 1, blocks);
        const Offset carry = detail::scan_chunks<BlockThreads>(counters, offsets, begin, end, temp_.block.scan);
        if (lane == 0) temp_.total = carry;
        cluster.sync();
        if (lane == 0) {
            Offset base = 0;
            for (int preceding = 0; preceding < rank; ++preceding)
                base += *cluster.map_shared_rank(&temp_.total, preceding);
            temp_.base = base;
        }
        __syncthreads();
        if (Index(lane) < end - begin)
            for (Index bucket = begin + Index(lane); bucket < end;) {
                offsets[bucket] += temp_.base;
                if (!detail::advance_index(bucket, end, Index(BlockThreads))) break;
            }
        if (thread == 0) offsets[b] = static_cast<Offset>(n);
        cluster.sync();
#endif
    }
    TempStorage& temp_;
};

} // namespace cuco::detail::bulk_tiles
