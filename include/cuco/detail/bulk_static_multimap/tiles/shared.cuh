/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuda/std/limits>
#include "block.cuh"
#include <type_traits>

namespace cuco::detail::bulk_tiles {

// Global retained CSR; caller-owned shared counters/offsets and optional output staging.
template <int BlockThreads, typename Record = KeyRecord, typename Hash = cuco::default_hash_function<RecordKey<Record>>,
          typename Offset = int32_t, typename Counter = Offset, auto StaticBuckets = 0, typename Bucket = uint32_t>
class SharedBlockHashGraph {
    using Builder = BlockHashGraph<BlockThreads, Record, Hash, Offset, Counter, StaticBuckets, Bucket>;
    typename Builder::TempStorage& temp_;
    using Index = detail::index_type<Offset>;

    struct Layout {
        size_t counters = 0, offsets = 0, cache = 0, bytes = 0;
    };
    static constexpr size_t count_alignment = alignof(Counter) > alignof(Offset) ? alignof(Counter) : alignof(Offset);
    static constexpr size_t metadata_alignment = count_alignment > alignof(Bucket) ? count_alignment : alignof(Bucket);
    template <typename Policy>
    __host__ __device__ static Layout layout(int64_t records, int64_t buckets, bool stage_records, Policy policy) {
        Layout result;
        if constexpr (StaticBuckets)
            if (buckets != StaticBuckets) return result;
        const int64_t b = StaticBuckets ? StaticBuckets : buckets;
        if (!detail::valid_sizes<Offset, Counter, Bucket>(records, buckets) ||
            uint64_t(records) > cuda::std::numeric_limits<size_t>::max() / sizeof(Record) ||
            (Policy::shared_pad_period && !stage_records)) return result;
        size_t bytes = 0, start;
        if (stage_records) {
            const int64_t slots = policy.staged_record_count(records);
            if (slots < 0 || !detail::reserve_region(bytes, size_t(slots), sizeof(Record), alignof(Record), start)) return {};
        }
        if (!detail::reserve_region(bytes, size_t(b), sizeof(Counter), alignof(Counter), result.counters) ||
            !detail::reserve_region(bytes, size_t(b) + 1, sizeof(Offset), alignof(Offset), result.offsets)) return {};
        if constexpr (Policy::shared_buckets)
            if (records > int64_t(Policy::register_items) * BlockThreads &&
                !detail::reserve_region(bytes, size_t(records), sizeof(Bucket), alignof(Bucket), result.cache)) return {};
        result.bytes = bytes;
        return result;
    }

    template <typename Storage, typename Observer, int RegisterItems, bool SharedBuckets, int PrefetchItems, int SharedPadPeriod>
    __device__ TileBuildStatus run(Storage storage, const TileBuildStorage<Record, Offset, Counter, Bucket>& retained,
                                   void* workspace, size_t capacity, bool stage_records,
                                   Hash hash, Observer observer,
                                   TileBuildPolicy<RegisterItems, SharedBuckets, PrefetchItems, SharedPadPeriod, Bucket> policy) {
        if constexpr (StaticBuckets)
            if (retained.bucket_count != StaticBuckets) return TileBuildStatus::invalid_size;
        const auto status = storage.validate();
        if (status != TileBuildStatus::success) return status;
        if (blockDim.x != BlockThreads || blockDim.y != 1 || blockDim.z != 1)
            return TileBuildStatus::invalid_block;
        if (!__isGlobal(retained.offsets) || (retained.record_count &&
            (!__isGlobal(retained.input) || !__isGlobal(retained.records))))
            return TileBuildStatus::invalid_storage;
        const size_t alignment = stage_records && alignof(Record) > metadata_alignment ? alignof(Record) : metadata_alignment;
        if (!workspace || !__isShared(workspace) || reinterpret_cast<uintptr_t>(workspace) % alignment)
            return TileBuildStatus::invalid_storage;
        if constexpr (SharedPadPeriod)
            if (!stage_records) return TileBuildStatus::invalid_storage;
        const auto sizes = layout(retained.record_count, retained.bucket_count, stage_records, policy);
        if (!sizes.bytes || capacity < sizes.bytes) return TileBuildStatus::insufficient_capacity;
        auto local = retained;
        auto* bytes = static_cast<unsigned char*>(workspace);
        if (stage_records && retained.record_count) {
            local.records = reinterpret_cast<Record*>(bytes);
            const int64_t slots = policy.staged_record_count(retained.record_count);
            if constexpr (SharedPadPeriod) local.record_capacity = slots;
        }
        local.counters = reinterpret_cast<Counter*>(bytes + sizes.counters);
        local.offsets = reinterpret_cast<Offset*>(bytes + sizes.offsets);
        if constexpr (SharedBuckets) {
            policy.bucket_cache = reinterpret_cast<Bucket*>(bytes + sizes.cache);
            policy.bucket_capacity = retained.record_count;
        }
        TileBuildStatus result;
        constexpr bool lrb = !std::is_same<Storage, TileBuildStorage<Record, Offset, Counter, Bucket>>::value;
        if constexpr (lrb) {
            storage.table = local;
            result = Builder(temp_).build(storage, hash, observer, policy);
        } else result = Builder(temp_).build(local, hash, observer, policy);
        if (result != TileBuildStatus::success) return result;
        const Index buckets = Index(StaticBuckets ? StaticBuckets : retained.bucket_count);
        for (Index b = threadIdx.x; b < buckets;) {
            retained.offsets[b] = local.offsets[b];
            if (!detail::advance_index(b, buckets, Index(BlockThreads))) break;
        }
        if (!threadIdx.x) retained.offsets[buckets] = local.offsets[buckets];
        if (stage_records) {
            const Index count = Index(retained.record_count);
            for (Index i = threadIdx.x; i < count;) {
                retained.records[i] = detail::load_record<SharedPadPeriod>(local.records, i);
                if (!detail::advance_index(i, count, Index(BlockThreads))) break;
            }
        }
        __syncthreads();
        observer.mark(lrb ? 8 : 5);
        return result;
    }

public:
    using TempStorage = typename Builder::TempStorage;
    __device__ explicit SharedBlockHashGraph(TempStorage& temp) : temp_(temp) {}

    template <int RegisterItems = 0, bool SharedBuckets = false, int PrefetchItems = 0, int SharedPadPeriod = 0>
    __host__ __device__ static size_t workspace_bytes(int64_t records, int64_t buckets, bool stage_records,
                    TileBuildPolicy<RegisterItems, SharedBuckets, PrefetchItems, SharedPadPeriod, Bucket> policy = {}) {
        return layout(records, buckets, stage_records, policy).bytes;
    }

    template <typename Observer = TileBuildNoObserver, int RegisterItems = 0, bool SharedBuckets = false, int PrefetchItems = 0, int SharedPadPeriod = 0>
    __device__ TileBuildStatus build(const TileBuildStorage<Record, Offset, Counter, Bucket>& storage, void* workspace,
                                     size_t capacity, bool stage_records = false,
                                     Hash hash = {}, Observer observer = {},
                                     TileBuildPolicy<RegisterItems, SharedBuckets, PrefetchItems, SharedPadPeriod, Bucket> policy = {}) {
        return run(storage, storage, workspace, capacity, stage_records, hash, observer, policy);
    }

    template <typename Observer = TileBuildNoObserver, int RegisterItems = 0, bool SharedBuckets = false, int PrefetchItems = 0, int SharedPadPeriod = 0>
    __device__ TileBuildStatus build(const TileLrbBuildStorage<Record, Offset, Counter, Bucket>& storage, void* workspace,
                                     size_t capacity, bool stage_records = false,
                                     Hash hash = {}, Observer observer = {},
                                     TileBuildPolicy<RegisterItems, SharedBuckets, PrefetchItems, SharedPadPeriod, Bucket> policy = {}) {
        return run(storage, storage.table, workspace, capacity, stage_records, hash, observer, policy);
    }
};

} // namespace cuco::detail::bulk_tiles
