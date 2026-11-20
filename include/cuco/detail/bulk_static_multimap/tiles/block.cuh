/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuda/std/limits>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cuco/bulk_static_multimap_builder.cuh>
#include <cuco/hash_functions.cuh>

namespace cuco::detail::bulk_tiles {

using KeyRecord = cuco::pair<int64_t, int64_t>;
using DefaultHash = cuco::default_hash_function<int64_t>;
template <typename Record>
using RecordKey = typename Record::first_type;
using TileBuildStatus = cuco::bulk_build_status;

struct TileBuildNoObserver {
    __device__ void mark(int) const {}
};

namespace detail {
template <typename T>
inline constexpr bool storage_integer = std::is_integral_v<T> &&
    (sizeof(T) == 4 || sizeof(T) == 8) && !std::is_same_v<T, bool>;

template <typename Offset, typename Counter, typename Bucket = uint32_t>
__host__ __device__ constexpr bool valid_sizes(int64_t records, int64_t buckets) {
    static_assert(storage_integer<Offset> && storage_integer<Counter>, "Offsets and counters must be 32- or 64-bit integers");
    static_assert(storage_integer<Bucket> && std::is_unsigned_v<Bucket>, "Bucket indices must be unsigned 32- or 64-bit integers");
    return records >= 0 && buckets > 0 &&
        uint64_t(records) <= uint64_t(cuda::std::numeric_limits<Offset>::max()) &&
        uint64_t(records) <= uint64_t(cuda::std::numeric_limits<Counter>::max()) &&
        uint64_t(buckets) <= uint64_t(cuda::std::numeric_limits<Offset>::max()) &&
        uint64_t(buckets) <= uint64_t(cuda::std::numeric_limits<Bucket>::max());
}

template <typename Offset>
using index_type = std::make_unsigned_t<Offset>;

template <typename Bucket, typename HashValue>
__host__ __device__ __forceinline__ Bucket reduce_hash(HashValue hash, Bucket buckets) {
    static_assert(std::is_integral_v<HashValue> && !std::is_same_v<HashValue, bool>);
    using HashBits = std::make_unsigned_t<HashValue>;
    using Reduction = std::common_type_t<HashBits, Bucket>;
    return static_cast<Bucket>(Reduction(static_cast<HashBits>(hash)) % Reduction(buckets));
}

// The final increment is skipped so full-range unsigned indices cannot wrap.
template <typename Index>
__host__ __device__ __forceinline__ bool advance_index(Index& index, Index end, Index stride) {
    if (end - index <= stride) return false;
    index += stride;
    return true;
}

template <typename Index>
__host__ __device__ __forceinline__ Index bounded_index(uint64_t value, Index end) {
    return value < uint64_t(end) ? Index(value) : end;
}

template <typename Index>
__host__ __device__ __forceinline__ Index partition_begin(Index size, unsigned rank, unsigned groups) {
    return (size / groups) * rank + ((size % groups) * rank) / groups;
}

template <auto Buckets, typename Offset, typename Bucket = uint32_t>
inline constexpr bool valid_static_buckets = Buckets == 0 ||
    (!(Buckets & (Buckets - 1)) && uint64_t(Buckets) <= uint64_t(cuda::std::numeric_limits<Offset>::max()) &&
     uint64_t(Buckets) <= uint64_t(cuda::std::numeric_limits<Bucket>::max()));

template <typename T>
__device__ __forceinline__ T atomic_add(T* address, T value = T{1}) {
    static_assert(storage_integer<T>);
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, unsigned int>)
        return atomicAdd(address, value);
    else
        return cuda::atomic_ref<T, cuda::thread_scope_device>(*address).fetch_add(value, cuda::memory_order_relaxed);
}

__host__ __device__ constexpr size_t align_bytes(size_t bytes, size_t alignment) {
    return (bytes + alignment - 1) / alignment * alignment;
}

__host__ __device__ inline bool reserve_region(size_t& cursor, uint64_t count, size_t width,
                                               size_t alignment, size_t& start) {
    constexpr size_t limit = cuda::std::numeric_limits<size_t>::max();
    if (!width || !alignment || cursor > limit - (alignment - 1)) return false;
    start = align_bytes(cursor, alignment);
    if (count > (limit - start) / width) return false;
    cursor = start + size_t(count) * width;
    return true;
}

template <int Threads, typename Offset, typename Counter, typename Index>
__device__ __forceinline__ Offset scan_chunks(Counter* counters, Offset* offsets, Index begin, Index end,
                                              typename cub::BlockScan<Offset, Threads>::TempStorage& temp) {
    using Scan = cub::BlockScan<Offset, Threads>;
    const Index lane = threadIdx.x;
    Offset carry = 0;
    for (Index base = begin; base < end;) {
        const bool present = lane < end - base;
        const Index bucket = present ? base + lane : base;
        const Offset count = present ? Offset(counters[bucket]) : Offset{0};
        Offset prefix, total;
        Scan(temp).ExclusiveSum(count, prefix, total);
        if (present) {
            offsets[bucket] = carry + prefix;
            counters[bucket] = 0;
        }
        carry += total;
        // Every lane finishes before the next chunk reuses CUB temporary storage.
        __syncthreads();
        if (!advance_index(base, end, Index(Threads))) break;
    }
    return carry;
}
} // namespace detail

// Register slots are unrolled; overflow uses caller-owned shared bucket IDs.
template <int RegisterItems = 0, bool SharedBuckets = false, int PrefetchItems = 0, int SharedPadPeriod = 0, typename Bucket = uint32_t>
struct TileBuildPolicy {
    static_assert(RegisterItems >= 0 && PrefetchItems >= 0 && SharedPadPeriod >= 0);
    static_assert(detail::storage_integer<Bucket> && std::is_unsigned_v<Bucket>);
    static constexpr int register_items = RegisterItems;
    static constexpr bool shared_buckets = SharedBuckets;
    static constexpr int prefetch_items = PrefetchItems;
    static constexpr int shared_pad_period = SharedPadPeriod;
    __host__ __device__ static int64_t staged_record_count(int64_t records) {
        if constexpr (SharedPadPeriod) {
            if (records <= 0) return records;
            const int64_t padding = (records - 1) / SharedPadPeriod;
            return records > cuda::std::numeric_limits<int64_t>::max() - padding ? -1 : records + padding;
        }
        else return records;
    }
    using bucket_type = Bucket;
    Bucket* bucket_cache = nullptr;
    int64_t bucket_capacity = 0;
    __host__ __device__ static size_t cache_bytes(int64_t records, int threads) {
        if (!SharedBuckets || records <= int64_t(RegisterItems) * threads ||
            uint64_t(records) > cuda::std::numeric_limits<size_t>::max() / sizeof(Bucket)) return 0;
        return size_t(records) * sizeof(Bucket);
    }
};

template <typename Record = KeyRecord, typename Hash = cuco::default_hash_function<RecordKey<Record>>, typename Offset = int32_t, typename Bucket = uint32_t>
struct TileCsrView {
    static_assert(detail::storage_integer<Offset>);
    using key_type = RecordKey<Record>;
    using offset_type = Offset;
    using bucket_type = Bucket;
    static_assert(detail::storage_integer<Bucket> && std::is_unsigned_v<Bucket>);
    const Offset* offsets;
    const Record* records;
    Offset buckets;
    Hash hash{};
    Offset size = 0;
};

template <typename Record = KeyRecord, typename Offset = int32_t, typename Counter = Offset, typename Bucket = uint32_t>
struct TileBuildStorage {
    static_assert(std::is_trivially_copyable_v<Record>, "Record must be trivially copyable");
    static_assert(detail::storage_integer<Offset> && detail::storage_integer<Counter>);
    using key_type = RecordKey<Record>;
    using offset_type = Offset;
    using counter_type = Counter;
    using bucket_type = Bucket;
    using index_type = detail::index_type<Offset>;
    const Record* input = nullptr;
    int64_t record_count = 0;
    int64_t bucket_count = 0;
    Record* records = nullptr;
    int64_t record_capacity = 0;
    Offset* offsets = nullptr;
    int64_t offset_capacity = 0;
    Counter* counters = nullptr;
    int64_t counter_capacity = 0;

    __host__ __device__ TileBuildStatus validate() const {
        if (!detail::valid_sizes<Offset, Counter, Bucket>(record_count, bucket_count) ||
            uint64_t(record_count) > cuda::std::numeric_limits<size_t>::max() / sizeof(Record) ||
            uint64_t(bucket_count) >= cuda::std::numeric_limits<size_t>::max() / sizeof(Offset) ||
            uint64_t(bucket_count) > cuda::std::numeric_limits<size_t>::max() / sizeof(Counter))
            return TileBuildStatus::invalid_size;
        if (record_capacity < record_count || offset_capacity <= bucket_count ||
            counter_capacity < bucket_count)
            return TileBuildStatus::insufficient_capacity;
        if (!offsets || !counters || static_cast<const void*>(offsets) == static_cast<const void*>(counters) ||
            (record_count && (!input || !records || input == records)))
            return TileBuildStatus::invalid_storage;
        return TileBuildStatus::success;
    }

    template <typename Hash = cuco::default_hash_function<RecordKey<Record>>>
    __host__ __device__ TileCsrView<Record, Hash, Offset, Bucket> view(Hash hash = {}) const {
        return {offsets, records, static_cast<Offset>(bucket_count), hash, static_cast<Offset>(record_count)};
    }
};

// Build-only scratch; the returned CSR owns no reference to these arrays.
template <typename Record = KeyRecord, typename Offset = int32_t, typename Counter = Offset, typename Bucket = uint32_t>
struct TileLrbBuildStorage {
    TileBuildStorage<Record, Offset, Counter, Bucket> table;
    Record* reordered = nullptr;
    int64_t reorder_capacity = 0;
    using bucket_type = Bucket;
    using index_type = detail::index_type<Offset>;
    Bucket* bucket_hashes = nullptr;
    int64_t hash_capacity = 0;
    Counter* bin_counters = nullptr;
    int64_t bin_counter_capacity = 0;
    Offset* bin_offsets = nullptr;
    int64_t bin_offset_capacity = 0;
    int64_t bucket_span = 2048;

    __host__ __device__ int64_t bin_count() const {
        return 1 + (table.bucket_count - 1) / bucket_span;
    }
    __host__ __device__ TileBuildStatus validate() const {
        const auto status = table.validate();
        if (status != TileBuildStatus::success) return status;
        if (bucket_span <= 0 || uint64_t(bucket_span) > uint64_t(cuda::std::numeric_limits<Bucket>::max()) ||
            uint64_t(bucket_span) > uint64_t(cuda::std::numeric_limits<Offset>::max()))
            return TileBuildStatus::invalid_size;
        if (reorder_capacity < table.record_count || hash_capacity < table.record_count ||
            bin_counter_capacity < bin_count() || bin_offset_capacity <= bin_count())
            return TileBuildStatus::insufficient_capacity;
        if (!bin_counters || !bin_offsets || (table.record_count && (!reordered || !bucket_hashes)))
            return TileBuildStatus::invalid_storage;
        const void* arrays[]{table.input, table.records, table.offsets, table.counters,
                             reordered, bucket_hashes, bin_counters, bin_offsets};
        #if defined(__CUDA_ARCH__)
        #pragma unroll
        #endif
        for (int i = 0; i < 8; ++i)
            #if defined(__CUDA_ARCH__)
        #pragma unroll
        #endif
            for (int j = 0; j < i; ++j)
                if (arrays[i] && arrays[i] == arrays[j]) return TileBuildStatus::invalid_storage;
        return TileBuildStatus::success;
    }
};

namespace detail {
template <int SharedPadPeriod = 0, typename Record, typename Index>
__device__ void store_record(Record* records, Index position, const Record& record) {
    if constexpr (SharedPadPeriod) records[size_t(position) + size_t(position) / SharedPadPeriod] = record;
    else records[position] = record;
}

template <int SharedPadPeriod = 0, typename Record, typename Index>
__device__ Record load_record(const Record* records, Index position) {
    if constexpr (SharedPadPeriod) return records[size_t(position) + size_t(position) / SharedPadPeriod];
    else return records[position];
}

template <typename Observer>
struct LrbBucketObserver {
    Observer observer;
    __device__ void mark(int phase) const {
        if (phase) observer.mark(phase + 3);
    }
};
} // namespace detail

// Collective over a complete 1D block. All arrays belong to the caller;
// input, output and scratch must be disjoint and descriptors block-uniform.
template <int BlockThreads, typename Record = KeyRecord, typename Hash = cuco::default_hash_function<RecordKey<Record>>,
          typename Offset = int32_t, typename Counter = Offset, auto StaticBuckets = 0, typename Bucket = uint32_t>
class BlockHashGraph {
    static_assert(BlockThreads > 0 && BlockThreads <= 1024);
    static_assert(detail::valid_static_buckets<StaticBuckets, Offset, Bucket>, "Static buckets must be zero or a representable power of two");
    using Scan = cub::BlockScan<Offset, BlockThreads>;
    using Index = detail::index_type<Offset>;

    template <typename Key>
    __device__ __forceinline__ static Bucket bucket_for(Key key, Bucket runtime_buckets, Hash hash) {
        if constexpr (StaticBuckets) return detail::reduce_hash<Bucket>(hash(key), Bucket(StaticBuckets));
        else return detail::reduce_hash<Bucket>(hash(key), runtime_buckets);
    }

    template <int Window, bool Registers, int CacheSize, typename Policy>
    __device__ __forceinline__ void count_prefetched(const TileBuildStorage<Record, Offset, Counter, Bucket>& storage, Index n,
                    Bucket buckets, Index base, Bucket (&cached)[CacheSize], int first, Hash hash, Policy policy) {
        const auto* input = storage.input;
        auto* counters = storage.counters;
        RecordKey<Record> keys[Window];
        #pragma unroll
        for (int item = 0; item < Window; ++item) {
            const Index delta = Index(item) * BlockThreads;
            const bool present = base < n && delta < n - base;
            const Index i = present ? base + delta : base;
            if (present && (!Registers || first + item < CacheSize)) keys[item] = input[i].first;
        }
        // Keep batch loads ahead of dependent hash/count operations.
        asm volatile("" ::: "memory");
        #pragma unroll
        for (int item = 0; item < Window; ++item) {
            const Index delta = Index(item) * BlockThreads;
            const bool present = base < n && delta < n - base;
            const Index i = present ? base + delta : base;
            if (present && (!Registers || first + item < CacheSize)) {
                const Bucket bucket = bucket_for(keys[item], buckets, hash);
                if constexpr (Registers) cached[first + item] = bucket;
                else if constexpr (Policy::shared_buckets) policy.bucket_cache[i] = bucket;
                detail::atomic_add(counters + bucket);
            }
        }
    }

    template <int Window, bool Registers, int CacheSize, typename Policy>
    __device__ __forceinline__ void scatter_prefetched(const TileBuildStorage<Record, Offset, Counter, Bucket>& storage, Index n,
                    Bucket buckets, Index base, const Bucket (&cached)[CacheSize], int first, Hash hash, Policy policy) {
        const auto* input = storage.input;
        auto* output = storage.records;
        auto* counters = storage.counters;
        const auto* offsets = storage.offsets;
        Record records[Window];
        Bucket ids[Window];
        #pragma unroll
        for (int item = 0; item < Window; ++item) {
            const Index delta = Index(item) * BlockThreads;
            const bool present = base < n && delta < n - base;
            const Index i = present ? base + delta : base;
            if (present && (!Registers || first + item < CacheSize)) {
                records[item] = input[i];
                if constexpr (Registers) ids[item] = cached[first + item];
                else if constexpr (Policy::shared_buckets) ids[item] = policy.bucket_cache[i];
            }
        }
        asm volatile("" ::: "memory");
        #pragma unroll
        for (int item = 0; item < Window; ++item) {
            const Index delta = Index(item) * BlockThreads;
            const bool present = base < n && delta < n - base;
            const Index i = present ? base + delta : base;
            if (present && (!Registers || first + item < CacheSize)) {
                Bucket bucket;
                if constexpr (Registers || Policy::shared_buckets) bucket = ids[item];
                else bucket = bucket_for(records[item].first, buckets, hash);
                const Offset position = offsets[bucket] + Offset(detail::atomic_add(counters + bucket));
                detail::store_record<Policy::shared_pad_period>(output, position, records[item]);
            }
        }
    }

public:
    struct TempStorage {
        typename Scan::TempStorage scan;
    };

    __device__ explicit BlockHashGraph(TempStorage& temp) : temp_(temp) {}

    template <typename Observer = TileBuildNoObserver, int RegisterItems = 0, bool SharedBuckets = false, int PrefetchItems = 0, int SharedPadPeriod = 0>
    __device__ TileBuildStatus build(const TileBuildStorage<Record, Offset, Counter, Bucket>& storage,
                                     Hash hash = {}, Observer observer = {},
                                     TileBuildPolicy<RegisterItems, SharedBuckets, PrefetchItems, SharedPadPeriod, Bucket> policy = {}) {
        if constexpr (sizeof(Index) < sizeof(uint64_t) && RegisterItems > 0)
            static_assert(uint64_t(RegisterItems) * BlockThreads <= cuda::std::numeric_limits<Index>::max());
        if constexpr (sizeof(Index) < sizeof(uint64_t) && PrefetchItems > 0)
            static_assert(uint64_t(PrefetchItems) * BlockThreads <= cuda::std::numeric_limits<Index>::max());
        if (blockDim.x != BlockThreads || blockDim.y != 1 || blockDim.z != 1)
            return TileBuildStatus::invalid_block;
        if constexpr (StaticBuckets)
            if (storage.bucket_count != StaticBuckets) return TileBuildStatus::invalid_size;
        const auto status = storage.validate();
        if (status != TileBuildStatus::success) return status;

        const int lane = threadIdx.x;
        const Index n = Index(storage.record_count);
        const Index b = Index(StaticBuckets ? StaticBuckets : storage.bucket_count);
        const Bucket buckets = static_cast<Bucket>(b);
        const bool in_registers = RegisterItems && n <= Index(RegisterItems) * BlockThreads;
        if constexpr (RegisterItems && !SharedBuckets)
            if (!in_registers) return TileBuildStatus::insufficient_capacity;
        if constexpr (SharedBuckets)
            if (!in_registers && n) {
                if (policy.bucket_capacity < n) return TileBuildStatus::insufficient_capacity;
                if (!policy.bucket_cache || !__isShared(policy.bucket_cache) ||
                    reinterpret_cast<uintptr_t>(policy.bucket_cache) % alignof(Bucket) ||
                    policy.bucket_cache == reinterpret_cast<const void*>(storage.input) ||
                    policy.bucket_cache == reinterpret_cast<void*>(storage.records) ||
                    policy.bucket_cache == reinterpret_cast<void*>(storage.counters) ||
                    policy.bucket_cache == reinterpret_cast<void*>(storage.offsets))
                    return TileBuildStatus::invalid_storage;
            }
        if constexpr (SharedPadPeriod) {
            if (n && !__isShared(storage.records)) return TileBuildStatus::invalid_storage;
            if (policy.staged_record_count(n) < 0 || storage.record_capacity < policy.staged_record_count(n)) return TileBuildStatus::insufficient_capacity;
        }
        Bucket cached[RegisterItems ? RegisterItems : 1];
        observer.mark(0);
        for (Index i = lane; i < b;) {
            storage.counters[i] = 0;
            if (!detail::advance_index(i, b, Index(BlockThreads))) break;
        }
        __syncthreads();
        observer.mark(1);
        if constexpr (PrefetchItems) {
            if constexpr (RegisterItems) {
                if (in_registers) {
                    #pragma unroll
                    for (int first = 0; first < RegisterItems; first += PrefetchItems)
                        count_prefetched<PrefetchItems, true>(storage, n, buckets,
                            Index(lane) + Index(first) * BlockThreads, cached, first, hash, policy);
                }
            }
            if (!in_registers)
                for (Index base = lane; base < n;) {
                    count_prefetched<PrefetchItems, false>(storage, n, buckets, base, cached, 0, hash, policy);
                    if (!detail::advance_index(base, n, Index(PrefetchItems) * BlockThreads)) break;
                }
        } else {
            if constexpr (RegisterItems) {
                if (in_registers) {
                    #pragma unroll
                    for (int item = 0; item < RegisterItems; ++item) {
                        const Index i = Index(lane) + Index(item) * BlockThreads;
                        if (i < n) {
                            cached[item] = bucket_for(storage.input[i].first, buckets, hash);
                            detail::atomic_add(storage.counters + cached[item]);
                        }
                    }
                }
            }
            if (!in_registers)
                for (Index i = lane; i < n;) {
                    const Bucket bucket = bucket_for(storage.input[i].first, buckets, hash);
                    if constexpr (SharedBuckets) policy.bucket_cache[i] = bucket;
                    detail::atomic_add(storage.counters + bucket);
                    if (!detail::advance_index(i, n, Index(BlockThreads))) break;
                }
        }
        __syncthreads();

        observer.mark(2);

        scan_counts<true>(storage.counters, storage.offsets, b, n);
        observer.mark(3);

        if constexpr (PrefetchItems) {
            if constexpr (RegisterItems) {
                if (in_registers) {
                    #pragma unroll
                    for (int first = 0; first < RegisterItems; first += PrefetchItems)
                        scatter_prefetched<PrefetchItems, true>(storage, n, buckets,
                            Index(lane) + Index(first) * BlockThreads, cached, first, hash, policy);
                }
            }
            if (!in_registers)
                for (Index base = lane; base < n;) {
                    scatter_prefetched<PrefetchItems, false>(storage, n, buckets, base, cached, 0, hash, policy);
                    if (!detail::advance_index(base, n, Index(PrefetchItems) * BlockThreads)) break;
                }
        } else {
            if constexpr (RegisterItems) {
                if (in_registers) {
                    #pragma unroll
                    for (int item = 0; item < RegisterItems; ++item) {
                        const Index i = Index(lane) + Index(item) * BlockThreads;
                        if (i < n) {
                            const Bucket bucket = cached[item];
                            const Offset position = storage.offsets[bucket] + Offset(detail::atomic_add(storage.counters + bucket));
                            detail::store_record<SharedPadPeriod>(storage.records, position, storage.input[i]);
                        }
                    }
                }
            }
            if (!in_registers)
                for (Index i = lane; i < n;) {
                    const auto record = storage.input[i];
                    Bucket bucket;
                    if constexpr (SharedBuckets) bucket = policy.bucket_cache[i];
                    else bucket = bucket_for(record.first, buckets, hash);
                    const Offset position = storage.offsets[bucket] + Offset(detail::atomic_add(storage.counters + bucket));
                    detail::store_record<SharedPadPeriod>(storage.records, position, record);
                    if (!detail::advance_index(i, n, Index(BlockThreads))) break;
                }
        }
        __syncthreads();
        observer.mark(4);
        return TileBuildStatus::success;
    }

    template <typename Observer = TileBuildNoObserver, int RegisterItems = 0, bool SharedBuckets = false, int PrefetchItems = 0, int SharedPadPeriod = 0>
    __device__ TileBuildStatus build(const TileLrbBuildStorage<Record, Offset, Counter, Bucket>& storage,
                                     Hash hash = {}, Observer observer = {},
                                     TileBuildPolicy<RegisterItems, SharedBuckets, PrefetchItems, SharedPadPeriod, Bucket> policy = {}) {
        if constexpr (sizeof(Index) < sizeof(uint64_t) && RegisterItems > 0)
            static_assert(uint64_t(RegisterItems) * BlockThreads <= cuda::std::numeric_limits<Index>::max());
        if constexpr (sizeof(Index) < sizeof(uint64_t) && PrefetchItems > 0)
            static_assert(uint64_t(PrefetchItems) * BlockThreads <= cuda::std::numeric_limits<Index>::max());
        if (blockDim.x != BlockThreads || blockDim.y != 1 || blockDim.z != 1)
            return TileBuildStatus::invalid_block;
        if constexpr (StaticBuckets)
            if (storage.table.bucket_count != StaticBuckets) return TileBuildStatus::invalid_size;
        const auto status = storage.validate();
        if (status != TileBuildStatus::success) return status;
        const auto& table = storage.table;
        const int lane = threadIdx.x;
        const Index bins = Index(StaticBuckets ? 1 + (int64_t(StaticBuckets) - 1) / storage.bucket_span : storage.bin_count());
        const Index n = Index(table.record_count);
        observer.mark(0);
        for (Index i = lane; i < bins;) {
            storage.bin_counters[i] = 0;
            if (!detail::advance_index(i, bins, Index(BlockThreads))) break;
        }
        __syncthreads();
        observer.mark(1);
        for (Index i = lane; i < n;) {
            const Bucket bucket = bucket_for(table.input[i].first, Bucket(table.bucket_count), hash);
            storage.bucket_hashes[i] = bucket;
            detail::atomic_add(storage.bin_counters + bucket / Bucket(storage.bucket_span));
            if (!detail::advance_index(i, n, Index(BlockThreads))) break;
        }
        __syncthreads();
        observer.mark(2);
        scan_counts(storage.bin_counters, storage.bin_offsets, bins, n);
        observer.mark(3);
        for (Index i = lane; i < n;) {
            const Bucket bin = storage.bucket_hashes[i] / Bucket(storage.bucket_span);
            const Offset position = storage.bin_offsets[bin] + Offset(detail::atomic_add(storage.bin_counters + bin));
            storage.reordered[position] = table.input[i];
            if (!detail::advance_index(i, n, Index(BlockThreads))) break;
        }
        __syncthreads();
        auto reordered = table;
        reordered.input = storage.reordered;
        return build(reordered, hash, detail::LrbBucketObserver<Observer>{observer}, policy);
    }

private:
    template <bool TableBuckets = false>
    __device__ void scan_counts(Counter* counters, Offset* offsets, Index runtime_buckets, Index n) {
        const Index b = TableBuckets && StaticBuckets ? Index(StaticBuckets) : runtime_buckets;
        const int lane = threadIdx.x;
        detail::scan_chunks<BlockThreads>(counters, offsets, Index{0}, b, temp_.scan);
        if (lane == 0) offsets[b] = static_cast<Offset>(n);
        __syncthreads();
    }
    TempStorage& temp_;
};

} // namespace cuco::detail::bulk_tiles
