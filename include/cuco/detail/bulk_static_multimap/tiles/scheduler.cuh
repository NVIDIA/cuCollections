/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "batch_cluster.cuh"
#include "subgroup.cuh"
#include <algorithm>
#include <array>
#include <cuda/std/limits>
#include <cub/device/device_radix_sort.cuh>

namespace cuco::detail::bulk_tiles {
enum class TableScheduleKind { block, cluster, cooperative };

struct TableScheduleRange {
    int64_t begin = 0, end = 0;
};

struct TableScheduleClass {
    int threads = 0, blocks = 1, memory = 0, resident_groups = 0;
    int64_t work_limit = 0;
    size_t shared_bytes = 0;
    uint32_t static_buckets = 0;
    TableScheduleKind kind = TableScheduleKind::block;
    int lrb_resident_groups = 0;
};

template <typename Bucket> struct TypedTableScheduleOptions {
    bool allow_clusters = true;
    size_t shared_budget = 0;
    TypedBatchBucketCaches<Bucket> bucket_caches {};
};

using TableScheduleOptions = TypedTableScheduleOptions<uint32_t>;

namespace schedule_detail {
constexpr int tiny_classes = 18, static_classes = 33, block_classes = 18, cluster_classes = 6;
constexpr int static_begin = tiny_classes, block_begin = static_begin + static_classes;
constexpr int cluster_begin = block_begin + block_classes, fallback_class = cluster_begin + cluster_classes,
              class_count = fallback_class + 1;
constexpr int range_count = 2 * class_count;
constexpr int priority_bits = 6, sort_bits = 14;

template <typename B> struct is_uniform : std::false_type { };

template <typename S> struct is_uniform<UniformTableBatch<S>> : std::true_type { };

template <typename S> using is_lrb = batch_detail::is_lrb_storage<S>;

struct Classes {
    TableScheduleClass entries[class_count] {};
};

struct Workspace {
    size_t bytes = 0, keys_in = 0, keys_out = 0, ids_in = 0, ids_out = 0, ranges = 0, sort = 0;
};

inline bool workspace(int64_t count, size_t sort_bytes, Workspace& result) {
    if (count < 0)
        return false;
    size_t cursor = 0;
    if (!detail::reserve_region(cursor, count, sizeof(uint32_t), alignof(uint32_t), result.keys_in)
        || !detail::reserve_region(cursor, count, sizeof(uint32_t), alignof(uint32_t), result.keys_out)
        || !detail::reserve_region(cursor, count, sizeof(int64_t), alignof(int64_t), result.ids_in)
        || !detail::reserve_region(cursor, count, sizeof(int64_t), alignof(int64_t), result.ids_out)
        || !detail::reserve_region(
            cursor, range_count + 1, sizeof(TableScheduleRange), alignof(TableScheduleRange), result.ranges)
        || !detail::reserve_region(cursor, sort_bytes, 1, 256, result.sort))
        return false;
    result.bytes = cursor;
    return true;
}

template <typename T> T* pointer(void* workspace, size_t offset) {
    return reinterpret_cast<T*>(static_cast<unsigned char*>(workspace) + offset);
}

template <typename B> __host__ bool basic_valid(const B& batch) {
    return batch.valid();
}

template <typename S> __host__ bool basic_valid(const UniformTableBatch<S>& batch) {
    return batch.table_count >= 0
        && (!batch.table_count || batch_detail::uniform_fits(batch.first, batch.table_count));
}

template <typename B> struct IndexedBatch {
    using storage_type = typename B::storage_type;
    B source;
    const int64_t* ids;
    int64_t table_count;

    __device__ storage_type operator[](int64_t i) const {
        return source[ids[i]];
    }

    __device__ int64_t output_index(int64_t i) const {
        return ids[i];
    }
};

template <typename B> struct GroupedBatch {
    using storage_type = typename B::storage_type;
    B source;
    const int64_t* ids;
    const TableScheduleRange* ranges;
    int group;
    int64_t table_count;

    __host__ bool valid() const {
        return basic_valid(source) && table_count == source.table_count && group >= 0 && group < range_count
            && (!table_count || (ids && ranges));
    }

    __device__ IndexedBatch<B> resolve() const {
        const auto range = ranges[group];
        return { source, ids + range.begin, range.end - range.begin };
    }
};

template <typename B> struct DirectResolved {
    using storage_type = std::remove_cv_t<
        std::remove_reference_t<decltype(batch_detail::table(std::declval<typename B::storage_type>()))>>;
    B source;
    int64_t table_count;

    __device__ storage_type operator[](int64_t i) const {
        return batch_detail::table(source[i]);
    }

    __device__ int64_t output_index(int64_t i) const {
        return batch_detail::output_index(source, i);
    }
};

template <typename B> struct DirectBatch {
    using storage_type = typename DirectResolved<B>::storage_type;
    B source;
    int64_t table_count;

    __host__ bool valid() const {
        return basic_valid(source) && table_count == source.table_count;
    }

    __device__ auto resolve() const {
        const auto resolved = batch_detail::resolve(source);
        return DirectResolved<decltype(resolved)> { resolved, resolved.table_count };
    }
};

template <typename B> __host__ auto as_direct(B source) {
    return DirectBatch<B> { source, source.table_count };
}

template <typename S> __host__ auto as_direct(UniformTableBatch<S> source) {
    using Direct = typename DirectResolved<UniformTableBatch<S>>::storage_type;
    return UniformTableBatch<Direct> { batch_detail::table(source.first), source.table_count };
}

template <typename Action> cudaError_t with_threads(int threads, Action action) {
    switch (threads) {
    case 32:
        return action(std::integral_constant<int, 32> {});
    case 64:
        return action(std::integral_constant<int, 64> {});
    case 128:
        return action(std::integral_constant<int, 128> {});
    case 256:
        return action(std::integral_constant<int, 256> {});
    case 512:
        return action(std::integral_constant<int, 512> {});
    case 1024:
        return action(std::integral_constant<int, 1024> {});
    default:
        return cudaErrorInvalidValue;
    }
}

template <int Maximum = 10, typename Action, int Exponent = 0>
cudaError_t with_bucket(int exponent, Action action) {
    if (exponent == Exponent)
        return action(std::integral_constant<uint32_t, uint32_t(1) << Exponent> {});
    if constexpr (Exponent < Maximum)
        return with_bucket<Maximum, Action, Exponent + 1>(exponent, action);
    else
        return cudaErrorInvalidValue;
}

__host__ __device__ inline int bucket_exponent(int64_t count) {
    if (count <= 0 || uint64_t(count) > cuda::std::numeric_limits<uint32_t>::max()
        || (uint64_t(count) & (uint64_t(count) - 1)))
        return -1;
    int result = 0;
    while (count > 1) {
        count >>= 1;
        ++result;
    }
    return result;
}

template <int Items, uint32_t Buckets> constexpr int static_threads() {
    int threads = 32;
    while (threads < 1024 && int64_t(Items) * threads < int64_t(8) * Buckets)
        threads *= 2;
    return threads;
}

template <typename S>
__host__ __device__ TileBuildStatus validate(const S& storage, const TableScheduleClass& selected) {
    const auto& table = batch_detail::table(storage);
    if (selected.kind == TableScheduleKind::cooperative) {
        using Bucket = typename S::bucket_type;
        return cooperative_detail::validate_storage(table, selected.threads, Bucket(selected.static_buckets));
    }
    if constexpr (is_lrb<S>::value)
        if (batch_detail::needs_record_lrb(storage))
            return storage.validate();
    return table.validate();
}

template <typename R, typename H, typename O, typename C, int Items, typename Bucket> struct Select {
    using Policy = TileBuildPolicy<Items, true, Items, 0, Bucket>;

    template <int T> __host__ __device__ static size_t block_bytes(int64_t n, int64_t b, int memory) {
        return SharedBlockHashGraph<T, R, H, O, C, 0, Bucket>::workspace_bytes(
            n, b, memory == int(TileBatchMemory::shared_output), Policy {});
    }

    __host__ __device__ static size_t block_bytes(int t, int64_t n, int64_t b, int memory) {
        switch (t) {
        case 32:
            return block_bytes<32>(n, b, memory);
        case 64:
            return block_bytes<64>(n, b, memory);
        case 128:
            return block_bytes<128>(n, b, memory);
        case 256:
            return block_bytes<256>(n, b, memory);
        case 512:
            return block_bytes<512>(n, b, memory);
        case 1024:
            return block_bytes<1024>(n, b, memory);
        }
        return 0;
    }

    template <typename S>
    __host__ __device__ static bool fits(
        const TableScheduleClass& candidate, const S& storage, int64_t work) {
        const int capacity = batch_detail::needs_record_lrb(storage) ? candidate.lrb_resident_groups
                                                                     : candidate.resident_groups;
        if (!capacity || work > candidate.work_limit)
            return false;
        if (candidate.memory == int(TileBatchMemory::global))
            return true;
        const auto& table = batch_detail::table(storage);
        const auto bytes
            = block_bytes(candidate.threads, table.record_count, table.bucket_count, candidate.memory);
        return bytes && bytes <= candidate.shared_bytes;
    }

    template <typename S> __host__ __device__ static int choose(const Classes& classes, const S& storage) {
        const auto& table = batch_detail::table(storage);
        const int64_t work
            = table.record_count > table.bucket_count ? table.record_count : table.bucket_count;
        const int exponent = bucket_exponent(table.bucket_count);
        if (exponent >= 0 && exponent < 6 && work <= 32) {
            for (int worker = 0; worker < 3; ++worker) {
                const int id = 6 * worker + exponent;
                const auto& candidate = classes.entries[id];
                if (candidate.resident_groups && work <= candidate.work_limit)
                    return id;
            }
        }
        for (int worker = 0; worker < 6; ++worker) {
            const int threads = 32 << worker;
            if (exponent >= 0 && exponent <= 10) {
                for (int mode = 0; mode < 3; ++mode) {
                    const int id = static_begin + 3 * exponent + mode;
                    const auto& candidate = classes.entries[id];
                    if (candidate.threads == threads && fits(candidate, storage, work))
                        return id;
                }
            }
            for (int mode = 0; mode < 3; ++mode) {
                const int id = block_begin + 3 * worker + mode;
                if (fits(classes.entries[id], storage, work))
                    return id;
            }
        }
        int largest = -1;
        for (int worker = 0; worker < 3; ++worker) {
            int selected = -1;
            for (int mode = 0; mode < 2; ++mode) {
                const int id = cluster_begin + 2 * worker + mode;
                const auto& candidate = classes.entries[id];
                const int capacity = batch_detail::needs_record_lrb(storage) ? candidate.lrb_resident_groups
                                                                             : candidate.resident_groups;
                if (!capacity)
                    continue;
                if (mode == 0) {
                    if constexpr (is_lrb<S>::value) {
                        if (batch_detail::needs_record_lrb(storage) && storage.bucket_span > 0) {
                            const auto bytes
                                = ClusterBuildPolicy<Items, false, true, Bucket>::template workspace_bytes<O,
                                    C>(storage.bin_count(), candidate.blocks);
                            if (bytes && bytes <= candidate.shared_bytes) {
                                selected = id;
                                break;
                            }
                        }
                    }
                } else
                    selected = id;
            }
            if (selected >= 0) {
                largest = selected;
                if (work <= classes.entries[selected].work_limit)
                    return selected;
            }
        }
        return largest >= 0 ? largest : fallback_class;
    }
};

template <typename R, typename H, typename O, typename C, int Items, typename B, typename Bucket = uint32_t>
struct Classification {
    B batch;
    Classes classes;
    uint32_t* keys;
    int64_t* ids;
    TileBuildStatus* statuses;
    TileCsrView<R, H, O, Bucket>* views;
    H hash;
};

template <typename R, typename H, typename O, typename C, int Items, typename B, typename Bucket>
__global__ void classify(Classification<R, H, O, C, Items, B, Bucket> args) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < args.batch.table_count;
         i += int64_t(blockDim.x) * gridDim.x) {
        const auto storage = args.batch[i];
        const int group = Select<R, H, O, C, Items, Bucket>::choose(args.classes, storage);
        const auto status = validate(storage, args.classes.entries[group]);
        uint32_t key = range_count << priority_bits;
        if (status == TileBuildStatus::success) {
            const auto& table = batch_detail::table(storage);
            const int64_t work
                = table.record_count > table.bucket_count ? table.record_count : table.bucket_count;
            const bool lrb = args.classes.entries[group].kind != TableScheduleKind::cooperative
                && batch_detail::needs_record_lrb(storage);
            key = (uint32_t(2 * group + int(lrb)) << priority_bits) | uint32_t(__clzll(uint64_t(work)));
        } else {
            args.statuses[i] = status;
            if (args.views)
                args.views[i] = { nullptr, nullptr, O { 0 }, args.hash };
        }
        args.keys[i] = key;
        args.ids[i] = i;
    }
}

__global__ void ranges(const uint32_t* keys, int64_t count, TableScheduleRange* result) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(blockDim.x) * gridDim.x) {
        const uint32_t group = keys[i] >> priority_bits;
        if (group > range_count)
            continue;
        if (!i || (keys[i - 1] >> priority_bits) != group)
            result[group].begin = i;
        if (i + 1 == count || (keys[i + 1] >> priority_bits) != group)
            result[group].end = i + 1;
    }
}

inline cudaError_t sort_pairs(void* temp, size_t& bytes, uint32_t* in, uint32_t* out, int64_t* ids,
    int64_t* sorted, int64_t count, cudaStream_t stream = nullptr) {
    if (count <= std::numeric_limits<int>::max())
        return cub::DeviceRadixSort::SortPairs(
            temp, bytes, in, out, ids, sorted, int(count), 0, sort_bits, stream);
    return cub::DeviceRadixSort::SortPairs(temp, bytes, in, out, ids, sorted, count, 0, sort_bits, stream);
}
} // namespace schedule_detail

template <typename Record = KeyRecord, typename Hash = cuco::default_hash_function<RecordKey<Record>>, typename Offset = int32_t,
    typename Counter = Offset, int Items = 8, typename Bucket = uint32_t>
class ScheduledHashGraph {
    static_assert(Items == 1 || Items == 2 || Items == 4 || Items == 8 || Items == 16);
    using Storage = TileBuildStorage<Record, Offset, Counter, Bucket>;
    using LrbStorage = TileLrbBuildStorage<Record, Offset, Counter, Bucket>;
    using View = TileCsrView<Record, Hash, Offset, Bucket>;
    using TilePolicy = TileBuildPolicy<Items, true, Items, 0, Bucket>;
    using FallbackPolicy = TileBuildPolicy<0, false, Items, 0, Bucket>;
    template <typename B>
    using WorkerBatch
        = std::conditional_t<schedule_detail::is_uniform<B>::value, B, schedule_detail::GroupedBatch<B>>;
    template <typename B>
    using DirectSource = decltype(schedule_detail::as_direct(std::declval<WorkerBatch<B>>()));

    template <typename B> static void check_type() {
        static_assert(std::is_same_v<typename B::storage_type, Storage>
            || std::is_same_v<typename B::storage_type, LrbStorage>);
    }

    template <int Threads, typename Source, typename Policy, TileBatchMemory Memory, uint32_t Buckets = 0>
    static cudaError_t block_resource(
        TableScheduleClass& result, size_t requested, const cudaDeviceProp& device, size_t budget) {
        auto kernel = batch_detail::block_build<Threads, Record, Hash, Offset, Counter, Source, Policy,
            Memory, Buckets, Bucket>;
        cudaFuncAttributes attributes {};
        auto error = cudaFuncGetAttributes(&attributes, kernel);
        if (error != cudaSuccess)
            return error;
        if (Threads > attributes.maxThreadsPerBlock || Threads > device.maxThreadsPerBlock
            || Threads % device.warpSize || attributes.sharedSizeBytes > device.sharedMemPerBlockOptin)
            return cudaSuccess;
        const size_t limit = device.sharedMemPerBlockOptin - attributes.sharedSizeBytes;
        size_t bytes = std::min(requested, limit);
        if (budget)
            bytes = std::min(bytes, budget);
        if (Memory != TileBatchMemory::global && !bytes)
            return cudaSuccess;
        error = batch_detail::configure_shared(kernel, bytes);
        if (error != cudaSuccess)
            return error;
        int capacity = 0;
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&capacity, kernel, Threads, bytes);
        if (error != cudaSuccess)
            return error;
        if (capacity)
            result = { Threads, 1, int(Memory), capacity * device.multiProcessorCount,
                int64_t(Items) * Threads, bytes, Buckets, TableScheduleKind::block };
        return cudaSuccess;
    }

    static TableScheduleClass merge_resources(TableScheduleClass direct, const TableScheduleClass& lrb) {
        if (!direct.resident_groups) {
            direct = lrb;
            direct.resident_groups = 0;
        } else if (lrb.resident_groups)
            direct.shared_bytes = std::min(direct.shared_bytes, lrb.shared_bytes);
        direct.lrb_resident_groups = lrb.resident_groups;
        return direct;
    }

    template <int Threads, uint32_t Buckets, typename Source>
    static cudaError_t tiny_resource(TableScheduleClass& result, const cudaDeviceProp& device) {
        auto kernel
            = cooperative_detail::build<Threads, Buckets, 128, Record, Hash, Offset, Counter, Source, Bucket>;
        cudaFuncAttributes attributes {};
        auto error = cudaFuncGetAttributes(&attributes, kernel);
        if (error != cudaSuccess)
            return error;
        if (128 > attributes.maxThreadsPerBlock || 128 > device.maxThreadsPerBlock)
            return cudaSuccess;
        int capacity = 0;
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&capacity, kernel, 128, 0);
        if (error != cudaSuccess)
            return error;
        if (capacity)
            result = { Threads, 1, 0, capacity * device.multiProcessorCount, Threads, 0, Buckets,
                TableScheduleKind::cooperative };
        return cudaSuccess;
    }

    template <typename Source, bool Cache, bool Shared>
    static cudaError_t cluster_resource(
        TableScheduleClass& result, int blocks, const cudaDeviceProp& device, size_t budget) {
        using Policy = ClusterBuildPolicy<Items, Cache, Shared, Bucket>;
        auto kernel = batch_detail::cluster_build<256, Record, Hash, Offset, Counter, Source, Policy, Bucket>;
        cudaFuncAttributes attributes {};
        auto error = cudaFuncGetAttributes(&attributes, kernel);
        if (error != cudaSuccess)
            return error;
        if (attributes.ptxVersion < 90) return cudaSuccess;
        if (256 > attributes.maxThreadsPerBlock || attributes.sharedSizeBytes > device.sharedMemPerBlockOptin)
            return cudaSuccess;
        size_t bytes = Shared ? size_t(Items) * 256 * (sizeof(Offset) + sizeof(Counter)) : 0;
        bytes = std::min(bytes, device.sharedMemPerBlockOptin - attributes.sharedSizeBytes);
        if (budget)
            bytes = std::min(bytes, budget);
        if (Shared && !bytes)
            return cudaSuccess;
        error = batch_detail::configure_shared(kernel, bytes);
        if (error != cudaSuccess)
            return error;
        cudaLaunchAttribute attribute {};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim = { unsigned(blocks), 1, 1 };
        cudaLaunchConfig_t config {};
        config.blockDim = dim3(256);
        config.gridDim = dim3((1 << 20) / blocks * blocks);
        config.dynamicSmemBytes = bytes;
        config.attrs = &attribute;
        config.numAttrs = 1;
        int capacity = 0;
        error = cudaOccupancyMaxActiveClusters(&capacity, kernel, &config);
        if (error == cudaErrorInvalidClusterSize || error == cudaErrorNotSupported
            || error == cudaErrorInvalidValue) {
            cudaGetLastError();
            return cudaSuccess;
        }
        if (error != cudaSuccess)
            return error;
        if (capacity)
            result = { 256, blocks, Shared ? 1 : 0, capacity, int64_t(Items) * 256 * blocks, bytes, 0,
                TableScheduleKind::cluster };
        return cudaSuccess;
    }

    template <typename B, bool Cache, bool Shared>
    static cudaError_t cluster_pair(
        TableScheduleClass& result, int blocks, const cudaDeviceProp& device, size_t budget) {
        TableScheduleClass direct {}, lrb {};
        auto error = cluster_resource<DirectSource<B>, Cache, Shared>(direct, blocks, device, budget);
        if (error != cudaSuccess)
            return error;
        if constexpr (schedule_detail::is_lrb<typename B::storage_type>::value) {
            error = cluster_resource<WorkerBatch<B>, Cache, Shared>(lrb, blocks, device, budget);
            if (error != cudaSuccess)
                return error;
        }
        result = merge_resources(direct, lrb);
        return cudaSuccess;
    }

    template <typename B, bool Cache>
    static cudaError_t cluster_resources(
        schedule_detail::Classes& classes, const cudaDeviceProp& device, size_t budget) {
        for (int i = 0; i < 3; ++i) {
            const int blocks = 2 << i;
            auto error = cluster_pair<B, Cache, false>(
                classes.entries[schedule_detail::cluster_begin + 2 * i + 1], blocks, device, budget);
            if (error != cudaSuccess)
                return error;
            if constexpr (schedule_detail::is_lrb<typename B::storage_type>::value) {
                error = cluster_pair<B, Cache, true>(
                    classes.entries[schedule_detail::cluster_begin + 2 * i], blocks, device, budget);
                if (error != cudaSuccess)
                    return error;
            }
        }
        return cudaSuccess;
    }

    template <int Threads, uint32_t Buckets, typename Source>
    static cudaError_t block_resources(
        schedule_detail::Classes& classes, int begin, const cudaDeviceProp& device, size_t budget) {
        constexpr int64_t upper = int64_t(Items) * Threads;
        const int64_t b = Buckets ? Buckets : upper;
        const size_t output
            = SharedBlockHashGraph<Threads, Record, Hash, Offset, Counter, Buckets, Bucket>::workspace_bytes(
                upper, b, true, TilePolicy {});
        const size_t metadata
            = SharedBlockHashGraph<Threads, Record, Hash, Offset, Counter, Buckets, Bucket>::workspace_bytes(
                upper, b, false, TilePolicy {});
        auto error = block_resource<Threads, Source, TilePolicy, TileBatchMemory::shared_output, Buckets>(
            classes.entries[begin], output, device, budget);
        if (error != cudaSuccess)
            return error;
        error = block_resource<Threads, Source, TilePolicy, TileBatchMemory::shared_metadata, Buckets>(
            classes.entries[begin + 1], metadata, device, budget);
        if (error != cudaSuccess)
            return error;
        return block_resource<Threads, Source, TilePolicy, TileBatchMemory::global, Buckets>(
            classes.entries[begin + 2], 0, device, budget);
    }

    template <int Threads, uint32_t Buckets, typename B>
    static cudaError_t block_pairs(
        schedule_detail::Classes& classes, int begin, const cudaDeviceProp& device, size_t budget) {
        auto error = block_resources<Threads, Buckets, DirectSource<B>>(classes, begin, device, budget);
        if (error != cudaSuccess)
            return error;
        if constexpr (schedule_detail::is_lrb<typename B::storage_type>::value) {
            schedule_detail::Classes other {};
            error = block_resources<Threads, Buckets, WorkerBatch<B>>(other, begin, device, budget);
            if (error != cudaSuccess)
                return error;
            for (int mode = 0; mode < 3; ++mode)
                classes.entries[begin + mode]
                    = merge_resources(classes.entries[begin + mode], other.entries[begin + mode]);
        }
        return cudaSuccess;
    }

    template <typename Source, bool Cache>
    static cudaError_t launch_cluster(const Source& source, TileBuildStatus* statuses,
        const TableScheduleClass& selected, TypedBatchBucketCaches<Bucket> caches, cudaStream_t stream,
        Hash hash, View* views) {
        using Builder = BatchedClusterHashGraph<256, Record, Hash, Offset, Counter, Bucket>;
        TypedClusterBatchOptions<Bucket> options;
        options.blocks_per_table = selected.blocks;
        options.shared_bytes = selected.shared_bytes;
        options.max_clusters = selected.resident_groups;
        options.bucket_caches = caches;
        if constexpr (schedule_detail::is_uniform<Source>::value) {
            options.max_clusters = 0;
            if constexpr (schedule_detail::is_lrb<typename Source::storage_type>::value)
                if (selected.memory)
                    options.shared_bytes
                        = ClusterBuildPolicy<Items, Cache, true, Bucket>::template workspace_bytes<Offset,
                            Counter>(source.first.bin_count(), selected.blocks);
        }
        if (selected.memory)
            return Builder::build(source, statuses, options, stream, hash,
                ClusterBuildPolicy<Items, Cache, true, Bucket> {}, views);
        return Builder::build(source, statuses, options, stream, hash,
            ClusterBuildPolicy<Items, Cache, false, Bucket> {}, views);
    }

    template <int Threads, uint32_t Buckets, bool Fallback = false, typename Source>
    static cudaError_t launch_block(const Source& source, TileBuildStatus* statuses, TileBatchOptions options,
        cudaStream_t stream, Hash hash, View* views) {
        using Builder = BatchedBlockHashGraph<Threads, Record, Hash, Offset, Counter, Buckets, Bucket>;
        if constexpr (Fallback)
            return Builder::build(source, statuses, options, stream, hash, FallbackPolicy {}, views);
        else
            return Builder::build(source, statuses, options, stream, hash, TilePolicy {}, views);
    }

    template <uint32_t Buckets, typename Source>
    static cudaError_t launch_tiny(int threads, const Source& source, TileBuildStatus* statuses,
        CooperativeBatchOptions options, cudaStream_t stream, Hash hash, View* views) {
        const auto& worker = source;
        if (threads == 8)
            return BatchedCooperativeHashGraph<8, Buckets, 128, Record, Hash, Offset, Counter, Bucket>::build(
                worker, statuses, options, stream, hash, views);
        if (threads == 16)
            return BatchedCooperativeHashGraph<16, Buckets, 128, Record, Hash, Offset, Counter,
                Bucket>::build(worker, statuses, options, stream, hash, views);
        return BatchedCooperativeHashGraph<32, Buckets, 128, Record, Hash, Offset, Counter, Bucket>::build(
            worker, statuses, options, stream, hash, views);
    }

    template <typename Source>
    static cudaError_t launch(int group, const TableScheduleClass& selected, const Source& source,
        TileBuildStatus* statuses, TypedBatchBucketCaches<Bucket> caches, bool cache_enabled,
        cudaStream_t stream, Hash hash, View* views) {
        if (selected.kind == TableScheduleKind::cooperative) {
            if constexpr (schedule_detail::is_lrb<typename Source::storage_type>::value)
                return cudaErrorInvalidValue;
            else {
                CooperativeBatchOptions options;
                options.max_blocks
                    = schedule_detail::is_uniform<Source>::value ? 0 : selected.resident_groups;
                return schedule_detail::with_bucket<5>(group % 6, [&](auto b) {
                    return launch_tiny<decltype(b)::value>(
                        selected.threads, source, statuses, options, stream, hash, views);
                });
            }
        }
        if (selected.kind == TableScheduleKind::cluster) {
            if (cache_enabled)
                return launch_cluster<Source, true>(source, statuses, selected, caches, stream, hash, views);
            return launch_cluster<Source, false>(source, statuses, selected, caches, stream, hash, views);
        }
        TileBatchOptions options;
        options.memory = TileBatchMemory(selected.memory);
        options.shared_bytes = selected.shared_bytes;
        options.max_blocks = selected.resident_groups;
        if constexpr (schedule_detail::is_uniform<Source>::value) {
            options.max_blocks = 0;
            if (options.memory != TileBatchMemory::global) {
                const auto& table = batch_detail::table(source.first);
                options.shared_bytes
                    = schedule_detail::Select<Record, Hash, Offset, Counter, Items, Bucket>::block_bytes(
                        selected.threads, table.record_count, table.bucket_count, selected.memory);
            }
        }
        if (group >= schedule_detail::static_begin && group < schedule_detail::block_begin) {
            const int exponent = (group - schedule_detail::static_begin) / 3;
            return schedule_detail::with_bucket(exponent, [&](auto b) {
                constexpr uint32_t B = decltype(b)::value;
                constexpr int T = schedule_detail::static_threads<Items, B>();
                return launch_block<T, B>(source, statuses, options, stream, hash, views);
            });
        }
        return schedule_detail::with_threads(selected.threads, [&](auto t) {
            if (group == schedule_detail::fallback_class)
                return launch_block<decltype(t)::value, 0, true>(
                    source, statuses, options, stream, hash, views);
            return launch_block<decltype(t)::value, 0>(source, statuses, options, stream, hash, views);
        });
    }

    template <typename Source>
    static cudaError_t launch_route(int group, bool lrb, TableScheduleClass selected, const Source& source,
        TileBuildStatus* statuses, TypedBatchBucketCaches<Bucket> caches, bool cache_enabled,
        cudaStream_t stream, Hash hash, View* views) {
        if constexpr (schedule_detail::is_lrb<typename Source::storage_type>::value) {
            if (lrb) {
                selected.resident_groups = selected.lrb_resident_groups;
                return launch(group, selected, source, statuses, caches, cache_enabled, stream, hash, views);
            }
        }
        const auto direct = schedule_detail::as_direct(source);
        return launch(group, selected, direct, statuses, caches, cache_enabled, stream, hash, views);
    }

public:
    __host__ __device__ static constexpr Bucket bucket_count(
        int64_t records, int64_t records_per_bucket = 8) {
        if (records < 0 || records_per_bucket <= 0
            || uint64_t(records) > uint64_t(cuda::std::numeric_limits<Offset>::max())
            || uint64_t(records) > uint64_t(cuda::std::numeric_limits<Counter>::max())
            || uint64_t(records) > cuda::std::numeric_limits<size_t>::max() / sizeof(Record))
            return 0;
        const uint64_t required = records ? 1 + (uint64_t(records) - 1) / uint64_t(records_per_bucket) : 1;
        if (required > uint64_t(cuda::std::numeric_limits<Bucket>::max()))
            return 0;
        uint64_t result = 1;
        while (result < required)
            result <<= 1;
        return result <= uint64_t(cuda::std::numeric_limits<Offset>::max())
                && result <= uint64_t(cuda::std::numeric_limits<Bucket>::max())
                && result <= uint64_t(cuda::std::numeric_limits<int64_t>::max())
            ? Bucket(result)
            : Bucket { 0 };
    }

    template <typename B> struct Configuration {
        schedule_detail::Classes classes {};
        schedule_detail::Workspace workspace {};
        std::array<int, schedule_detail::class_count> launch_order {};
        int64_t table_count = 0;
        size_t sort_bytes = 0;
        bool ready = false, cache_enabled = false;
        TypedBatchBucketCaches<Bucket> caches {};
    };

    template <typename B>
    static cudaError_t configure(
        B batch, Configuration<B>& result, TypedTableScheduleOptions<Bucket> options = {}) {
        check_type<B>();
        result = {};
        if (!schedule_detail::basic_valid(batch) || !options.bucket_caches.valid(batch.table_count))
            return cudaErrorInvalidValue;
        result.table_count = batch.table_count;
        result.caches = options.bucket_caches;
        result.cache_enabled = options.bucket_caches.descriptors || options.bucket_caches.uniform.data;
        if (!batch.table_count) {
            result.ready = true;
            return cudaSuccess;
        }
        int ordinal;
        auto error = cudaGetDevice(&ordinal);
        if (error != cudaSuccess)
            return error;
        cudaDeviceProp device {};
        error = cudaGetDeviceProperties(&device, ordinal);
        if (error != cudaSuccess)
            return error;
        using Source = DirectSource<B>;
        using Tiny = DirectSource<B>;
        for (int exponent = 0; exponent <= 10; ++exponent) {
            error = schedule_detail::with_bucket(exponent, [&](auto b) {
                constexpr uint32_t Buckets = decltype(b)::value;
                constexpr int T = schedule_detail::static_threads<Items, Buckets>();
                auto e = block_pairs<T, Buckets, B>(result.classes,
                    schedule_detail::static_begin + 3 * exponent, device, options.shared_budget);
                if (e != cudaSuccess)
                    return e;
                if constexpr (Buckets <= 32) {
                    e = tiny_resource<8, Buckets, Tiny>(result.classes.entries[exponent], device);
                    if (e != cudaSuccess)
                        return e;
                    e = tiny_resource<16, Buckets, Tiny>(result.classes.entries[6 + exponent], device);
                    if (e != cudaSuccess)
                        return e;
                    return tiny_resource<32, Buckets, Tiny>(result.classes.entries[12 + exponent], device);
                } else
                    return cudaSuccess;
            });
            if (error != cudaSuccess)
                return error;
        }
        for (int worker = 0; worker < 6; ++worker) {
            error = schedule_detail::with_threads(32 << worker, [&](auto t) {
                return block_pairs<decltype(t)::value, 0, B>(
                    result.classes, schedule_detail::block_begin + 3 * worker, device, options.shared_budget);
            });
            if (error != cudaSuccess)
                return error;
        }
        int supported = 0;
        error = cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, ordinal);
        if (error != cudaSuccess)
            return error;
        if (supported && options.allow_clusters) {
            if (result.cache_enabled)
                error = cluster_resources<B, true>(result.classes, device, options.shared_budget);
            else
                error = cluster_resources<B, false>(result.classes, device, options.shared_budget);
            if (error != cudaSuccess)
                return error;
        }
        for (int threads : { 256, 512, 1024 }) {
            TableScheduleClass candidate {};
            error = schedule_detail::with_threads(threads, [&](auto t) {
                auto e = block_resource<decltype(t)::value, Source, FallbackPolicy, TileBatchMemory::global>(
                    candidate, 0, device, options.shared_budget);
                if (e != cudaSuccess)
                    return e;
                if constexpr (schedule_detail::is_lrb<typename B::storage_type>::value) {
                    TableScheduleClass lrb {};
                    e = block_resource<decltype(t)::value, WorkerBatch<B>, FallbackPolicy,
                        TileBatchMemory::global>(lrb, 0, device, options.shared_budget);
                    if (e != cudaSuccess)
                        return e;
                    candidate = merge_resources(candidate, lrb);
                }
                return cudaSuccess;
            });
            if (error != cudaSuccess)
                return error;
            if (candidate.resident_groups
                && (!schedule_detail::is_lrb<typename B::storage_type>::value
                    || candidate.lrb_resident_groups))
                result.classes.entries[schedule_detail::fallback_class] = candidate;
        }
        if (!result.classes.entries[schedule_detail::fallback_class].resident_groups)
            return cudaErrorInvalidConfiguration;
        if constexpr (schedule_detail::is_uniform<B>::value) {
            const int group = schedule_detail::Select<Record, Hash, Offset, Counter, Items, Bucket>::choose(
                result.classes, batch.first);
            if (schedule_detail::validate(batch.first, result.classes.entries[group])
                != TileBuildStatus::success)
                return cudaErrorInvalidValue;
        } else {
            error = schedule_detail::sort_pairs(
                nullptr, result.sort_bytes, nullptr, nullptr, nullptr, nullptr, batch.table_count);
            if (error != cudaSuccess)
                return error;
            if (!schedule_detail::workspace(batch.table_count, result.sort_bytes, result.workspace))
                return cudaErrorInvalidValue;
        }
        for (int i = 0; i < class_count; ++i)
            result.launch_order[i] = i;
        std::sort(result.launch_order.begin(), result.launch_order.end(), [&](int a, int b) {
            const auto& x = result.classes.entries[a];
            const auto& y = result.classes.entries[b];
            const int64_t wx
                = a == schedule_detail::fallback_class ? std::numeric_limits<int64_t>::max() : x.work_limit;
            const int64_t wy
                = b == schedule_detail::fallback_class ? std::numeric_limits<int64_t>::max() : y.work_limit;
            return wx != wy ? wx > wy : a > b;
        });
        result.ready = true;
        return cudaSuccess;
    }

    template <typename B> static size_t workspace_bytes(const Configuration<B>& c) {
        return c.ready ? c.workspace.bytes : 0;
    }

    template <typename B> static uint32_t* classification_keys(const Configuration<B>& c, void* temp) {
        if constexpr (schedule_detail::is_uniform<B>::value)
            return nullptr;
        return c.ready && c.workspace.bytes && temp
            ? schedule_detail::pointer<uint32_t>(temp, c.workspace.keys_in)
            : nullptr;
    }

    template <typename B> static const TableScheduleRange* ranges(const Configuration<B>& c, void* temp) {
        if constexpr (schedule_detail::is_uniform<B>::value)
            return nullptr;
        return c.ready && c.workspace.bytes && temp
            ? schedule_detail::pointer<TableScheduleRange>(temp, c.workspace.ranges)
            : nullptr;
    }

    static constexpr int class_count = schedule_detail::class_count;
    static constexpr int range_count = schedule_detail::range_count;

    static constexpr int class_from_key(uint32_t key) {
        return key >> (schedule_detail::priority_bits + 1);
    }

    static constexpr bool record_lrb_from_key(uint32_t key) {
        return (key >> schedule_detail::priority_bits) & 1;
    }

    static constexpr int range_from_key(uint32_t key) {
        return key >> schedule_detail::priority_bits;
    }

    template <typename B>
    static cudaError_t build(const Configuration<B>& c, B batch, TileBuildStatus* statuses, void* temp,
        size_t capacity, cudaStream_t stream = nullptr, Hash hash = {}, View* views = nullptr) {
        check_type<B>();
        if (!c.ready || c.table_count != batch.table_count || !schedule_detail::basic_valid(batch)
            || (batch.table_count && !statuses)
            || !batch_detail::slab_fits<TileBuildStatus>(batch.table_count, 1)
            || (views && !batch_detail::slab_fits<View>(batch.table_count, 1)))
            return cudaErrorInvalidValue;
        if (!batch.table_count)
            return cudaSuccess;
        if constexpr (schedule_detail::is_uniform<B>::value) {
            const int group = schedule_detail::Select<Record, Hash, Offset, Counter, Items, Bucket>::choose(
                c.classes, batch.first);
            if (schedule_detail::validate(batch.first, c.classes.entries[group]) != TileBuildStatus::success)
                return cudaErrorInvalidValue;
            const bool lrb = c.classes.entries[group].kind != TableScheduleKind::cooperative
                && batch_detail::needs_record_lrb(batch.first);
            return launch_route(group, lrb, c.classes.entries[group], batch, statuses, c.caches,
                c.cache_enabled, stream, hash, views);
        } else {
            if (!temp || reinterpret_cast<uintptr_t>(temp) % 256 || capacity < c.workspace.bytes)
                return cudaErrorInvalidValue;
            const auto& w = c.workspace;
            auto* keys = schedule_detail::pointer<uint32_t>(temp, w.keys_in);
            auto* sorted_keys = schedule_detail::pointer<uint32_t>(temp, w.keys_out);
            auto* ids = schedule_detail::pointer<int64_t>(temp, w.ids_in);
            auto* sorted_ids = schedule_detail::pointer<int64_t>(temp, w.ids_out);
            auto* groups = schedule_detail::pointer<TableScheduleRange>(temp, w.ranges);
            auto error = cudaMemsetAsync(groups, 0, (range_count + 1) * sizeof(TableScheduleRange), stream);
            if (error != cudaSuccess)
                return error;
            const unsigned blocks = unsigned(
                std::min<int64_t>(1 + (batch.table_count - 1) / 256, std::numeric_limits<int>::max()));
            schedule_detail::Classification<Record, Hash, Offset, Counter, Items, B, Bucket> arguments {
                batch, c.classes, keys, ids, statuses, views, hash
            };
            schedule_detail::classify<<<blocks, 256, 0, stream>>>(arguments);
            error = cudaGetLastError();
            if (error != cudaSuccess)
                return error;
            size_t bytes = c.sort_bytes;
            error = schedule_detail::sort_pairs(static_cast<unsigned char*>(temp) + w.sort, bytes, keys,
                sorted_keys, ids, sorted_ids, batch.table_count, stream);
            if (error != cudaSuccess)
                return error;
            schedule_detail::ranges<<<blocks, 256, 0, stream>>>(sorted_keys, batch.table_count, groups);
            error = cudaGetLastError();
            if (error != cudaSuccess)
                return error;
            for (int group : c.launch_order) {
                const auto& selected = c.classes.entries[group];
                for (int lrb = 0; lrb < 2; ++lrb) {
                    if (!(lrb ? selected.lrb_resident_groups : selected.resident_groups))
                        continue;
                    schedule_detail::GroupedBatch<B> source { batch, sorted_ids, groups, 2 * group + lrb,
                        batch.table_count };
                    error = launch_route(group, bool(lrb), selected, source, statuses, c.caches,
                        c.cache_enabled, stream, hash, views);
                    if (error != cudaSuccess)
                        return error;
                }
            }
            return cudaSuccess;
        }
    }
};
} // namespace cuco::detail::bulk_tiles
