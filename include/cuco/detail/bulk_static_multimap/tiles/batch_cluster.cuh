/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "batch_block.cuh"
#include "cluster.cuh"
#if CUDART_VERSION < 12000
#error Batched cluster HashGraph requires CUDA 12 or newer
#endif

namespace cuco::detail::bulk_tiles {
template <typename Bucket>
struct TypedTableBucketCache {
    Bucket* data = nullptr;
    int64_t capacity = 0;
};
template <typename Bucket>
struct TypedBatchBucketCaches {
    const TypedTableBucketCache<Bucket>* descriptors = nullptr;
    TypedTableBucketCache<Bucket> uniform{};
    __host__ bool valid(int64_t count) const {
        return !(descriptors && uniform.data) &&
            (descriptors ? batch_detail::slab_fits<TypedTableBucketCache<Bucket>>(count, 1) :
                          batch_detail::slab_fits<Bucket>(count, uniform.capacity));
    }
    __device__ TypedTableBucketCache<Bucket> operator[](int64_t index) const {
        if (descriptors) return descriptors[index];
        return {batch_detail::advance(uniform.data, index, uniform.capacity), uniform.capacity};
    }
};
template <typename Bucket>
struct TypedClusterBatchOptions {
    int blocks_per_table = 2;
    size_t shared_bytes = 0;
    int max_clusters = 0;
    bool allow_nonportable_cluster_size = false;
    TypedBatchBucketCaches<Bucket> bucket_caches{};
};

using TableBucketCache = TypedTableBucketCache<uint32_t>;
using BatchBucketCaches = TypedBatchBucketCaches<uint32_t>;
using ClusterBatchOptions = TypedClusterBatchOptions<uint32_t>;

namespace batch_detail {
template <int Threads, typename Record, typename Hash, typename Offset, typename Counter, typename Batch, typename Policy, typename Bucket>
struct ClusterArguments {
    Batch batch;
    TileBuildStatus* statuses;
    TileCsrView<Record, Hash, Offset, Bucket>* views;
    size_t shared_bytes;
    TypedBatchBucketCaches<Bucket> caches;
    Hash hash;
};

template <int Threads, typename Record, typename Hash, typename Offset, typename Counter, typename Batch, typename Policy, typename Bucket>
__global__ void cluster_build(ClusterArguments<Threads, Record, Hash, Offset, Counter, Batch, Policy, Bucket> args) {
    const auto batch = resolve(args.batch);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using Builder = ClusterHashGraph<Threads, Record, Hash, Offset, Counter, Bucket>;
    __shared__ typename Builder::TempStorage temp;
    constexpr size_t alignment = 16;
    extern __shared__ __align__(alignment) unsigned char workspace[];
    const auto cluster = cooperative_groups::this_cluster();
    const int64_t stride = gridDim.x / cluster.num_blocks();
    for (int64_t index = blockIdx.x / cluster.num_blocks(); index < batch.table_count;) {
        const auto storage = batch[index];
        const int64_t destination = output_index(batch, index);
        Policy policy{};
        if constexpr (Policy::cache_buckets) {
            const auto cache = args.caches[destination];
            policy.bucket_cache = cache.data;
            policy.bucket_capacity = cache.capacity;
        }
        if constexpr (Policy::shared_lrb_bins) {
            policy.shared_workspace = workspace;
            policy.workspace_capacity = args.shared_bytes;
        }
        const auto result = Builder(temp).build(storage, args.hash, {}, policy);
        if (!cluster.thread_rank()) {
            args.statuses[destination] = result;
            if (args.views) args.views[destination] = result == TileBuildStatus::success ?
                table(storage).view(args.hash) : TileCsrView<Record, Hash, Offset, Bucket>{nullptr, nullptr, Offset{0}, args.hash};
        }
        if (batch.table_count - index <= stride) break;
        cluster.sync();
        index += stride;
    }
#else
    if (!threadIdx.x)
        for (int64_t index = blockIdx.x; index < batch.table_count; index += gridDim.x) {
            const int64_t destination = output_index(batch, index);
            args.statuses[destination] = TileBuildStatus::unsupported_architecture;
            if (args.views) args.views[destination] = TileCsrView<Record, Hash, Offset, Bucket>{nullptr, nullptr, Offset{0}, args.hash};
        }
#endif
}
} // namespace batch_detail

template <int Threads, typename Record = KeyRecord, typename Hash = cuco::default_hash_function<RecordKey<Record>>,
          typename Offset = int32_t, typename Counter = Offset, typename Bucket = uint32_t>
class BatchedClusterHashGraph {
    static_assert(Threads > 0 && Threads <= 1024);
    using Storage = TileBuildStorage<Record, Offset, Counter, Bucket>;
    using LrbStorage = TileLrbBuildStorage<Record, Offset, Counter, Bucket>;
    using View = TileCsrView<Record, Hash, Offset, Bucket>;
    template <typename Batch>
    __host__ static void check_type() {
        static_assert(std::is_same_v<typename Batch::storage_type, Storage> ||
                      std::is_same_v<typename Batch::storage_type, LrbStorage>, "Batch storage must match builder types");
    }
    template <typename Policy>
    __host__ static bool valid_options(TypedClusterBatchOptions<Bucket> options, int64_t count) {
        static_assert(std::is_same_v<typename Policy::bucket_type, Bucket>, "Bucket cache policy must match builder bucket type");
        return options.blocks_per_table > 0 && options.max_clusters >= 0 &&
            options.shared_bytes <= size_t(std::numeric_limits<int>::max()) &&
            (options.blocks_per_table <= 8 || options.allow_nonportable_cluster_size) &&
            (!Policy::cache_buckets || options.bucket_caches.valid(count));
    }
    __host__ static cudaLaunchConfig_t launch_config(int64_t count, TypedClusterBatchOptions<Bucket> options,
                                                     cudaStream_t stream, cudaLaunchAttribute& attribute) {
        const int limit = std::numeric_limits<int>::max() / options.blocks_per_table;
        int64_t clusters = count;
        if (clusters > limit) clusters = limit;
        if (options.max_clusters && clusters > options.max_clusters) clusters = options.max_clusters;
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim = {unsigned(options.blocks_per_table), 1, 1};
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(unsigned(clusters * options.blocks_per_table));
        config.blockDim = dim3(Threads);
        config.dynamicSmemBytes = options.shared_bytes;
        config.stream = stream;
        config.attrs = &attribute;
        config.numAttrs = 1;
        return config;
    }
public:
    template <typename Policy = ClusterBuildPolicy<0, false, false, Bucket>>
    __host__ static size_t workspace_bytes(int64_t bins, int blocks_per_table, Policy policy = {}) {
        static_assert(std::is_same_v<typename Policy::bucket_type, Bucket>, "Bucket cache policy must match builder bucket type");
        return policy.template workspace_bytes<Offset, Counter>(bins, blocks_per_table);
    }

    template <typename Batch, typename Policy = ClusterBuildPolicy<0, false, false, Bucket>>
    __host__ static cudaError_t configure(Batch batch, TypedClusterBatchOptions<Bucket> options = {}, Policy = {},
                                         int* active_clusters = nullptr) {
        check_type<Batch>();
        if (active_clusters) *active_clusters = 0;
        if (!batch.valid() || !valid_options<Policy>(options, batch.table_count)) return cudaErrorInvalidValue;
        if (!batch.table_count) return cudaSuccess;
        int device, supported;
        auto error = cudaGetDevice(&device);
        if (error != cudaSuccess) return error;
        error = cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, device);
        if (error != cudaSuccess) return error;
        if (!supported) return cudaErrorNotSupported;
        auto kernel = batch_detail::cluster_build<Threads, Record, Hash, Offset, Counter, Batch, Policy, Bucket>;
        cudaFuncAttributes attributes{};
        error = cudaFuncGetAttributes(&attributes, kernel);
        if (error != cudaSuccess) return error;
        if (attributes.ptxVersion < 90) return cudaErrorNotSupported;
        error = batch_detail::configure_shared(kernel, options.shared_bytes);
        if (error != cudaSuccess) return error;
        if (options.allow_nonportable_cluster_size) {
            error = cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            if (error != cudaSuccess) return error;
        }
        cudaLaunchAttribute attribute{};
        auto config = launch_config(batch.table_count, options, nullptr, attribute);
        int capacity = 0;
        error = cudaOccupancyMaxActiveClusters(&capacity, kernel, &config);
        if (error != cudaSuccess) return error;
        if (!capacity) return cudaErrorInvalidConfiguration;
        if (active_clusters) *active_clusters = capacity;
        return cudaSuccess;
    }

    template <typename Batch, typename Policy = ClusterBuildPolicy<0, false, false, Bucket>>
    __host__ static cudaError_t build(Batch batch, TileBuildStatus* statuses, TypedClusterBatchOptions<Bucket> options = {},
                                      cudaStream_t stream = nullptr, Hash hash = {}, Policy = {}, View* views = nullptr) {
        check_type<Batch>();
        if (!batch_detail::valid_outputs(batch, statuses, views) || !valid_options<Policy>(options, batch.table_count))
            return cudaErrorInvalidValue;
        if (!batch.table_count) return cudaSuccess;
        cudaLaunchAttribute attribute{};
        auto config = launch_config(batch.table_count, options, stream, attribute);
        using Arguments = batch_detail::ClusterArguments<Threads, Record, Hash, Offset, Counter, Batch, Policy, Bucket>;
        Arguments arguments{batch, statuses, views, options.shared_bytes, options.bucket_caches, hash};
        return cudaLaunchKernelEx(&config,
            batch_detail::cluster_build<Threads, Record, Hash, Offset, Counter, Batch, Policy, Bucket>, arguments);
    }
};
} // namespace cuco::detail::bulk_tiles
