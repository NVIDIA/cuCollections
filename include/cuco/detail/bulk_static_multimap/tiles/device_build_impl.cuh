/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "device_build.cuh"

#include <cub/device/device_scan.cuh>

#include <algorithm>

namespace cuco::detail::bulk_tiles {
namespace single_detail {
constexpr int threads = 256;

template <typename Counter, typename Offset>
cudaError_t scan_counts(void* workspace,
                        size_t& bytes,
                        Counter* counters,
                        Offset* offsets,
                        int64_t count,
                        cudaStream_t stream = nullptr)
{
  if (count <= std::numeric_limits<int>::max())
    return cub::DeviceScan::ExclusiveSum(workspace, bytes, counters, offsets, int(count), stream);
  return cub::DeviceScan::ExclusiveSum(workspace, bytes, counters, offsets, count, stream);
}

template <typename Record, typename Hash, typename Offset, typename Counter, typename Bucket>
__global__ void hash_input(TileLrbBuildStorage<Record, Offset, Counter, Bucket> s, Hash hash)
{
  using Index        = detail::index_type<Offset>;
  Index const count  = Index(s.table.record_count);
  Index const stride = detail::bounded_index(uint64_t(blockDim.x) * gridDim.x, count);
  for (Index i = detail::bounded_index(uint64_t(blockIdx.x) * blockDim.x + threadIdx.x, count);
       i < count;) {
    s.bucket_hashes[i] =
      detail::reduce_hash<Bucket>(hash(s.table.input[i].first), Bucket(s.table.bucket_count));
    if (!detail::advance_index(i, count, stride)) break;
  }
}

template <int P,
          bool Fused,
          bool Shared,
          typename Record,
          typename Hash,
          typename Offset,
          typename Counter,
          typename Bucket>
__global__ void count_lrb(TileLrbBuildStorage<Record, Offset, Counter, Bucket> s,
                          detail::index_type<Offset> bins,
                          Hash hash)
{
  using Index = detail::index_type<Offset>;
  extern __shared__ __align__(16) unsigned char workspace[];
  auto* local = reinterpret_cast<Counter*>(workspace);
  if constexpr (Shared) {
    for (Index b = threadIdx.x; b < bins;) {
      local[b] = 0;
      if (!detail::advance_index(b, bins, Index(blockDim.x))) break;
    }
    __syncthreads();
  }
  Index const count  = Index(s.table.record_count);
  Index const stride = detail::bounded_index(uint64_t(blockDim.x) * gridDim.x * P, count);
  for (Index base =
         detail::bounded_index(uint64_t(blockIdx.x) * blockDim.x * P + threadIdx.x, count);
       base < count;) {
    Bucket buckets[P];
#pragma unroll
    for (int k = 0; k < P; ++k) {
      Index const delta  = Index(blockDim.x) * k;
      bool const present = delta < count - base;
      Index const i      = present ? base + delta : base;
      if (present) {
        if constexpr (Fused) {
          buckets[k] =
            detail::reduce_hash<Bucket>(hash(s.table.input[i].first), Bucket(s.table.bucket_count));
          s.bucket_hashes[i] = buckets[k];
        } else
          buckets[k] = s.bucket_hashes[i];
      }
    }
#pragma unroll
    for (int k = 0; k < P; ++k)
      if (Index(blockDim.x) * k < count - base) {
        Bucket const bin = buckets[k] / Bucket(s.bucket_span);
        if constexpr (Shared)
          detail::atomic_add(local + bin);
        else
          detail::atomic_add(s.bin_counters + bin);
      }
    if (!detail::advance_index(base, count, stride)) break;
  }
  if constexpr (Shared) {
    __syncthreads();
    for (Index b = threadIdx.x; b < bins;) {
      if (local[b]) detail::atomic_add(s.bin_counters + b, local[b]);
      if (!detail::advance_index(b, bins, Index(blockDim.x))) break;
    }
  }
}

template <int P, typename Record, typename Offset, typename Counter, typename Bucket>
__global__ void reorder(TileLrbBuildStorage<Record, Offset, Counter, Bucket> s)
{
  using Index        = detail::index_type<Offset>;
  Index const count  = Index(s.table.record_count);
  Index const stride = detail::bounded_index(uint64_t(blockDim.x) * gridDim.x * P, count);
  for (Index base =
         detail::bounded_index(uint64_t(blockIdx.x) * blockDim.x * P + threadIdx.x, count);
       base < count;) {
    Record records[P];
    Bucket buckets[P], bins[P];
    Counter positions[P];
#pragma unroll
    for (int k = 0; k < P; ++k)
      if (Index(blockDim.x) * k < count - base) {
        records[k] = s.table.input[base + Index(blockDim.x) * k];
        buckets[k] = s.bucket_hashes[base + Index(blockDim.x) * k];
        bins[k]    = buckets[k] / Bucket(s.bucket_span);
      }
#pragma unroll
    for (int k = 0; k < P; ++k)
      if (Index(blockDim.x) * k < count - base)
        positions[k] = detail::atomic_add(s.bin_counters + bins[k]);
#pragma unroll
    for (int k = 0; k < P; ++k)
      if (Index(blockDim.x) * k < count - base) {
        Offset const pos = Offset(positions[k]) + s.bin_offsets[bins[k]];
        s.reordered[pos] = records[k];
      }
    if (!detail::advance_index(base, count, stride)) break;
  }
}

template <int P, typename Record, typename Hash, typename Offset, typename Counter, typename Bucket>
__global__ void count_buckets(TileLrbBuildStorage<Record, Offset, Counter, Bucket> s, Hash hash)
{
  using Index        = detail::index_type<Offset>;
  Index const count  = Index(s.table.record_count);
  Index const stride = detail::bounded_index(uint64_t(blockDim.x) * gridDim.x * P, count);
  for (Index base =
         detail::bounded_index(uint64_t(blockIdx.x) * blockDim.x * P + threadIdx.x, count);
       base < count;) {
    Bucket buckets[P];
#pragma unroll
    for (int k = 0; k < P; ++k)
      if (Index(blockDim.x) * k < count - base) {
        buckets[k] = detail::reduce_hash<Bucket>(
          hash(s.reordered[base + Index(blockDim.x) * k].first), Bucket(s.table.bucket_count));
      }
#pragma unroll
    for (int k = 0; k < P; ++k)
      if (Index(blockDim.x) * k < count - base) detail::atomic_add(s.table.counters + buckets[k]);
    if (!detail::advance_index(base, count, stride)) break;
  }
}

template <int P, typename Record, typename Hash, typename Offset, typename Counter, typename Bucket>
__global__ void scatter(TileLrbBuildStorage<Record, Offset, Counter, Bucket> s, Hash hash)
{
  using Index        = detail::index_type<Offset>;
  Index const count  = Index(s.table.record_count);
  Index const stride = detail::bounded_index(uint64_t(blockDim.x) * gridDim.x * P, count);
  for (Index base =
         detail::bounded_index(uint64_t(blockIdx.x) * blockDim.x * P + threadIdx.x, count);
       base < count;) {
    Record records[P];
    Bucket buckets[P];
    Counter positions[P];
#pragma unroll
    for (int k = 0; k < P; ++k)
      if (Index(blockDim.x) * k < count - base) {
        records[k] = s.reordered[base + Index(blockDim.x) * k];
        buckets[k] =
          detail::reduce_hash<Bucket>(hash(records[k].first), Bucket(s.table.bucket_count));
      }
#pragma unroll
    for (int k = 0; k < P; ++k)
      if (Index(blockDim.x) * k < count - base)
        positions[k] = detail::atomic_add(s.table.counters + buckets[k]);
#pragma unroll
    for (int k = 0; k < P; ++k)
      if (Index(blockDim.x) * k < count - base)
        s.table.records[Offset(positions[k]) + s.table.offsets[buckets[k]]] = records[k];
    if (!detail::advance_index(base, count, stride)) break;
  }
}

template <typename Record, typename Offset, typename Counter, typename Bucket>
__global__ void finish_offset(Offset* offsets,
                              detail::index_type<Offset> buckets,
                              detail::index_type<Offset> count)
{
  if (!threadIdx.x) offsets[buckets] = Offset(count);
}

template <int P,
          bool Shared,
          bool Fused,
          typename Record,
          typename Hash,
          typename Offset,
          typename Counter,
          typename Bucket>
cudaError_t build(TileLrbBuildStorage<Record, Offset, Counter, Bucket> s,
                  SingleBuildWorkspace w,
                  int blocks,
                  cudaStream_t stream,
                  cudaEvent_t* events,
                  Hash hash)
{
  int64_t const bins = s.bin_count();
#define CHECK_CUDA(call)                \
  do {                                  \
    const auto err = (call);            \
    if (err != cudaSuccess) return err; \
  } while (false)
#define MARK(i)                                                 \
  do {                                                          \
    if (events) CHECK_CUDA(cudaEventRecord(events[i], stream)); \
  } while (false)
  MARK(0);
  CHECK_CUDA(cudaMemsetAsync(s.table.counters, 0, s.table.bucket_count * sizeof(Counter), stream));
  CHECK_CUDA(cudaMemsetAsync(s.bin_counters, 0, bins * sizeof(Counter), stream));
  MARK(1);
  if constexpr (!Fused) hash_input<Record, Hash><<<blocks, threads, 0, stream>>>(s, hash);
  count_lrb<P, Fused, Shared, Record, Hash>
    <<<blocks, threads, Shared ? bins * sizeof(Counter) : 0, stream>>>(s, bins, hash);
  MARK(2);
  CHECK_CUDA(scan_counts(w.scan, w.scan_bytes, s.bin_counters, s.bin_offsets, bins, stream));
  CHECK_CUDA(cudaMemsetAsync(s.bin_counters, 0, bins * sizeof(Counter), stream));
  MARK(3);
  int64_t const required_batches =
    s.table.record_count ? 1 + (s.table.record_count - 1) / (threads * P) : 1;
  int const batches = int(std::min<int64_t>(required_batches, std::numeric_limits<int>::max()));
  reorder<P, Record><<<batches, threads, 0, stream>>>(s);
  MARK(4);
  count_buckets<P, Record, Hash><<<batches, threads, 0, stream>>>(s, hash);
  MARK(5);
  CHECK_CUDA(scan_counts(
    w.scan, w.scan_bytes, s.table.counters, s.table.offsets, s.table.bucket_count, stream));
  finish_offset<Record, Offset, Counter, Bucket>
    <<<1, 1, 0, stream>>>(s.table.offsets, s.table.bucket_count, s.table.record_count);
  CHECK_CUDA(cudaMemsetAsync(s.table.counters, 0, s.table.bucket_count * sizeof(Counter), stream));
  MARK(6);
  scatter<P, Record, Hash><<<batches, threads, 0, stream>>>(s, hash);
  MARK(7);
  return cudaGetLastError();
#undef MARK
#undef CHECK_CUDA
}

template <int P,
          bool Fused,
          typename Record,
          typename Hash,
          typename Offset,
          typename Counter,
          typename Bucket>
cudaError_t configure_shared(int64_t bins, size_t limit, bool* enabled)
{
  cudaFuncAttributes attributes{};
  auto err = cudaFuncGetAttributes(
    &attributes, count_lrb<P, Fused, true, Record, Hash, Offset, Counter, Bucket>);
  if (err != cudaSuccess) return err;
  size_t const bytes = size_t(bins) * sizeof(Counter);
  if (attributes.sharedSizeBytes > limit || bytes > limit - attributes.sharedSizeBytes) {
    *enabled = false;
    return cudaSuccess;
  }
  return cudaFuncSetAttribute(count_lrb<P, Fused, true, Record, Hash, Offset, Counter, Bucket>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              int(bytes));
}

template <int P, typename Record, typename Hash, typename Offset, typename Counter, typename Bucket>
cudaError_t select(TileLrbBuildStorage<Record, Offset, Counter, Bucket> s,
                   SingleBuildWorkspace w,
                   SingleBuildPolicy p,
                   int blocks,
                   cudaStream_t stream,
                   cudaEvent_t* e,
                   Hash hash)
{
  if (p.shared_lrb_counts) {
    if (p.fuse_hash_count) return build<P, true, true, Record, Hash>(s, w, blocks, stream, e, hash);
    return build<P, true, false, Record, Hash>(s, w, blocks, stream, e, hash);
  }
  if (p.fuse_hash_count) return build<P, false, true, Record, Hash>(s, w, blocks, stream, e, hash);
  return build<P, false, false, Record, Hash>(s, w, blocks, stream, e, hash);
}
}  // namespace single_detail

template <typename Record, typename Offset, typename Counter, typename Bucket>
cudaError_t single_lrb_workspace_bytes(
  TileLrbBuildStorage<Record, Offset, Counter, Bucket> const& s, size_t* bytes)
{
  static_assert(std::is_trivially_copyable<Record>::value, "Record must be trivially copyable");
  if (!bytes || s.validate() != TileBuildStatus::success) return cudaErrorInvalidValue;
  size_t lrb_bytes = 0, final_bytes = 0;
  int64_t const bins = s.bin_count();
  auto err = single_detail::scan_counts(nullptr, lrb_bytes, s.bin_counters, s.bin_offsets, bins);
  if (err != cudaSuccess) return err;
  err = single_detail::scan_counts(
    nullptr, final_bytes, s.table.counters, s.table.offsets, s.table.bucket_count);
  if (err == cudaSuccess) *bytes = std::max(lrb_bytes, final_bytes);
  return err;
}

template <typename Record, typename Hash, typename Offset, typename Counter, typename Bucket>
cudaError_t single_lrb_build(TileLrbBuildStorage<Record, Offset, Counter, Bucket> const& s,
                             SingleBuildWorkspace w,
                             SingleBuildPolicy p,
                             cudaStream_t stream,
                             cudaEvent_t* events,
                             Hash hash)
{
  static_assert(std::is_trivially_copyable<Record>::value, "Record must be trivially copyable");
  if (s.validate() != TileBuildStatus::success || !w.scan || !w.scan_bytes || p.blocks_per_sm < 1)
    return cudaErrorInvalidValue;
  size_t required      = 0;
  auto workspace_error = single_lrb_workspace_bytes(s, &required);
  if (workspace_error != cudaSuccess) return workspace_error;
  if (w.scan_bytes < required || reinterpret_cast<uintptr_t>(w.scan) % 16)
    return cudaErrorInvalidValue;
  int device;
  cudaDeviceProp prop{};
  auto err = cudaGetDevice(&device);
  if (err != cudaSuccess) return err;
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) return err;
  int64_t const bins = s.bin_count();
  if (p.shared_lrb_counts && size_t(bins) * sizeof(Counter) > prop.sharedMemPerBlockOptin)
    p.shared_lrb_counts = false;
  if (p.shared_lrb_counts) {
#define OPTIN(P)                                                                              \
  if (p.prefetch_items == P) {                                                                \
    if (p.fuse_hash_count)                                                                    \
      err = single_detail::configure_shared<P, true, Record, Hash, Offset, Counter, Bucket>(  \
        bins, prop.sharedMemPerBlockOptin, &p.shared_lrb_counts);                             \
    else                                                                                      \
      err = single_detail::configure_shared<P, false, Record, Hash, Offset, Counter, Bucket>( \
        bins, prop.sharedMemPerBlockOptin, &p.shared_lrb_counts);                             \
  }
    OPTIN(1)
    OPTIN(2) OPTIN(4) OPTIN(8) OPTIN(16)
#undef OPTIN
      if (err != cudaSuccess) return err;
  }
  int const blocks = int(
    std::max<int64_t>(1,
                      std::min((s.table.record_count ? 1 + (s.table.record_count - 1) / 256 : 0),
                               int64_t(prop.multiProcessorCount) * p.blocks_per_sm)));
#define DISPATCH(P) \
  case P: return single_detail::select<P, Record, Hash>(s, w, p, blocks, stream, events, hash)
  switch (p.prefetch_items) {
    DISPATCH(1);
    DISPATCH(2);
    DISPATCH(4);
    DISPATCH(8);
    DISPATCH(16);
    default: return cudaErrorInvalidValue;
  }
#undef DISPATCH
}
}  // namespace cuco::detail::bulk_tiles
