/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuco/detail/error.hpp>
#include <cuco/hash_csr_multimap_ref.cuh>

#include <cuda/atomic>
#include <cuda/std/algorithm>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <cooperative_groups.h>

#include <algorithm>

namespace cuco::detail {
inline constexpr int hash_csr_block_size       = 256;
inline constexpr int hash_csr_warp_size        = 32;
inline constexpr int hash_csr_warps_per_block  = hash_csr_block_size / hash_csr_warp_size;
inline constexpr int hash_csr_outputs_per_lane = 32;

/// Maps every input to the representative of its key and counts the occurrences of each key.
template <class InputIt, class KeyEqual, class Hash>
__global__ void hash_csr_build_count(cuda::std::int64_t size,
                                     InputIt first,
                                     hash_csr_table_ref table,
                                     cuda::std::uint32_t* counts,
                                     cuda::std::uint32_t* representatives,
                                     KeyEqual equal,
                                     Hash hash)
{
  using value_type  = typename cuda::std::iterator_traits<InputIt>::value_type;
  auto const stride = static_cast<cuda::std::int64_t>(gridDim.x) * blockDim.x;
  for (auto i = static_cast<cuda::std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
       i += stride) {
    value_type const pair = first[i];
    auto const index      = static_cast<cuda::std::uint32_t>(i);
    auto const representative =
      table.insert(hash_csr_hash(hash, pair.first), index, [&](cuda::std::uint32_t other) {
        value_type const candidate = first[other];
        return equal(pair.first, candidate.first);
      });
    representatives[index] = representative;
    // Lanes of a warp holding the same key add their occurrences with one atomic.
    auto const peers = __match_any_sync(__activemask(), representative);
    if (threadIdx.x % hash_csr_warp_size == static_cast<unsigned>(__ffs(peers) - 1)) {
      cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{counts[representative]}
        .fetch_add(__popc(peers), cuda::memory_order_relaxed);
    }
  }
}

/**
 * @brief Moves every input next to the other occurrences of its key.
 *
 * `offsets` holds the cumulative end of each group and doubles as its scatter cursor. Once a
 * group is complete its cursor is the group's begin.
 */
template <class InputIt, class Value>
__global__ void hash_csr_build_fill(cuda::std::int64_t size,
                                    InputIt first,
                                    cuda::std::uint32_t const* representatives,
                                    cuda::std::uint32_t* offsets,
                                    Value* records)
{
  auto const stride = static_cast<cuda::std::int64_t>(gridDim.x) * blockDim.x;
  for (auto i = static_cast<cuda::std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
       i += stride) {
    auto const representative = representatives[i];
    auto const peers          = __match_any_sync(__activemask(), representative);
    auto const lane           = threadIdx.x % hash_csr_warp_size;
    auto const leader         = static_cast<unsigned>(__ffs(peers) - 1);
    cuda::std::uint32_t end{};
    if (lane == leader) {
      end =
        cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{offsets[representative]}
          .fetch_sub(__popc(peers), cuda::memory_order_relaxed);
    }
    end                     = __shfl_sync(peers, end, leader);
    auto const rank         = __popc(peers & ((cuda::std::uint32_t{1} << lane) - 1));
    records[end - rank - 1] = first[i];
  }
}

/**
 * @brief Writes the matches of every probe, balancing the output evenly across warps.
 *
 * `offsets` holds the output position of each probe plus the total. A warp owns a contiguous
 * output range and locates the probes it covers by binary search, so a probe with many matches
 * is spread over many warps.
 */
template <class InputIt, class Value, class ProbeOutput, class MatchOutput>
__global__ void hash_csr_retrieve(cuda::std::int64_t output_size,
                                  cuda::std::int64_t num_probes,
                                  cuda::std::int64_t outputs_per_warp,
                                  cuda::std::int64_t const* offsets,
                                  cuda::std::uint32_t const* begins,
                                  InputIt first,
                                  Value const* records,
                                  ProbeOutput probes,
                                  MatchOutput matches)
{
  auto const warp = cooperative_groups::tiled_partition<hash_csr_warp_size>(
    cooperative_groups::this_thread_block());
  auto const lane        = static_cast<cuda::std::int64_t>(warp.thread_rank());
  auto const global_warp = static_cast<cuda::std::int64_t>(blockIdx.x) * hash_csr_warps_per_block +
                           threadIdx.x / hash_csr_warp_size;
  auto const range_begin = outputs_per_warp * global_warp;
  if (range_begin >= output_size) { return; }
  auto const range_end = cuda::std::min(range_begin + outputs_per_warp, output_size);

  cuda::std::int64_t endpoint_probe{};
  if (lane < 2) {
    auto const endpoint = lane == 0 ? range_begin : range_end - 1;
    endpoint_probe =
      cuda::std::upper_bound(offsets, offsets + num_probes + 1, endpoint) - offsets - 1;
  }
  auto const first_probe = warp.shfl(endpoint_probe, 0);
  auto const last_probe  = warp.shfl(endpoint_probe, 1);

#pragma unroll
  for (int item = 0; item < hash_csr_outputs_per_lane; ++item) {
    auto const output = range_begin + lane + item * hash_csr_warp_size;
    if (output < range_end) {
      auto const probe =
        first_probe == last_probe
          ? first_probe
          : cuda::std::upper_bound(offsets + first_probe, offsets + last_probe + 2, output) -
              offsets - 1;
      probes[output]  = first[probe];
      matches[output] = records[begins[probe] + (output - offsets[probe])];
    }
  }
}

inline unsigned hash_csr_grid_size(cuda::std::int64_t size)
{
  return static_cast<unsigned>((size - 1) / hash_csr_block_size + 1);
}

template <class InputIt, class Value, class ProbeOutput, class MatchOutput>
void hash_csr_launch_retrieve(cuda::std::int64_t output_size,
                              cuda::std::int64_t num_probes,
                              cuda::std::int64_t const* offsets,
                              cuda::std::uint32_t const* begins,
                              InputIt first,
                              Value const* records,
                              ProbeOutput probes,
                              MatchOutput matches,
                              cuda::stream_ref stream)
{
  if (!output_size) { return; }
  int device{};
  int multiprocessors{};
  CUCO_CUDA_TRY(cudaGetDevice(&device));
  CUCO_CUDA_TRY(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
  constexpr cuda::std::int64_t outputs_per_block = hash_csr_block_size * hash_csr_outputs_per_lane;
  auto const blocks = std::max<cuda::std::int64_t>((output_size - 1) / outputs_per_block + 1,
                                                   cuda::std::int64_t{2} * multiprocessors);
  auto const warps  = blocks * hash_csr_warps_per_block;
  hash_csr_retrieve<<<static_cast<unsigned>(blocks), hash_csr_block_size, 0, stream.get()>>>(
    output_size,
    num_probes,
    (output_size - 1) / warps + 1,
    offsets,
    begins,
    first,
    records,
    probes,
    matches);
  CUCO_CUDA_TRY(cudaGetLastError());
}
}  // namespace cuco::detail
