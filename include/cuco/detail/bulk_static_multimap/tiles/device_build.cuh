/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "block.cuh"

namespace cuco::detail::bulk_tiles {

struct SingleBuildPolicy {
  int prefetch_items     = 2;
  int blocks_per_sm      = 8;
  bool fuse_hash_count   = true;
  bool shared_lrb_counts = true;
};

// All scratch is caller-owned and may be released after the build completes.
struct SingleBuildWorkspace {
  void* scan        = nullptr;
  size_t scan_bytes = 0;
};

template <typename Record, typename Offset, typename Counter, typename Bucket>
cudaError_t single_lrb_workspace_bytes(
  TileLrbBuildStorage<Record, Offset, Counter, Bucket> const& storage, size_t* bytes);
template <typename Record,
          typename Hash    = cuco::default_hash_function<RecordKey<Record>>,
          typename Offset  = int32_t,
          typename Counter = Offset,
          typename Bucket  = uint32_t>
cudaError_t single_lrb_build(TileLrbBuildStorage<Record, Offset, Counter, Bucket> const& storage,
                             SingleBuildWorkspace workspace,
                             SingleBuildPolicy policy  = {},
                             cudaStream_t stream       = nullptr,
                             cudaEvent_t* phase_events = nullptr,
                             Hash hash                 = {});

}  // namespace cuco::detail::bulk_tiles

#include "device_build_impl.cuh"
