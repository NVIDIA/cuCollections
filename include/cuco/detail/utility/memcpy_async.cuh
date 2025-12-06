/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuco/detail/error.hpp>

#include <cuda/stream_ref>

#include <cstring>

namespace cuco::detail {

/**
 * @brief Asynchronous memory copy utility using cudaMemcpyBatchAsync when possible
 *
 * Uses cudaMemcpyBatchAsync for CUDA 12.8+ with proper edge case handling.
 * Falls back to cudaMemcpyAsync for older CUDA versions or edge cases.
 *
 * @param dst Destination memory address
 * @param src Source memory address
 * @param count Number of bytes to copy
 * @param kind Memory copy direction
 * @param stream CUDA stream for the operation
 */
inline void memcpy_async(
  void* dst, void const* src, size_t count, cudaMemcpyKind kind, cuda::stream_ref stream)
{
  if (dst == nullptr || src == nullptr || count == 0) { return; }

#if CUDART_VERSION >= 12080
  if (stream.get() == 0) {
    CUCO_CUDA_TRY(cudaMemcpyAsync(dst, src, count, kind, stream.get()));
    return;
  }

  void* dsts[1]             = {dst};
  void* srcs[1]             = {const_cast<void*>(src)};
  std::size_t sizes[1]      = {count};
  std::size_t attrs_idxs[1] = {0};

  cudaMemcpyAttributes attrs[1] = {};
  attrs[0].srcAccessOrder       = cudaMemcpySrcAccessOrderStream;
  attrs[0].flags                = cudaMemcpyFlagPreferOverlapWithCompute;

#if CUDART_VERSION >= 13000
  CUCO_CUDA_TRY(cudaMemcpyBatchAsync(dsts, srcs, sizes, 1, attrs, attrs_idxs, 1, stream.get()));
#else
  std::size_t fail_idx;
  CUCO_CUDA_TRY(
    cudaMemcpyBatchAsync(dsts, srcs, sizes, 1, attrs, attrs_idxs, 1, &fail_idx, stream.get()));
#endif  // CUDART_VERSION >= 13000
#else
  // CUDA < 12.8 - use regular cudaMemcpyAsync
  CUCO_CUDA_TRY(cudaMemcpyAsync(dst, src, count, kind, stream.get()));
#endif  // CUDART_VERSION >= 12080
}
}  // namespace cuco::detail
