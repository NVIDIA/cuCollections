/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include <cstddef>

namespace cuco::detail {

/**
 * @brief Asynchronous memory copy utility using cudaMemcpyBatchAsync when possible
 *
 * Uses cudaMemcpyBatchAsync for CUDA 13.0+ to avoid driver-side locking overhead.
 * Falls back to cudaMemcpyAsync for older CUDA versions or edge cases.
 *
 * @param dst Destination memory address
 * @param src Source memory address
 * @param count Number of bytes to copy
 * @param kind Memory copy direction
 * @param stream CUDA stream for the operation
 * @return cudaError_t Error code from the memory copy operation
 */
[[nodiscard]] inline cudaError_t memcpy_async(
  void* dst, void const* src, size_t count, cudaMemcpyKind kind, cuda::stream_ref stream)
{
  if (dst == nullptr || src == nullptr || count == 0) { return cudaSuccess; }

#if CUDART_VERSION >= 13000
  if (stream.get() == nullptr) { return cudaMemcpyAsync(dst, src, count, kind, stream.get()); }

  void* dsts[1]             = {dst};
  void* srcs[1]             = {const_cast<void*>(src)};
  std::size_t sizes[1]      = {count};
  std::size_t attrs_idxs[1] = {0};

  cudaMemcpyAttributes attrs[1] = {};
  attrs[0].srcAccessOrder       = cudaMemcpySrcAccessOrderStream;
  attrs[0].flags                = cudaMemcpyFlagPreferOverlapWithCompute;

  return cudaMemcpyBatchAsync(dsts, srcs, sizes, 1, attrs, attrs_idxs, 1, stream.get());
#else
  // CUDA < 13.0 - use regular cudaMemcpyAsync
  return cudaMemcpyAsync(dst, src, count, kind, stream.get());
#endif  // CUDART_VERSION >= 13000
}

}  // namespace cuco::detail
