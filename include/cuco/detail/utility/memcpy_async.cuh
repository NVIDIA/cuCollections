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

namespace cuco::detail {

/**
 * @brief Asynchronous memory copy utility that works around cudaMemcpyAsync bugs
 *
 * This function provides a drop-in replacement for cudaMemcpyAsync that uses
 * cudaMemcpyBatchAsync internally to work around known issues with cudaMemcpyAsync
 * when available (CUDA 12.8+). For older CUDA versions, it falls back to the
 * original cudaMemcpyAsync. The function automatically handles the different API
 * signatures between CUDA runtime versions.
 *
 * @param dst Destination memory address
 * @param src Source memory address
 * @param count Number of bytes to copy
 * @param kind Type of memory copy (cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, etc.)
 * @param stream CUDA stream for the asynchronous operation
 */
inline void memcpy_async(
  void* dst, const void* src, size_t count, cudaMemcpyKind kind, cuda::stream_ref stream)
{
#if CUDART_VERSION >= 12080
  // CUDA 12.8+ - Use cudaMemcpyBatchAsync as a workaround for cudaMemcpyAsync bugs
  void* dsts[1]                 = {dst};
  void* srcs[1]                 = {const_cast<void*>(src)};
  size_t sizes[1]               = {count};
  cudaMemcpyAttributes attrs[1] = {{.srcAccessOrder = cudaMemcpySrcAccessOrderStream}};
  size_t attrsIdxs[1]           = {0};

#if CUDART_VERSION >= 13000
  // CUDA 13.0+ API - no failIdx parameter
  CUCO_CUDA_TRY(cudaMemcpyBatchAsync(dsts, srcs, sizes, 1, attrs, attrsIdxs, 1, stream.get()));
#else
  // CUDA 12.8-12.x API - requires failIdx parameter
  size_t failIdx;
  CUCO_CUDA_TRY(
    cudaMemcpyBatchAsync(dsts, srcs, sizes, 1, attrs, attrsIdxs, 1, &failIdx, stream.get()));
#endif

#else
  // CUDA 12.0-12.7 - Fall back to original cudaMemcpyAsync
  // Note: This may still have the original bugs that cudaMemcpyBatchAsync was designed to fix
  CUCO_CUDA_TRY(cudaMemcpyAsync(dst, src, count, kind, stream.get()));
#endif
}

}  // namespace cuco::detail
