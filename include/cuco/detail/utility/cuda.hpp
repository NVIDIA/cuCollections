/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/error.hpp>
#include <cuco/detail/utility/cuda.cuh>

#include <cuda/stream_ref>

#include <cstdint>

namespace cuco {
namespace detail {

constexpr std::int32_t default_block_size() noexcept { return 128; }
/// Default stride
constexpr std::int32_t default_stride() noexcept { return 1; }

/**
 * @brief Computes the desired 1D grid size with the given parameters
 *
 * @param num Number of elements to handle in the kernel
 * @param cg_size Number of threads per CUDA Cooperative Group
 * @param stride Number of elements to be handled by each thread
 * @param block_size Number of threads in each thread block
 *
 * @return The resulting grid size
 */
constexpr auto grid_size(index_type num,
                         std::int32_t cg_size    = 1,
                         std::int32_t stride     = default_stride(),
                         std::int32_t block_size = default_block_size()) noexcept
{
  return (cg_size * num + stride * block_size - 1) / (stride * block_size);
}

/**
 * @brief Returns the number of SMs that a kernel on `stream` can use
 *
 * For a stream of a green context, this is the SM count of the green context. For other streams,
 * it is the SM count of the device. The stream query needs CUDA 13.1. With an older toolkit or
 * driver, the function returns the SM count of the current device.
 *
 * @param stream CUDA stream of the kernel launch
 *
 * @return Number of SMs
 */
inline std::int32_t num_multiprocessors([[maybe_unused]] cuda::stream_ref stream)
{
#if CUDART_VERSION >= 13010
  cudaDevResource resource{};
  if (cudaStreamGetDevResource(stream.get(), &resource, cudaDevResourceTypeSm) == cudaSuccess) {
    return static_cast<std::int32_t>(resource.sm.smCount);
  }
  // The driver does not have the query. Clear the error and use the device attribute.
  static_cast<void>(cudaGetLastError());
#endif
  int device = 0;
  CUCO_CUDA_TRY(cudaGetDevice(&device));

  int num_sms = -1;
  CUCO_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  return num_sms;
}

/**
 * @brief Computes the ideal 1D grid size for a kernel launch on `stream`
 *
 * @tparam Kernel Kernel type
 *
 * @param block_size Number of threads in each thread block
 * @param kernel CUDA kernel to launch
 * @param stream CUDA stream of the kernel launch
 * @param dynamic_shm_size Dynamic shared memory size
 *
 * @return The grid size that gives the highest occupancy on the SMs that `stream` can use
 */
template <typename Kernel>
auto max_occupancy_grid_size(std::int32_t block_size,
                             Kernel kernel,
                             cuda::stream_ref stream,
                             std::size_t dynamic_shm_size = 0)
{
  int max_active_blocks_per_multiprocessor{};
  CUCO_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks_per_multiprocessor, kernel, block_size, dynamic_shm_size));

  return max_active_blocks_per_multiprocessor * num_multiprocessors(stream);
}

}  // namespace detail
}  // namespace cuco
