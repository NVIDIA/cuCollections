/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

namespace cuco::detail {

template <typename T, typename U, typename Extent>
constexpr __host__ __device__ T load_chunk(U const* const data, Extent index) noexcept
{
  auto const bytes = reinterpret_cast<cuda::std::byte const*>(data);
  T chunk;
  memcpy(&chunk, bytes + index * sizeof(T), sizeof(T));
  return chunk;
}

constexpr __host__ __device__ cuda::std::uint32_t rotl32(cuda::std::uint32_t x,
                                                         cuda::std::int8_t r) noexcept
{
  return (x << r) | (x >> (32 - r));
}

constexpr __host__ __device__ cuda::std::uint64_t rotl64(cuda::std::uint64_t x,
                                                         cuda::std::int8_t r) noexcept
{
  return (x << r) | (x >> (64 - r));
}

};  // namespace cuco::detail
