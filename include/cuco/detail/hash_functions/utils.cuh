/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuda/std/cstddef>

namespace cuco::detail {

template <typename T, typename U, typename Extent>
constexpr __host__ __device__ T load_chunk(U const* const data, Extent index) noexcept
{
  auto const bytes = reinterpret_cast<cuda::std::byte const*>(data);
  T chunk;
  memcpy(&chunk, bytes + index * sizeof(T), sizeof(T));
  return chunk;
}

constexpr __host__ __device__ std::uint32_t rotl32(std::uint32_t x, std::int8_t r) noexcept
{
  return (x << r) | (x >> (32 - r));
}

constexpr __host__ __device__ std::uint64_t rotl64(std::uint64_t x, std::int8_t r) noexcept
{
  return (x << r) | (x >> (64 - r));
}

};  // namespace cuco::detail
