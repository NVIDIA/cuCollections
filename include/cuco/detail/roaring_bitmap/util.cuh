/*
 * Copyright (c) 2025 NVIDIA CORPORATION.
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

#include <cuco/utility/traits.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/iterator>

#include <nv/target>

namespace cuco::detail {

__host__ __device__ cuda::std::uint32_t container_offset(cuda::std::byte const* offsets,
                                                         bool offsets_aligned,
                                                         cuda::std::int32_t i)
{
  cuda::std::uint32_t offset = 0;
  if (offsets_aligned) {
    offset =
      *reinterpret_cast<cuda::std::uint32_t const*>(offsets + i * sizeof(cuda::std::uint32_t));
  } else {
    cuda::std::memcpy(
      &offset, offsets + i * sizeof(cuda::std::uint32_t), sizeof(cuda::std::uint32_t));
  }
  return offset;
}

__host__ __device__ bool is_run_container(cuda::std::uint8_t const* run_container_bitmap,
                                          bool has_run,
                                          cuda::std::int32_t i)
{
  if (not has_run) return false;
  return run_container_bitmap[i / 8] & (1 << (i % 8));
}

template <class T>
struct roaring_bitmap_metadata {
  static_assert(cuco::dependent_false<T>, "T must be either uint32_t or uint64_t");
};

template <>
struct roaring_bitmap_metadata<cuda::std::uint32_t> {
  cuda::std::size_t size_bytes           = 0;
  cuda::std::size_t num_keys             = 0;
  cuda::std::size_t run_container_bitmap = 0;
  cuda::std::size_t key_cards            = 0;
  cuda::std::size_t container_offsets    = 0;
  cuda::std::int32_t num_containers      = 0;
  bool has_run                           = false;
  bool offsets_aligned                   = false;
  bool valid                             = false;

  __host__ __device__ roaring_bitmap_metadata(cuda::std::byte const* bitmap)
  {
    constexpr cuda::std::uint32_t serial_cookie_no_runcontainer = 12346;
    constexpr cuda::std::uint32_t serial_cookie                 = 12347;
    // constexpr cuda::std::uint32_t frozen_cookie                 = 13766;
    constexpr cuda::std::int32_t no_offset_threshold = 4;

    cuda::std::byte const* buf = bitmap;

    cuda::std::uint32_t cookie;
    cuda::std::memcpy(&cookie, buf, sizeof(cuda::std::uint32_t));
    buf += sizeof(cuda::std::uint32_t);
    if ((cookie & 0xFFFF) != serial_cookie && cookie != serial_cookie_no_runcontainer) {
      valid = false;
      NV_IF_TARGET(NV_IS_HOST,
                   CUCO_FAIL("Invalid bitmap format");)  // TODO device error handling
      return;
    }

    if ((cookie & 0xFFFF) == serial_cookie)
      num_containers = (cookie >> 16) + 1;
    else {
      cuda::std::memcpy(&num_containers, buf, sizeof(cuda::std::uint32_t));
      buf += sizeof(cuda::std::uint32_t);
    }
    if (num_containers < 0) {
      valid = false;
      NV_IF_TARGET(NV_IS_HOST,
                   CUCO_FAIL("Invalid bitmap format");)  // TODO device error handling
      return;
    }
    if (num_containers > (1 << 16)) {
      valid = false;
      NV_IF_TARGET(NV_IS_HOST,
                   CUCO_FAIL("Invalid bitmap format");)  // TODO device error handling
      return;
    }

    has_run = (cookie & 0xFFFF) == serial_cookie;
    if (has_run) {
      valid = false;  // TODO run container bitmap is not supported yet
      NV_IF_TARGET(NV_IS_HOST,
                   CUCO_FAIL("Invalid bitmap format");)  // TODO device error handling
      return;
      cuda::std::size_t s  = (num_containers + 7) / 8;
      run_container_bitmap = cuda::std::distance(bitmap, buf);
      buf += s;
    }

    key_cards = cuda::std::distance(bitmap, buf);
    buf += num_containers * 2 * sizeof(cuda::std::uint16_t);

    if ((!has_run) || (num_containers >= no_offset_threshold)) {
      container_offsets = cuda::std::distance(bitmap, buf);
      offsets_aligned   = (reinterpret_cast<cuda::std::uintptr_t>(bitmap + container_offsets) %
                         sizeof(cuda::std::uint32_t)) == 0;
      buf += num_containers * 4;
    }

    num_keys = 0;
    cuda::std::uint16_t const* cards =
      reinterpret_cast<cuda::std::uint16_t const*>(bitmap + key_cards);
    cuda::std::uint32_t card = 0;
    for (cuda::std::int32_t i = 0; i < num_containers; i++) {
      // cuda::std::uint16_t key  = key_cards[i * 2];
      card = cards[i * 2 + 1] + 1;
      num_keys += card;
    }

    // find end of roaring bitmap
    cuda::std::byte const* end =
      bitmap + container_offset(bitmap + container_offsets, offsets_aligned, num_containers - 1);
    if (is_run_container(reinterpret_cast<cuda::std::uint8_t const*>(bitmap + run_container_bitmap),
                         has_run,
                         num_containers - 1)) {
      // TODO implement
    } else {
      if (card <= 4096) {  // TODO check if this is correct
        end += card * sizeof(cuda::std::uint16_t);
      } else {
        end += 8192;  // fixed size bitset container
      }
    }

    size_bytes = static_cast<cuda::std::size_t>(cuda::std::distance(bitmap, end));
    valid      = true;
  }
};

// TODO implement roaring_bitmap_metadata<cuda::std::uint64_t>

}  // namespace cuco::detail