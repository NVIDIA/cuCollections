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

#include <cuco/detail/error.hpp>
#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuco/utility/traits.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/span>
#include <cuda/stream_ref>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include <nv/target>

namespace cuco::detail {

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
};

// TODO implement roaring_bitmap_metadata<cuda::std::uint64_t>

// primary template
template <class T, cuda::thread_scope Scope>
class roaring_bitmap_impl {
  static_assert(cuco::dependent_false<T>, "T must be either uint32_t or uint64_t");
};

template <cuda::thread_scope Scope>
class roaring_bitmap_impl<cuda::std::uint32_t, Scope> {
  // Constants from the Roaring format spec
  static constexpr cuda::std::uint32_t serial_cookie_no_runcontainer = 12346;
  static constexpr cuda::std::uint32_t serial_cookie                 = 12347;
  static constexpr cuda::std::uint32_t frozen_cookie                 = 13766;
  static constexpr cuda::std::int32_t no_offset_threshold            = 4;
  static constexpr cuda::std::uint32_t binary_search_threshold = 8;  // TODO determine optimal value

 public:
  using metadata_type                = roaring_bitmap_metadata<cuda::std::uint32_t>;
  static constexpr auto thread_scope = Scope;

  __host__ __device__ roaring_bitmap_impl(cuda::std::byte const* bitmap,
                                          metadata_type metadata,
                                          cuda_thread_scope<Scope> /* scope */)
  {
    NV_IF_TARGET(
      NV_IS_HOST,
      CUCO_EXPECTS(metadata.valid, "Invalid bitmap format");)  // TODO device error handling

    if (metadata.valid) {
      data_           = cuda::std::span<cuda::std::byte const>{bitmap, metadata.size_bytes};
      size_           = metadata.num_keys;
      num_containers_ = metadata.num_containers;
      run_container_bitmap_ =
        reinterpret_cast<cuda::std::uint8_t const*>(bitmap + metadata.run_container_bitmap);
      key_cards_ = reinterpret_cast<cuda::std::uint16_t const*>(bitmap + metadata.key_cards);
      offsets_   = reinterpret_cast<cuda::std::byte const*>(bitmap + metadata.container_offsets);
      offsets_aligned_ = metadata.offsets_aligned;
      has_run_         = metadata.has_run;
    }
  }

  __device__ roaring_bitmap_impl(cuda::std::byte const* bitmap, cuda_thread_scope<Scope> scope)
    : roaring_bitmap_impl(bitmap, read_metadata(bitmap), scope)
  {
  }

  template <class InputIt, class OutputIt>
  __host__ void contains(InputIt first,
                         InputIt last,
                         OutputIt contained,
                         cuda::stream_ref stream = {}) const
  {
    this->contains_async(first, last, contained, stream);
    stream.wait();
  }

  template <class InputIt, class OutputIt>
  __host__ void contains_async(InputIt first,
                               InputIt last,
                               OutputIt contained,
                               cuda::stream_ref stream = {}) const noexcept
  {
    auto nosync_exec_policy = thrust::cuda::par_nosync.on(stream.get());
    if (this->empty()) {
      thrust::fill(
        nosync_exec_policy, contained, contained + cuda::std::distance(first, last), false);
    } else {
      thrust::transform(nosync_exec_policy,
                        first,
                        last,
                        contained,
                        cuda::proclaim_return_type<bool>(
                          [*this] __device__(auto key) { return this->contains(key); }));
    }
  }

  __device__ bool contains(cuda::std::uint32_t value) const
  {
    cuda::std::uint16_t upper = value >> 16;
    cuda::std::uint16_t lower = value & 0xFFFF;

    // Binary search on key_cards_ to find container with matching upper key
    cuda::std::uint32_t left  = 0;
    cuda::std::uint32_t right = num_containers_;

    if (num_containers_ < binary_search_threshold) {
      for (cuda::std::uint32_t i = 0; i < num_containers_; i++) {
        if (key_cards_[i * 2] == upper) { return this->contains_container(lower, i); }
      }
    } else {
      while (left < right) {
        cuda::std::uint32_t mid     = left + (right - left) / 2;
        cuda::std::uint16_t mid_key = key_cards_[mid * 2];

        if (mid_key == upper) {
          return this->contains_container(lower, mid);
        } else if (mid_key < upper) {
          left = mid + 1;
        } else {
          right = mid;
        }
      }
    }
    return false;
  }

  [[nodiscard]] __host__ __device__ cuda::std::size_t size() const noexcept { return size_; }

  [[nodiscard]] __host__ __device__ bool empty() const noexcept { return size_ == 0; }

  [[nodiscard]] __host__ __device__ cuda::std::span<cuda::std::byte const> data() const noexcept
  {
    return data_;
  }

  __host__ __device__ static metadata_type const read_metadata(
    cuda::std::byte const* bitmap) noexcept
  {
    cuda::std::byte const* buf = bitmap;
    metadata_type metadata;

    cuda::std::uint32_t cookie;
    cuda::std::memcpy(&cookie, buf, sizeof(cuda::std::uint32_t));
    buf += sizeof(cuda::std::uint32_t);
    if ((cookie & 0xFFFF) != serial_cookie && cookie != serial_cookie_no_runcontainer) {
      metadata.valid = false;
      return metadata;
    }

    if ((cookie & 0xFFFF) == serial_cookie)
      metadata.num_containers = (cookie >> 16) + 1;
    else {
      cuda::std::memcpy(&metadata.num_containers, buf, sizeof(cuda::std::uint32_t));
      buf += sizeof(cuda::std::uint32_t);
    }
    if (metadata.num_containers < 0) {
      metadata.valid = false;
      return metadata;
    }
    if (metadata.num_containers > (1 << 16)) {
      metadata.valid = false;
      return metadata;
    }

    metadata.has_run = (cookie & 0xFFFF) == serial_cookie;
    if (metadata.has_run) {
      metadata.valid = false;
      return metadata;  // TODO run container bitmap is not supported yet
      cuda::std::size_t s           = (metadata.num_containers + 7) / 8;
      metadata.run_container_bitmap = cuda::std::distance(bitmap, buf);
      buf += s;
    }

    metadata.key_cards = cuda::std::distance(bitmap, buf);
    buf += metadata.num_containers * 2 * sizeof(cuda::std::uint16_t);

    if ((!metadata.has_run) || (metadata.num_containers >= no_offset_threshold)) {
      metadata.container_offsets = cuda::std::distance(bitmap, buf);
      metadata.offsets_aligned =
        (reinterpret_cast<cuda::std::uintptr_t>(bitmap + metadata.container_offsets) %
         sizeof(cuda::std::uint32_t)) == 0;
      buf += metadata.num_containers * 4;
    }

    metadata.num_keys = 0;
    cuda::std::uint16_t const* key_cards =
      reinterpret_cast<cuda::std::uint16_t const*>(bitmap + metadata.key_cards);
    cuda::std::uint32_t card = 0;
    for (cuda::std::int32_t i = 0; i < metadata.num_containers; i++) {
      // cuda::std::uint16_t key  = key_cards[i * 2];
      card = key_cards[i * 2 + 1] + 1;
      metadata.num_keys += card;
    }

    // find end of roaring bitmap
    cuda::std::byte const* end = bitmap + container_offset(bitmap + metadata.container_offsets,
                                                           metadata.offsets_aligned,
                                                           metadata.num_containers - 1);
    if (is_run_container(
          reinterpret_cast<cuda::std::uint8_t const*>(bitmap + metadata.run_container_bitmap),
          metadata.has_run,
          metadata.num_containers - 1)) {
      // TODO implement
    } else {
      if (card <= 4096) {  // TODO check if this is correct
        end += card * sizeof(cuda::std::uint16_t);
      } else {
        end += 8192;  // fixed size bitset container
      }
    }

    metadata.size_bytes = static_cast<cuda::std::size_t>(cuda::std::distance(bitmap, end));
    metadata.valid      = true;
    return metadata;
  }

 private:
  __host__ __device__ static bool is_run_container(cuda::std::uint8_t const* run_container_bitmap,
                                                   bool has_run,
                                                   cuda::std::int32_t i)
  {
    if (not has_run) return false;
    return run_container_bitmap[i / 8] & (1 << (i % 8));
  }

  __device__ bool contains_container(cuda::std::uint16_t lower, cuda::std::uint32_t index) const
  {
    cuda::std::uint32_t card             = key_cards_[index * 2 + 1] + 1;
    cuda::std::uint16_t const* container = reinterpret_cast<cuda::std::uint16_t const*>(
      data_.data() + container_offset(offsets_, offsets_aligned_, index));
    if (is_run_container(run_container_bitmap_, has_run_, index)) {
      return this->contains_run_container(container, lower, card);
    } else {
      if (card <= 4096) {  // TODO check if this is correct
        return this->contains_array_container(container, lower, card);
      } else {
        return this->contains_bitset_container(container, lower, card);
      }
    }
  }

  __device__ bool contains_array_container(cuda::std::uint16_t const* container,
                                           cuda::std::uint16_t lower,
                                           cuda::std::uint32_t card) const
  {
    // Use linear search for small arrays, binary search for larger ones
    if (card < binary_search_threshold) {
      for (cuda::std::uint32_t i = 0; i < card; i++) {
        if (container[i] == lower) { return true; }
      }
      return false;
    } else {
      cuda::std::uint32_t left  = 0;
      cuda::std::uint32_t right = card;

      while (left < right) {
        cuda::std::uint32_t mid = left + (right - left) / 2;
        if (container[mid] == lower) {
          return true;
        } else if (container[mid] < lower) {
          left = mid + 1;
        } else {
          right = mid;
        }
      }
      return false;
    }
  }

  __device__ bool contains_bitset_container(cuda::std::uint16_t const* container,
                                            cuda::std::uint16_t lower,
                                            cuda::std::uint32_t card) const
  {
    // check if bit at position lower is set
    return container[lower / 16] & (1 << (lower % 16));
  }

  __device__ bool contains_run_container(cuda::std::uint16_t const* container,
                                         cuda::std::uint16_t lower,
                                         cuda::std::uint32_t card) const
  {
    // TODO implement
    return false;
  }

  __host__ __device__ static cuda::std::uint32_t container_offset(cuda::std::byte const* offsets,
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

  cuda::std::span<cuda::std::byte const> data_;
  cuda::std::size_t size_;
  cuda::std::int32_t num_containers_;
  cuda::std::uint8_t const* run_container_bitmap_;
  cuda::std::uint16_t const* key_cards_;  // TODO uint8?
  cuda::std::byte const* offsets_;
  bool offsets_aligned_;
  bool has_run_;
};

template <cuda::thread_scope Scope>
class roaring_bitmap_impl<cuda::std::uint64_t, Scope> {
  using bucket_type = roaring_bitmap_impl<cuda::std::uint32_t, Scope>;
  // TODO implement
};

}  // namespace cuco::detail