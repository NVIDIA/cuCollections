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
#include <cuco/detail/roaring_bitmap/roaring_bitmap_storage.cuh>
#include <cuco/detail/roaring_bitmap/util.cuh>
#include <cuco/utility/traits.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/stream_ref>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

namespace cuco::detail {

// primary template
template <class T>
class roaring_bitmap_impl {
  static_assert(cuco::dependent_false<T>, "T must be either uint32_t or uint64_t");
};

template <>
class roaring_bitmap_impl<cuda::std::uint32_t> {
 public:
  using storage_ref_type = roaring_bitmap_storage_ref<cuda::std::uint32_t>;

  static constexpr cuda::std::uint32_t binary_search_threshold = 8;  // TODO determine optimal value

  __host__ __device__ roaring_bitmap_impl(storage_ref_type const& storage_ref)
  {
    auto const& meta = storage_ref.metadata();
    if (meta.valid) {
      data_           = storage_ref.data();
      size_bytes_     = meta.size_bytes;
      size_           = meta.num_keys;
      num_containers_ = meta.num_containers;
      run_container_bitmap_ =
        reinterpret_cast<cuda::std::uint8_t const*>(storage_ref.data() + meta.run_container_bitmap);
      key_cards_ =
        reinterpret_cast<cuda::std::uint16_t const*>(storage_ref.data() + meta.key_cards);
      offsets_ =
        reinterpret_cast<cuda::std::byte const*>(storage_ref.data() + meta.container_offsets);
      offsets_aligned_ = meta.offsets_aligned;
      has_run_         = meta.has_run;
    }
  }

  __device__ roaring_bitmap_impl(cuda::std::byte const* bitmap)
    : roaring_bitmap_impl{storage_ref_type{bitmap}}
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
    cuda::std::uint16_t const upper = value >> 16;
    cuda::std::uint16_t const lower = value & 0xFFFF;

    if (num_containers_ < binary_search_threshold) {
// linear search
#pragma unroll
      for (cuda::std::uint32_t i = 0; i < num_containers_; i++) {
        cuda::std::uint16_t const key = key_cards_[i * 2];
        if (key == upper) { return this->contains_container(lower, i); }
        if (key > upper) { return false; }
      }
    } else {
      // binary search
      cuda::std::uint32_t left  = 0;
      cuda::std::uint32_t right = num_containers_;
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

  [[nodiscard]] __host__ __device__ cuda::std::byte const* data() const noexcept { return data_; }

  [[nodiscard]] __host__ __device__ cuda::std::size_t size_bytes() const noexcept
  {
    return size_bytes_;
  }

 private:
  __device__ bool contains_container(cuda::std::uint16_t lower, cuda::std::uint32_t index) const
  {
    cuda::std::uint32_t card             = key_cards_[index * 2 + 1] + 1;
    cuda::std::uint16_t const* container = reinterpret_cast<cuda::std::uint16_t const*>(
      data_ + container_offset(offsets_, offsets_aligned_, index));
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
    // TODO implement linear search
    return false;
  }

  cuda::std::byte const* data_;
  cuda::std::size_t size_bytes_;
  cuda::std::size_t size_;
  cuda::std::int32_t num_containers_;
  cuda::std::uint8_t const* run_container_bitmap_;
  cuda::std::uint16_t const* key_cards_;  // TODO uint8?
  cuda::std::byte const* offsets_;
  bool offsets_aligned_;
  bool has_run_;
};

template <>
class roaring_bitmap_impl<cuda::std::uint64_t> {
  using bucket_type = roaring_bitmap_impl<cuda::std::uint32_t>;
  // TODO implement
};

}  // namespace cuco::detail