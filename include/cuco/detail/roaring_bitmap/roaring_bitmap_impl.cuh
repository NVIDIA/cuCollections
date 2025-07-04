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

namespace cuco::detail {

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

 public:
  static constexpr auto thread_scope = Scope;

  __host__ roaring_bitmap_impl(cuda::std::span<cuda::std::byte const> compressed_bitmap_h,
                               cuda::std::span<cuda::std::byte const> compressed_bitmap_d,
                               cuda_thread_scope<Scope> /* scope */)
    : data_{compressed_bitmap_d}
  {
    bool success = this->read_header(compressed_bitmap_h);
    CUCO_EXPECTS(success, "Failed to read compressed bitmap");
  }

  __device__ roaring_bitmap_impl(cuda::std::span<cuda::std::byte const> compressed_bitmap,
                                 cuda_thread_scope<Scope> /* scope */)
    : data_{compressed_bitmap}
  {
    this->read_header(compressed_bitmap);  // TODO error handling
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

    // TODO binary search on key_cards_
    for (cuda::std::int32_t i = 0; i < num_containers_; i++) {
      if (key_cards_[i * 2] == upper) {
        cuda::std::uint32_t card = key_cards_[i * 2 + 1] + 1;
        cuda::std::uint16_t const* container =
          reinterpret_cast<cuda::std::uint16_t const*>(data_.data() + this->container_offset(i));
        if (this->is_run_container(i)) {
          return this->contains_run_container(container, lower, card);
        } else {
          if (card <= 4096) {  // TODO check if this is correct
            return this->contains_array_container(container, lower, card);
          } else {
            return this->contains_bitset_container(container, lower, card);
          }
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

 private:
  __device__ bool is_run_container(cuda::std::int32_t i) const
  {
    if (not has_run_) return false;
    return run_container_bitmap_[i / 8] & (1 << (i % 8));
  }

  __device__ bool contains_array_container(cuda::std::uint16_t const* container,
                                           cuda::std::uint16_t lower,
                                           cuda::std::uint32_t card) const
  {
    // TODO binary search on container
    // if (card < 256) -> linear search
    for (cuda::std::uint32_t i = 0; i < card; i++) {
      if (container[i] == lower) { return true; }
    }
    return false;
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

  __device__ cuda::std::uint32_t container_offset(cuda::std::int32_t i) const
  {
    cuda::std::uint32_t offset;
    cuda::std::memcpy(
      &offset, offsets_ + i * sizeof(cuda::std::uint32_t), sizeof(cuda::std::uint32_t));
    return offset;
  }

  __host__ __device__ bool read_header(cuda::std::span<cuda::std::byte const> compressed_bitmap)
  {
    cuda::std::size_t length                     = compressed_bitmap.size();
    cuda::std::byte const* buf                   = compressed_bitmap.data();
    [[maybe_unused]] cuda::std::size_t readbytes = 0;

    // cookie and num_containers
    if (length < 4) {
      // printf("length is less than 4\n");
      return false;
    }

    cuda::std::uint32_t cookie;
    cuda::std::memcpy(&cookie, buf, sizeof(cuda::std::uint32_t));
    readbytes += sizeof(cuda::std::uint32_t);
    buf += sizeof(cuda::std::uint32_t);
    if ((cookie & 0xFFFF) != serial_cookie && cookie != serial_cookie_no_runcontainer) {
      // printf("cookie is not serial cookie or serial cookie no runcontainer\n");
      return false;
    }

    if ((cookie & 0xFFFF) == serial_cookie)
      num_containers_ = (cookie >> 16) + 1;
    else {
      readbytes += sizeof(cuda::std::uint32_t);
      if (readbytes > length) {
        // printf("readbytes is greater than length\n");
        return false;
      }
      cuda::std::memcpy(&num_containers_, buf, sizeof(cuda::std::uint32_t));
      buf += sizeof(cuda::std::uint32_t);
    }
    if (num_containers_ < 0) {
      // printf("num_containers_ is less than 0\n");
      return false;
    }
    if (num_containers_ > (1 << 16)) {
      // printf("num_containers_ is greater than 65536\n");
      return false;
    }
    // printf("num_containers_: %d\n", num_containers_);

    has_run_ = (cookie & 0xFFFF) == serial_cookie;
    if (has_run_) {
      cuda::std::size_t s = (num_containers_ + 7) / 8;
      readbytes += s;
      if (readbytes > length) {
        // printf("readbytes is greater than length\n");
        return false;
      }
      run_container_bitmap_ = reinterpret_cast<cuda::std::uint8_t const*>(buf);
      buf += s;
    }
    // printf("has_run: %d\n", has_run_);

    key_cards_ = reinterpret_cast<cuda::std::uint16_t const*>(buf);
    readbytes += num_containers_ * 2 * sizeof(cuda::std::uint16_t);
    if (readbytes > length) {
      // printf("readbytes is greater than length\n");
      return false;
    }
    buf += num_containers_ * 2 * sizeof(cuda::std::uint16_t);

    if ((!has_run_) || (num_containers_ >= no_offset_threshold)) {
      readbytes += num_containers_ * 4;
      if (readbytes > length) {
        // printf("readbytes is greater than length\n");
        return false;
      }
      offsets_ = buf;
      buf += num_containers_ * 4;
    }

    readbytes += num_containers_ * 4;
    if (readbytes > length) {
      // printf("readbytes is greater than length\n");
      return false;
    }

    size_ = 0;
    for (cuda::std::int32_t i = 0; i < num_containers_; i++) {
      // cuda::std::uint16_t key  = key_cards_[i * 2];
      cuda::std::uint32_t card = key_cards_[i * 2 + 1] + 1;
      size_ += card;
      // printf("key: %d, card: %d\n", key, card);
    }

    return true;
  }

  cuda::std::span<cuda::std::byte const> data_;
  cuda::std::size_t size_;
  cuda::std::int32_t num_containers_;
  cuda::std::uint8_t const* run_container_bitmap_;
  cuda::std::uint16_t const* key_cards_;
  cuda::std::byte const* offsets_;
  bool has_run_;
};

template <cuda::thread_scope Scope>
class roaring_bitmap_impl<cuda::std::uint64_t, Scope> {
  using bucket_type = roaring_bitmap_impl<cuda::std::uint32_t, Scope>;
  // TODO implement
};

}  // namespace cuco::detail