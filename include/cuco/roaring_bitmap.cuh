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

#include <cuco/detail/roaring_bitmap/roaring_bitmap_storage.cuh>
#include <cuco/roaring_bitmap_ref.cuh>
#include <cuco/utility/allocator.hpp>

#include <cuda/std/cstddef>
#include <cuda/stream_ref>

namespace cuco {

template <class T, class Allocator = cuco::cuda_allocator<cuda::std::byte>>
class roaring_bitmap {
 public:
  using storage_type   = detail::roaring_bitmap_storage<T, Allocator>;
  using allocator_type = typename storage_type::allocator_type;
  using ref_type       = roaring_bitmap_ref<T>;

  roaring_bitmap(cuda::std::byte const* bitmap,
                 Allocator const& alloc  = {},
                 cuda::stream_ref stream = {});

  roaring_bitmap(roaring_bitmap const& other)            = default;
  roaring_bitmap(roaring_bitmap&& other)                 = default;
  roaring_bitmap& operator=(roaring_bitmap const& other) = default;
  roaring_bitmap& operator=(roaring_bitmap&& other)      = default;

  ~roaring_bitmap() = default;

  template <class InputIt, class OutputIt>
  void contains(InputIt first,
                InputIt last,
                OutputIt contained,
                cuda::stream_ref stream = {}) const;

  template <class InputIt, class OutputIt>
  void contains_async(InputIt first,
                      InputIt last,
                      OutputIt contained,
                      cuda::stream_ref stream = {}) const noexcept;

  // TODO contains_if, contains_if_async, empty

  [[nodiscard]] cuda::std::size_t size() const noexcept;

  [[nodiscard]] bool empty() const noexcept;

  [[nodiscard]] cuda::std::byte const* data() const noexcept;

  [[nodiscard]] cuda::std::size_t size_bytes() const noexcept;

  [[nodiscard]] allocator_type allocator() const noexcept;

  [[nodiscard]] ref_type ref() const noexcept;

 private:
  storage_type storage_;
  ref_type ref_;
};

}  // namespace cuco

#include <cuco/detail/roaring_bitmap/roaring_bitmap.inl>