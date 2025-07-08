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

#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/roaring_bitmap_ref.cuh>
#include <cuco/utility/allocator.hpp>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cuda/std/cstddef>
#include <cuda/std/span>
#include <cuda/stream_ref>

#include <memory>

namespace cuco {

template <class T,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class Allocator          = cuco::cuda_allocator<cuda::std::byte>>
class roaring_bitmap {
 public:
  static constexpr auto thread_scope = Scope;

  using allocator_type = Allocator;

  template <cuda::thread_scope NewScope = thread_scope>
  using ref_type = roaring_bitmap_ref<T, NewScope>;

  __host__ roaring_bitmap(cuda::std::byte const* bitmap,
                          cuda_thread_scope<Scope> scope = {},
                          Allocator const& alloc         = {},
                          cuda::stream_ref stream        = {});

  roaring_bitmap(roaring_bitmap const& other)            = default;
  roaring_bitmap(roaring_bitmap&& other)                 = default;
  roaring_bitmap& operator=(roaring_bitmap const& other) = default;
  roaring_bitmap& operator=(roaring_bitmap&& other)      = default;

  ~roaring_bitmap() = default;

  template <class InputIt, class OutputIt>
  __host__ void contains(InputIt first,
                         InputIt last,
                         OutputIt contained,
                         cuda::stream_ref stream = {}) const;

  template <class InputIt, class OutputIt>
  __host__ void contains_async(InputIt first,
                               InputIt last,
                               OutputIt contained,
                               cuda::stream_ref stream = {}) const noexcept;

  // TODO contains_if, contains_if_async, empty

  [[nodiscard]] __host__ cuda::std::size_t size() const noexcept;

  [[nodiscard]] __host__ cuda::std::span<cuda::std::byte const> data() const noexcept;

  [[nodiscard]] __host__ allocator_type allocator() const noexcept;

  [[nodiscard]] __host__ ref_type<> ref() const noexcept;

 private:
  allocator_type allocator_;
  typename ref_type<>::metadata_type metadata_;
  std::unique_ptr<cuda::std::byte, detail::custom_deleter<cuda::std::size_t, allocator_type>> data_;
  ref_type<> ref_;
};

}  // namespace cuco

#include <cuco/detail/roaring_bitmap/roaring_bitmap.inl>