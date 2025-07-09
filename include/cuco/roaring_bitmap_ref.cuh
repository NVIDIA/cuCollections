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

#include <cuco/detail/roaring_bitmap/roaring_bitmap_impl.cuh>

#include <cuda/std/cstddef>
#include <cuda/stream_ref>

namespace cuco {

template <class T>
class roaring_bitmap_ref {
  using impl_type = detail::roaring_bitmap_impl<T>;

 public:
  using storage_ref_type = typename impl_type::storage_ref_type;

  __host__ __device__ roaring_bitmap_ref(storage_ref_type const& storage_ref);

  template <class U = T,
            class   = cuda::std::enable_if_t<cuda::std::is_same_v<U, cuda::std::uint32_t>>>
  __device__ roaring_bitmap_ref(cuda::std::byte const* bitmap);

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

  __device__ bool contains(T value) const;

  [[nodiscard]] __host__ __device__ cuda::std::size_t size() const noexcept;

  [[nodiscard]] __host__ __device__ bool empty() const noexcept;

  [[nodiscard]] __host__ __device__ cuda::std::byte const* data() const noexcept;

  [[nodiscard]] __host__ __device__ cuda::std::size_t size_bytes() const noexcept;

 private:
  impl_type impl_;
};

}  // namespace cuco

#include <cuco/detail/roaring_bitmap/roaring_bitmap_ref.inl>