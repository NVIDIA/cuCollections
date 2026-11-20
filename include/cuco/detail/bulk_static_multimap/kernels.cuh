/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuco/detail/error.hpp>
#include <cuco/detail/storage/storage_base.cuh>

#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>

namespace cuco::detail {
template <class Function>
__global__ void bulk_multimap_for_indices(std::size_t size, Function function)
{
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
       i += std::size_t(gridDim.x) * blockDim.x)
    function(i);
}

template <class Function>
void bulk_multimap_launch(std::size_t size, cuda::stream_ref stream, Function function)
{
  if (!size) return;
  auto const blocks = std::min(std::size_t{65535}, (size - 1) / 256 + 1);
  bulk_multimap_for_indices<<<static_cast<unsigned>(blocks), 256, 0, stream.get()>>>(size,
                                                                                     function);
  CUCO_CUDA_TRY(cudaGetLastError());
}

template <class T, class Allocator>
class bulk_multimap_buffer {
  using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<T>;
  using deleter_type   = custom_deleter<std::size_t, allocator_type>;

 public:
  bulk_multimap_buffer(std::size_t size, Allocator const& allocator, cuda::stream_ref stream)
    : pointer_{allocate(size, allocator_type{allocator}, stream),
               deleter_type{size, allocator_type{allocator}, stream}}
  {
  }
  T* data() const noexcept { return pointer_.get(); }

 private:
  static T* allocate(std::size_t size, allocator_type allocator, cuda::stream_ref stream)
  {
    CUCO_EXPECTS(size <= std::numeric_limits<std::size_t>::max() / sizeof(T),
                 "Allocation byte count overflows");
    return size ? allocator.allocate(size, stream) : nullptr;
  }
  std::unique_ptr<T, deleter_type> pointer_;
};

template <class InputIt>
std::size_t bulk_multimap_range_size(InputIt first, InputIt last)
{
  if (first == last) return 0;
  auto const distance = last - first;
  CUCO_EXPECTS(distance >= 0, "Reversed input range");
  return static_cast<std::size_t>(distance);
}
}  // namespace cuco::detail
