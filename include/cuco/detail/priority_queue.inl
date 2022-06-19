/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cuco/detail/priority_queue_kernels.cuh>

#include <cmath>

namespace cuco {

template <typename T, typename Compare, typename Allocator>
priority_queue<T, Compare, Allocator>::priority_queue(std::size_t initial_capacity,
                                                      Allocator const& allocator,
                                                      cudaStream_t stream)
  : int_allocator_{allocator}, t_allocator_{allocator}, size_t_allocator_{allocator}
{
  node_size_ = 1024;

  // Round up to the nearest multiple of node size
  const int nodes = ((initial_capacity + node_size_ - 1) / node_size_);

  node_capacity_      = nodes;
  lowest_level_start_ = 1 << static_cast<int>(std::log2(nodes));

  // Allocate device variables

  d_size_ = std::allocator_traits<int_allocator_type>::allocate(int_allocator_, 1);

  CUCO_CUDA_TRY(cudaMemsetAsync(d_size_, 0, sizeof(int), stream));

  d_p_buffer_size_ = std::allocator_traits<size_t_allocator_type>::allocate(size_t_allocator_, 1);

  CUCO_CUDA_TRY(cudaMemsetAsync(d_p_buffer_size_, 0, sizeof(std::size_t), stream));

  d_heap_ = std::allocator_traits<t_allocator_type>::allocate(
    t_allocator_, node_capacity_ * node_size_ + node_size_);

  d_locks_ =
    std::allocator_traits<int_allocator_type>::allocate(int_allocator_, node_capacity_ + 1);

  CUCO_CUDA_TRY(cudaMemsetAsync(d_locks_, 0, sizeof(int) * (node_capacity_ + 1), stream));
}

template <typename T, typename Compare, typename Allocator>
priority_queue<T, Compare, Allocator>::~priority_queue()
{
  std::allocator_traits<int_allocator_type>::deallocate(int_allocator_, d_size_, 1);
  std::allocator_traits<size_t_allocator_type>::deallocate(size_t_allocator_, d_p_buffer_size_, 1);
  std::allocator_traits<t_allocator_type>::deallocate(
    t_allocator_, d_heap_, node_capacity_ * node_size_ + node_size_);
  std::allocator_traits<int_allocator_type>::deallocate(
    int_allocator_, d_locks_, node_capacity_ + 1);
}

template <typename T, typename Compare, typename Allocator>
template <typename InputIt>
void priority_queue<T, Compare, Allocator>::push(InputIt first, InputIt last, cudaStream_t stream)
{
  constexpr int block_size = 256;

  const int num_nodes  = static_cast<int>((last - first) / node_size_) + 1;
  const int num_blocks = std::min(64000, num_nodes);

  detail::push_kernel<<<num_blocks, block_size, get_shmem_size(block_size), stream>>>(
    first,
    last - first,
    d_heap_,
    d_size_,
    node_size_,
    d_locks_,
    d_p_buffer_size_,
    lowest_level_start_,
    compare_);

  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename T, typename Compare, typename Allocator>
template <typename OutputIt>
void priority_queue<T, Compare, Allocator>::pop(OutputIt first, OutputIt last, cudaStream_t stream)
{
  constexpr int block_size = 256;
  const int pop_size   = last - first;

  const int num_nodes  = static_cast<int>(pop_size / node_size_) + 1;
  const int num_blocks = std::min(64000, num_nodes);

  detail::pop_kernel<<<num_blocks, block_size, get_shmem_size(block_size), stream>>>(
    first,
    pop_size,
    d_heap_,
    d_size_,
    node_size_,
    d_locks_,
    d_p_buffer_size_,
    lowest_level_start_,
    node_capacity_,
    compare_);

  CUCO_CUDA_TRY(cudaGetLastError());
}

template <typename T, typename Compare, typename Allocator>
template <typename CG, typename InputIt>
__device__ void priority_queue<T, Compare, Allocator>::device_mutable_view::push(CG const& g,
                                                                                 InputIt first,
                                                                                 InputIt last,
                                                                                 void* temp_storage)
{
  const detail::shared_memory_layout<T> shmem =
    detail::get_shared_memory_layout<T>((int*)temp_storage, g.size(), node_size_);

  const auto push_size = last - first;
  for (std::size_t i = 0; i < push_size / node_size_; i++) {
    detail::push_single_node(g,
                             first + i * node_size_,
                             d_heap_,
                             d_size_,
                             node_size_,
                             d_locks_,
                             lowest_level_start_,
                             shmem,
                             compare_);
  }

  if (push_size % node_size_ != 0) {
    detail::push_partial_node(g,
                              first + (push_size / node_size_) * node_size_,
                              push_size % node_size_,
                              d_heap_,
                              d_size_,
                              node_size_,
                              d_locks_,
                              d_p_buffer_size_,
                              lowest_level_start_,
                              shmem,
                              compare_);
  }
}

template <typename T, typename Compare, typename Allocator>
template <typename CG, typename OutputIt>
__device__ void priority_queue<T, Compare, Allocator>::device_mutable_view::pop(CG const& g,
                                                                                OutputIt first,
                                                                                OutputIt last,
                                                                                void* temp_storage)
{
  const detail::shared_memory_layout<T> shmem =
    detail::get_shared_memory_layout<T>((int*)temp_storage, g.size(), node_size_);

  const auto pop_size = last - first;
  for (std::size_t i = 0; i < pop_size / node_size_; i++) {
    detail::pop_single_node(g,
                            first + i * node_size_,
                            d_heap_,
                            d_size_,
                            node_size_,
                            d_locks_,
                            d_p_buffer_size_,
                            lowest_level_start_,
                            node_capacity_,
                            shmem,
                            compare_);
  }

  if (pop_size % node_size_ != 0) {
    detail::pop_partial_node(g,
                             first + (pop_size / node_size_) * node_size_,
                             last - first,
                             d_heap_,
                             d_size_,
                             node_size_,
                             d_locks_,
                             d_p_buffer_size_,
                             lowest_level_start_,
                             node_capacity_,
                             shmem,
                             compare_);
  }
}

}  // namespace cuco
