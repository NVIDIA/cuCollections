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

template <typename T, typename Compare, bool FavorInsertionPerformance, typename Allocator>
priority_queue<T, Compare, FavorInsertionPerformance, Allocator>::priority_queue(
  size_t initial_capacity, Allocator const& allocator)
  : allocator_{allocator},
    int_allocator_{allocator},
    t_allocator_{allocator},
    size_t_allocator_{allocator}
{
  node_size_ = NodeSize;

  // Round up to the nearest multiple of node size
  int nodes = ((initial_capacity + node_size_ - 1) / node_size_);

  node_capacity_      = nodes;
  lowest_level_start_ = 1 << (int)log2(nodes);

  // Allocate device variables

  d_size_ = std::allocator_traits<int_allocator_type>::allocate(int_allocator_, 1);

  CUCO_CUDA_TRY(cudaMemset(d_size_, 0, sizeof(int)));

  d_p_buffer_size_ = std::allocator_traits<size_t_allocator_type>::allocate(size_t_allocator_, 1);

  CUCO_CUDA_TRY(cudaMemset(d_p_buffer_size_, 0, sizeof(size_t)));

  d_heap_ = std::allocator_traits<t_allocator_type>::allocate(
    t_allocator_, node_capacity_ * node_size_ + node_size_);

  d_locks_ =
    std::allocator_traits<int_allocator_type>::allocate(int_allocator_, node_capacity_ + 1);

  CUCO_CUDA_TRY(cudaMemset(d_locks_, 0, sizeof(int) * (node_capacity_ + 1)));
}

template <typename T, typename Compare, bool FavorInsertionPerformance, typename Allocator>
priority_queue<T, Compare, FavorInsertionPerformance, Allocator>::~priority_queue()
{
  std::allocator_traits<int_allocator_type>::deallocate(int_allocator_, d_size_, 1);
  std::allocator_traits<size_t_allocator_type>::deallocate(size_t_allocator_, d_p_buffer_size_, 1);
  std::allocator_traits<t_allocator_type>::deallocate(
    t_allocator_, d_heap_, node_capacity_ * node_size_ + node_size_);
  std::allocator_traits<int_allocator_type>::deallocate(
    int_allocator_, d_locks_, node_capacity_ + 1);
}

template <typename T, typename Compare, bool FavorInsertionPerformance, typename Allocator>
template <typename InputIt>
void priority_queue<T, Compare, FavorInsertionPerformance, Allocator>::push(InputIt first,
                                                                            InputIt last,
                                                                            cudaStream_t stream)
{
  const int kBlockSize = min(256, (int)node_size_);
  const int kNumBlocks = min(64000, max(1, (int)((last - first) / node_size_)));

  PushKernel<<<kNumBlocks, kBlockSize, get_shmem_size(kBlockSize), stream>>>(first,
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

template <typename T, typename Compare, bool FavorInsertionPerformance, typename Allocator>
template <typename OutputIt>
void priority_queue<T, Compare, FavorInsertionPerformance, Allocator>::pop(OutputIt first,
                                                                           OutputIt last,
                                                                           cudaStream_t stream)
{
  int pop_size      = last - first;
  const int partial = pop_size % node_size_;

  const int kBlockSize = min(256, (int)node_size_);
  const int kNumBlocks = min(64000, max(1, (int)((pop_size - partial) / node_size_)));

  PopKernel<<<kNumBlocks, kBlockSize, get_shmem_size(kBlockSize), stream>>>(first,
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

template <typename T, typename Compare, bool FavorInsertionPerformance, typename Allocator>
template <typename CG, typename InputIt>
__device__ void
priority_queue<T, Compare, FavorInsertionPerformance, Allocator>::device_mutable_view::push(
  CG const& g, InputIt first, InputIt last, void* temp_storage)
{
  SharedMemoryLayout<T> shmem = GetSharedMemoryLayout<T>((int*)temp_storage, g.size(), node_size_);

  auto push_size = last - first;
  for (size_t i = 0; i < push_size / node_size_; i++) {
    PushSingleNode(g,
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
    PushPartialNode(g,
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

template <typename T, typename Compare, bool FavorInsertionPerformance, typename Allocator>
template <typename CG, typename OutputIt>
__device__ void
priority_queue<T, Compare, FavorInsertionPerformance, Allocator>::device_mutable_view::pop(
  CG const& g, OutputIt first, OutputIt last, void* temp_storage)
{
  SharedMemoryLayout<T> shmem = GetSharedMemoryLayout<T>((int*)temp_storage, g.size(), node_size_);

  auto pop_size = last - first;
  for (size_t i = 0; i < pop_size / node_size_; i++) {
    PopSingleNode(g,
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
    PopPartialNode(g,
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
