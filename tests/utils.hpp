/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <utils.cuh>

#include <cuco/detail/error.hpp>

#include <thrust/functional.h>

#include <cooperative_groups.h>

#include <iterator>

namespace cuco {
namespace test {

namespace cg = cooperative_groups;

constexpr int32_t block_size = 128;

enum class probe_sequence { linear_probing, double_hashing };

// User-defined logical algorithms to reduce compilation time
template <typename Iterator, typename Predicate>
int count_if(Iterator begin, Iterator end, Predicate p, cudaStream_t stream = 0)
{
  auto const size      = std::distance(begin, end);
  auto const grid_size = (size + block_size - 1) / block_size;

  int* count;
  CUCO_CUDA_TRY(cudaMallocManaged(&count, sizeof(int)));

  *count = 0;
  int device_id;
  CUCO_CUDA_TRY(cudaGetDevice(&device_id));
  CUCO_CUDA_TRY(cudaMemPrefetchAsync(count, sizeof(int), device_id, stream));

  detail::count_if<<<grid_size, block_size, 0, stream>>>(begin, end, count, p);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  auto const res = *count;

  CUCO_CUDA_TRY(cudaFree(count));

  return res;
}

template <typename Iterator, typename Predicate>
bool all_of(Iterator begin, Iterator end, Predicate p, cudaStream_t stream = 0)
{
  auto const size  = std::distance(begin, end);
  auto const count = count_if(begin, end, p, stream);

  return size == count;
}

template <typename Iterator, typename Predicate>
bool any_of(Iterator begin, Iterator end, Predicate p, cudaStream_t stream = 0)
{
  auto const count = count_if(begin, end, p, stream);
  return count > 0;
}

template <typename Iterator, typename Predicate>
bool none_of(Iterator begin, Iterator end, Predicate p, cudaStream_t stream = 0)
{
  return not all_of(begin, end, p, stream);
}

template <typename Iterator1, typename Iterator2, typename Predicate>
bool equal(Iterator1 begin1, Iterator1 end1, Iterator2 begin2, Predicate p, cudaStream_t stream = 0)
{
  auto const size      = std::distance(begin1, end1);
  auto const grid_size = (size + block_size - 1) / block_size;

  int* count;
  CUCO_CUDA_TRY(cudaMallocManaged(&count, sizeof(int)));

  *count = 0;
  int device_id;
  CUCO_CUDA_TRY(cudaGetDevice(&device_id));
  CUCO_CUDA_TRY(cudaMemPrefetchAsync(count, sizeof(int), device_id, stream));

  detail::count_if<<<grid_size, block_size, 0, stream>>>(begin1, end1, begin2, count, p);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  auto const res = *count;

  CUCO_CUDA_TRY(cudaFree(count));

  return res == size;
}

}  // namespace test
}  // namespace cuco
