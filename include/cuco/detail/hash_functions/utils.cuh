/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cassert>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>

namespace cuco::detail {

template <typename T, typename U, typename Extent>
constexpr __host__ __device__ T load_chunk(U const* const data, Extent index) noexcept
{
  auto const bytes = reinterpret_cast<std::byte const*>(data);
  T chunk;
  memcpy(&chunk, bytes + index * sizeof(T), sizeof(T));
  return chunk;
}

template <typename T>
struct AndOrPair {
  __device__ AndOrPair(T and_value, T or_value) : and_value(and_value), or_value(or_value) {}
  __device__ AndOrPair(T other) : and_value(other), or_value(other) {}
  __device__ AndOrPair() = default;
  __device__ AndOrPair& operator=(T const& other)
  {
    and_value = other;
    or_value  = other;
    return *this;
  }

  T and_value;
  T or_value;
};

struct AndOrPairCombiner {
  template <typename T>
  __device__ AndOrPair<T> operator()(const AndOrPair<T>& a, const AndOrPair<T>& b) const
  {
    return AndOrPair<T>(a.and_value & b.and_value, a.or_value | b.or_value);
  }
  template <typename T>
  __device__ AndOrPair<T> operator()(const AndOrPair<T>& a, const T& b) const
  {
    return AndOrPair<T>(a.and_value & b, a.or_value | b);
  }
};

template <typename T>
void reduce_and_or_helper(AndOrPair<T>* output, T* inputs, int64_t inputs_size)
{
  AndOrPairCombiner combiner_op;
  auto init = AndOrPair<T>(T(-1), T(0));

  // Determine temporary device storage requirements
  void* d_temp_storage      = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(
    d_temp_storage, temp_storage_bytes, inputs, output, inputs_size, combiner_op, init);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run reduction
  cub::DeviceReduce::Reduce(
    d_temp_storage, temp_storage_bytes, inputs, output, inputs_size, combiner_op, init);
  cudaFree(d_temp_storage);
}

// Helper function to allocate memory on the device and copy data from host to device
template <typename T>
T* allocate_and_copy_to_device(const T* host_data, int64_t size)
{
  T* device_data;
  cudaMalloc(&device_data, size * sizeof(T));
  cudaMemcpy(device_data, host_data, size * sizeof(T), cudaMemcpyHostToDevice);
  return device_data;
}

// Helper function to allocate memory on the device
template <typename T>
T* allocate_device_memory(int64_t size)
{
  T* device_data;
  cudaMalloc(&device_data, size * sizeof(T));
  return device_data;
}

// Helper function to copy data from device to host
template <typename T>
void copy_to_host(T* host_data, const T* device_data, int64_t size)
{
  cudaMemcpy(host_data, device_data, size * sizeof(T), cudaMemcpyDeviceToHost);
}

// The first element of and_or gets the binary and reduction of all
// the inputs. The second element of and_or gets the binary or
// reduction of all the inputs.
template <typename T>
AndOrPair<T> reduce_and_or_cuda(const T* host_input, int64_t num_elements)
{
  T* device_input             = allocate_and_copy_to_device(host_input, num_elements);
  AndOrPair<T>* device_and_or = allocate_device_memory<AndOrPair<T>>(1);

  reduce_and_or_helper(device_and_or, device_input, num_elements);

  AndOrPair<T> host_and_or;
  copy_to_host(&host_and_or, device_and_or, 1);

  cudaFree(device_input);
  cudaFree(device_and_or);

  return host_and_or;
}

};  // namespace cuco::detail
