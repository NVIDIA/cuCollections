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

#include <utils.hpp>

#include <cuco/bit_vector.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <catch2/catch_test_macros.hpp>

__global__ void bitvector_get_kernel(cuco::experimental::bit_vector* bv, size_t n, uint32_t* output) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  while (index < n) {
    output[index] = bv->get(index);
    index += stride;
  }
}

TEST_CASE("Get test", "")
{
  constexpr std::size_t num_elements{400};

  cuco::experimental::bit_vector bv;

  for (size_t i = 0; i < num_elements; i++) {
    bv.add(i % 7 == 0); // Alternate 0s and 1s pattern
  }
  bv.build();

  cuco::experimental::bit_vector* bv_device_copy;
  CUCO_CUDA_TRY(cudaMalloc(&bv_device_copy, sizeof(cuco::experimental::bit_vector)));
  CUCO_CUDA_TRY(cudaMemcpy(bv_device_copy, &bv, sizeof(cuco::experimental::bit_vector), cudaMemcpyHostToDevice));

  thrust::device_vector<uint32_t> get_result(num_elements);

  bitvector_get_kernel<<<1, 1024>>>(bv_device_copy, num_elements, thrust::raw_pointer_cast(get_result.data()));

  CUCO_CUDA_TRY(cudaFree(bv_device_copy));

  size_t num_set = thrust::reduce(thrust::device, get_result.begin(), get_result.end(), 0);
  REQUIRE(num_set == num_elements / 7 + 1);
}
