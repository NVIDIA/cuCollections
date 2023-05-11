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
#include <thrust/host_vector.h>

#include <catch2/catch_test_macros.hpp>

__global__ void bitvector_select_kernel(cuco::experimental::bit_vector* bv, size_t n, uint32_t* output) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  while (index < n) {
    output[index] = bv->select(index);
    index += stride;
  }
}

extern bool modulo_bitgen(uint32_t i);

TEST_CASE("Select test", "")
{
  constexpr std::size_t num_elements{400};

  cuco::experimental::bit_vector bv;

  uint32_t num_set = 0;
  for (size_t i = 0; i < num_elements; i++) {
    bv.add(modulo_bitgen(i));
    num_set += modulo_bitgen(i);
  }
  bv.build();

  thrust::device_vector<uint32_t> select_result_device(num_set);

  cuco::experimental::bit_vector* bv_device_copy;
  CUCO_CUDA_TRY(cudaMalloc(&bv_device_copy, sizeof(cuco::experimental::bit_vector)));
  CUCO_CUDA_TRY(cudaMemcpy(bv_device_copy, &bv, sizeof(cuco::experimental::bit_vector), cudaMemcpyHostToDevice));

  bitvector_select_kernel<<<1, 1024>>>(bv_device_copy, num_set, thrust::raw_pointer_cast(select_result_device.data()));

  CUCO_CUDA_TRY(cudaFree(bv_device_copy));

  thrust::host_vector<uint32_t> select_result = select_result_device;

  uint32_t num_matches = 0;
  uint32_t cur_set_pos = -1u;
  for (size_t i = 0; i < num_set; i++) {
    do {
      cur_set_pos++;
    } while (cur_set_pos < num_elements and !modulo_bitgen(cur_set_pos));

    num_matches += cur_set_pos == select_result[i];
  }
  REQUIRE(num_matches == num_set);
}
