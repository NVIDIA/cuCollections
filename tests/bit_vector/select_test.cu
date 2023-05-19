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

template <class BitVectorRef>
__global__ void select_kernel(BitVectorRef ref, size_t n, uint64_t* output) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  while (index < n) {
    output[index] = ref.select(index);
    index += stride;
  }
}


template <class BitVectorRef>
__global__ void select0_kernel(BitVectorRef ref, size_t n, uint64_t* output) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  while (index < n) {
    output[index] = ref.select0(index);
    index += stride;
  }
}

extern bool modulo_bitgen(uint32_t i);

TEST_CASE("Select test", "")
{
  constexpr std::size_t num_elements{400};

  using Key = uint64_t;
  cuco::experimental::bit_vector bv{cuco::experimental::extent<std::size_t>{400}};

  uint32_t num_set = 0;
  for (size_t i = 0; i < num_elements; i++) {
    bv.add(modulo_bitgen(i));
    num_set += modulo_bitgen(i);
  }
  bv.build();
  auto ref                = bv.ref(cuco::experimental::select);


  // Check select
  {
      thrust::device_vector<uint64_t> device_result(num_set);
  select_kernel<<<1, 1024>>>(ref, num_set, thrust::raw_pointer_cast(device_result.data()));
  thrust::host_vector<uint64_t> host_result = device_result;

  uint32_t num_matches = 0;
  uint32_t cur_set_pos = -1u;
  for (size_t i = 0; i < num_set; i++) {
    do {
      cur_set_pos++;
    } while (cur_set_pos < num_elements and !modulo_bitgen(cur_set_pos));

    num_matches += cur_set_pos == host_result[i];
  }
  REQUIRE(num_matches == num_set);
  }

  // Check select0
  {
  uint32_t num_not_set = num_elements - num_set;

  thrust::device_vector<uint64_t> device_result(num_not_set);
  select0_kernel<<<1, 1024>>>(ref, num_not_set, thrust::raw_pointer_cast(device_result.data()));
  thrust::host_vector<uint64_t> host_result = device_result;

  uint32_t num_matches = 0;
  uint32_t cur_not_set_pos = -1u;
  for (size_t i = 0; i < num_not_set; i++) {
    do {
      cur_not_set_pos++;
    } while (cur_not_set_pos < num_elements and modulo_bitgen(cur_not_set_pos));

    num_matches += cur_not_set_pos == host_result[i];
  }
  REQUIRE(num_matches == num_not_set);
  }
}
