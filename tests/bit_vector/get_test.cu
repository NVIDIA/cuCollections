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

#include <cuco/detail/trie/bit_vector/bit_vector.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <catch2/catch_test_macros.hpp>

template <class BitVectorRef>
__global__ void get_kernel(BitVectorRef ref, size_t n, uint64_t* output)
{
  size_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  while (index < n) {
    output[index] = ref.get(index);
    index += stride;
  }
}

bool modulo_bitgen(uint64_t i) { return i % 7 == 0; }

TEST_CASE("Get test", "")
{
  constexpr std::size_t num_elements{400};

  cuco::experimental::bit_vector bv;

  size_t num_set_ref = 0;
  for (size_t i = 0; i < num_elements; i++) {
    bv.append(modulo_bitgen(i));
    num_set_ref += modulo_bitgen(i);
  }
  bv.build();

  auto ref = bv.ref(cuco::experimental::bv_read);
  thrust::device_vector<uint64_t> get_result(num_elements);
  get_kernel<<<1, 1024>>>(ref, num_elements, thrust::raw_pointer_cast(get_result.data()));

  size_t num_set = thrust::reduce(thrust::device, get_result.begin(), get_result.end(), 0);
  REQUIRE(num_set == num_set_ref);
}
