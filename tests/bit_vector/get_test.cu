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

#include <catch2/catch_test_macros.hpp>
#include <cuco/detail/trie/bit_vector/bit_vector.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <utils.hpp>

template <class BitVectorRef, typename size_type, typename OutputIt>
__global__ void get_kernel(BitVectorRef ref, size_type num_elements, OutputIt output)
{
  size_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  while (index < num_elements) {
    output[index] = ref.get(index);
    index += stride;
  }
}

bool modulo_bitgen(uint64_t i) { return i % 7 == 0; }

TEST_CASE("Get test", "")
{
  cuco::experimental::detail::bit_vector bv;

  using size_type = std::size_t;
  constexpr size_type num_elements{400};

  size_type num_set_ref = 0;
  for (size_type i = 0; i < num_elements; i++) {
    bv.append(modulo_bitgen(i));
    num_set_ref += modulo_bitgen(i);
  }
  bv.build();

  // Device-ref test
  auto ref = bv.ref();
  thrust::device_vector<size_type> get_result(num_elements);
  get_kernel<<<1, 1024>>>(ref, num_elements, get_result.data());

  size_type num_set = thrust::reduce(thrust::device, get_result.begin(), get_result.end(), 0);
  REQUIRE(num_set == num_set_ref);

  // Host-bulk test
  thrust::device_vector<size_type> keys(num_elements);
  thrust::sequence(keys.begin(), keys.end(), 0);
  thrust::fill(get_result.begin(), get_result.end(), 0);

  bv.get(keys.begin(), keys.end(), get_result.begin());

  num_set = thrust::reduce(thrust::device, get_result.begin(), get_result.end(), 0);
  REQUIRE(num_set == num_set_ref);
}
