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
__global__ void rank_kernel(BitVectorRef ref, size_t n, uint32_t* output)
{
  size_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  while (index < n) {
    output[index] = ref.rank(index);
    index += stride;
  }
}

extern bool modulo_bitgen(uint32_t i);

TEST_CASE("Rank test", "")
{
  constexpr std::size_t num_elements{400};

  using Key = uint64_t;
  cuco::experimental::bit_vector bv{cuco::experimental::extent<std::size_t>{400}};

  for (size_t i = 0; i < num_elements; i++) {
    bv.add(modulo_bitgen(i));
  }
  bv.build();

  thrust::device_vector<uint32_t> rank_result_device(num_elements);
  auto ref = bv.ref(cuco::experimental::bv_read);
  rank_kernel<<<1, 1024>>>(ref, num_elements, thrust::raw_pointer_cast(rank_result_device.data()));

  thrust::host_vector<uint32_t> rank_result = rank_result_device;
  uint32_t cur_rank                         = 0;
  uint32_t num_matches                      = 0;
  for (size_t i = 0; i < num_elements; i++) {
    num_matches += cur_rank == rank_result[i];
    if (modulo_bitgen(i)) { cur_rank++; }
  }
  REQUIRE(num_matches == num_elements);
}
