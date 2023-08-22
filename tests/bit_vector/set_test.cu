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

template <class BitVectorRef>
__global__ void set_kernel(BitVectorRef ref, uint64_t* keys, uint64_t* vals, uint64_t num_keys)
{
  size_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  while (index < num_keys) {
    ref.set(keys[index], vals[index]);
    index += stride;
  }
}

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

TEST_CASE("Set test", "")
{
  constexpr std::size_t num_elements{400};
  cuco::experimental::bit_vector bv;

  // Set odd bits on host
  for (size_t i = 0; i < num_elements; i++) {
    bv.append(i % 2 == 1);
  }
  bv.build();

  auto get_ref = bv.ref(cuco::experimental::bv_read);
  thrust::device_vector<uint64_t> get_result(num_elements);
  get_kernel<<<32, 32>>>(get_ref, num_elements, thrust::raw_pointer_cast(get_result.data()));
  size_t num_set = thrust::reduce(thrust::device, get_result.begin(), get_result.end(), 0);
  REQUIRE(num_set == num_elements / 2);

  // Set all bits on device
  thrust::device_vector<uint64_t> d_keys(num_elements);
  thrust::sequence(d_keys.begin(), d_keys.end(), 0);

  thrust::device_vector<uint64_t> d_vals(num_elements);
  thrust::fill(d_vals.begin(), d_vals.end(), 1);

  auto set_ref = bv.ref(cuco::experimental::bv_set);
  set_kernel<<<32, 32>>>(set_ref,
                         thrust::raw_pointer_cast(d_keys.data()),
                         thrust::raw_pointer_cast(d_vals.data()),
                         num_elements);

  // Check that all bits are set
  get_kernel<<<32, 32>>>(get_ref, num_elements, thrust::raw_pointer_cast(get_result.data()));
  num_set = thrust::reduce(thrust::device, get_result.begin(), get_result.end(), 0);
  REQUIRE(num_set == num_elements);
}
