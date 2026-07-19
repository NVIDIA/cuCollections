/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/trie/dynamic_bitset/dynamic_bitset.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <catch2/catch_test_macros.hpp>

template <class BitsetRef, typename size_type, typename OutputIt>
__global__ void test_kernel(BitsetRef ref, size_type num_elements, OutputIt output)
{
  cuco::detail::index_type index  = blockIdx.x * blockDim.x + threadIdx.x;
  cuco::detail::index_type stride = gridDim.x * blockDim.x;
  while (index < num_elements) {
    output[index] = ref.test(index);
    index += stride;
  }
}

using cuco::test::modulo_bitgen;

TEST_CASE("dynamic_bitset get test", "")
{
  cuco::experimental::detail::dynamic_bitset bv;

  using size_type = std::size_t;
  constexpr size_type num_elements{400};

  size_type num_set_ref = 0;
  for (size_type i = 0; i < num_elements; i++) {
    bv.push_back(modulo_bitgen(i));
    num_set_ref += modulo_bitgen(i);
  }

  // Host-bulk test
  thrust::device_vector<size_type> keys(num_elements);
  thrust::sequence(keys.begin(), keys.end(), 0);

  thrust::device_vector<size_type> test_result(num_elements);
  thrust::fill(test_result.begin(), test_result.end(), 0);

  bv.test(keys.begin(), keys.end(), test_result.begin());

  size_type num_set = thrust::reduce(thrust::device, test_result.begin(), test_result.end(), 0);
  REQUIRE(num_set == num_set_ref);

  // Device-ref test
  auto ref = bv.ref();
  thrust::fill(test_result.begin(), test_result.end(), 0);
  test_kernel<<<1, 1024>>>(ref, num_elements, test_result.data());

  num_set = thrust::reduce(thrust::device, test_result.begin(), test_result.end(), 0);
  REQUIRE(num_set == num_set_ref);
}
