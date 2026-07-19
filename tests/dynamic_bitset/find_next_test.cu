/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/trie/dynamic_bitset/dynamic_bitset.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include <catch2/catch_test_macros.hpp>

template <class BitsetRef, typename size_type, typename OutputIt>
__global__ void find_next_kernel(BitsetRef ref, size_type num_elements, OutputIt output)
{
  cuco::detail::index_type index  = blockIdx.x * blockDim.x + threadIdx.x;
  cuco::detail::index_type stride = gridDim.x * blockDim.x;
  while (index < num_elements) {
    output[index] = ref.find_next(index);
    index += stride;
  }
}

using cuco::test::modulo_bitgen;

TEST_CASE("dynamic_bitset find next set test", "")
{
  cuco::experimental::detail::dynamic_bitset bv;

  using size_type = std::size_t;
  constexpr size_type num_elements{400};

  for (size_type i = 0; i < num_elements; i++) {
    bv.push_back(modulo_bitgen(i));
  }

  thrust::device_vector<size_type> device_result(num_elements);
  auto ref = bv.ref();
  find_next_kernel<<<1, 1024>>>(ref, num_elements, device_result.data());

  thrust::host_vector<size_type> host_result = device_result;
  size_type num_matches                      = 0;

  size_type next_set_pos = -1lu;
  do {
    next_set_pos++;
  } while (next_set_pos < num_elements and !modulo_bitgen(next_set_pos));

  for (size_type key = 0; key < num_elements; key++) {
    num_matches += host_result[key] == next_set_pos;

    if (key == next_set_pos) {
      do {
        next_set_pos++;
      } while (next_set_pos < num_elements and !modulo_bitgen(next_set_pos));
    }
  }
  REQUIRE(num_matches == num_elements);
}
