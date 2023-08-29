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
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <catch2/catch_test_macros.hpp>

template <class BitVectorRef, typename size_type, typename OutputIt>
__global__ void select0_kernel(BitVectorRef ref, size_type num_elements, OutputIt output)
{
  size_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  while (index < num_elements) {
    output[index] = ref.select0(index);
    index += stride;
  }
}

extern bool modulo_bitgen(uint64_t i);  // Defined in get_test.cu

TEST_CASE("Select test", "")
{
  cuco::experimental::detail::bit_vector bv;

  using size_type = cuco::experimental::detail::bit_vector::size_type;
  constexpr size_type num_elements{4000};

  size_type num_set = 0;
  for (size_type i = 0; i < num_elements; i++) {
    bv.append(modulo_bitgen(i));
    num_set += modulo_bitgen(i);
  }
  bv.build();
  auto ref = bv.ref(cuco::experimental::bv_read);

  // Check select
  {
    thrust::device_vector<size_type> keys(num_set);
    thrust::sequence(keys.begin(), keys.end(), 0);

    thrust::device_vector<size_type> d_selects(num_set);

    bv.selects(keys.begin(), keys.end(), d_selects.begin());

    thrust::host_vector<size_type> h_selects = d_selects;

    size_type num_matches = 0;
    size_type cur_set_pos = -1lu;
    for (size_type i = 0; i < num_set; i++) {
      do {
        cur_set_pos++;
      } while (cur_set_pos < num_elements and !modulo_bitgen(cur_set_pos));

      num_matches += cur_set_pos == h_selects[i];
    }
    REQUIRE(num_matches == num_set);
  }

  // Check select0
  {
    size_type num_not_set = num_elements - num_set;

    thrust::device_vector<size_type> device_result(num_not_set);
    select0_kernel<<<1, 1024>>>(ref, num_not_set, device_result.data());
    thrust::host_vector<size_type> host_result = device_result;

    size_type num_matches     = 0;
    size_type cur_not_set_pos = -1lu;
    for (size_type i = 0; i < num_not_set; i++) {
      do {
        cur_not_set_pos++;
      } while (cur_not_set_pos < num_elements and modulo_bitgen(cur_not_set_pos));

      num_matches += cur_not_set_pos == host_result[i];
    }
    REQUIRE(num_matches == num_not_set);
  }
}
