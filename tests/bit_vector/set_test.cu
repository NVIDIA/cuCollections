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

TEST_CASE("Set test", "")
{
  cuco::experimental::bit_vector bv;

  using size_type = cuco::experimental::bit_vector<>::size_type;
  size_type num_elements{400};

  // Set odd bits on host
  for (size_t i = 0; i < num_elements; i++) {
    bv.append(i % 2 == 1);
  }
  bv.build();

  // Set all bits on device
  thrust::device_vector<size_type> keys(num_elements);
  thrust::sequence(keys.begin(), keys.end(), 0);

  thrust::device_vector<bool> vals(num_elements);
  thrust::fill(vals.begin(), vals.end(), 1);

  bv.set(keys.begin(), keys.end(), vals.begin());

  // Check that all bits are set
  thrust::device_vector<size_type> get_outputs(num_elements);
  bv.get(keys.begin(), keys.end(), get_outputs.begin());

  size_type num_set = thrust::reduce(thrust::device, get_outputs.begin(), get_outputs.end(), 0);
  REQUIRE(num_set == num_elements);
}
