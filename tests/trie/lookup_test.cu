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

#include <cuco/trie.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <catch2/catch_test_macros.hpp>

struct valid_key {
  __host__ __device__ bool operator()(uint64_t x) const {
      return x != -1lu;
  }
};

TEST_CASE("Lookup test", "")
{

  using KeyType = int;
  cuco::experimental::trie<KeyType> trie;

  std::size_t num_keys = 3;
  thrust::host_vector<KeyType> flatten_keys = std::vector<KeyType>{1, 2, 3, 1, 2, 4, 1, 4, 2};
  thrust::host_vector<uint64_t> key_offsets = std::vector<KeyType>{0, 3, 6, 9};

  for (size_t key_id = 0; key_id < num_keys; key_id++) {
    std::vector<KeyType> cur_key;
    for (size_t pos = key_offsets[key_id]; pos < key_offsets[key_id + 1]; pos++) {
      cur_key.push_back(flatten_keys[pos]);
    }
    trie.add(cur_key);
  }

  trie.build();

  thrust::device_vector<uint64_t> lookup_result(num_keys, -1lu);
  {
    thrust::device_vector<KeyType> device_keys = flatten_keys;
    thrust::device_vector<uint64_t> device_offsets = key_offsets;

    trie.lookup(thrust::raw_pointer_cast(device_keys.data()),
                thrust::raw_pointer_cast(device_offsets.data()),
                thrust::raw_pointer_cast(lookup_result.data()), 3, 0, 0);

    thrust::host_vector<uint64_t> host_lookup_result = lookup_result;
    for (size_t key_id = 0; key_id < num_keys; key_id++) {
      REQUIRE(host_lookup_result[key_id] == key_id);
    }
  }

  thrust::transform(thrust::device, lookup_result.begin(), lookup_result.end(), lookup_result.begin(), valid_key());
  size_t num_matches = thrust::reduce(thrust::device, lookup_result.begin(), lookup_result.end(), 0);
  REQUIRE(num_matches == num_keys);
}
