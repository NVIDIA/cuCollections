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
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include <catch2/catch_test_macros.hpp>

#include "trie_utils.hpp"

TEST_CASE("Lookup test", "")
{
  using LabelType = int;

  std::size_t num_keys       = 64 * 1024;
  std::size_t max_key_length = 6;
  thrust::host_vector<LabelType> keys;
  thrust::host_vector<size_t> offsets;

  generate_keys(keys, offsets, num_keys, max_key_length);

  cuco::experimental::trie<LabelType> trie;

  {
    std::vector<std::vector<LabelType>> all_keys;
    for (size_t key_id = 0; key_id < num_keys; key_id++) {
      std::vector<LabelType> cur_key;
      for (size_t pos = offsets[key_id]; pos < offsets[key_id + 1]; pos++) {
        cur_key.push_back(keys[pos]);
      }
      all_keys.push_back(cur_key);
    }

    sort(all_keys.begin(), all_keys.end(), vectorKeyCompare<LabelType>());

    for (auto key : all_keys) {
      trie.insert(key.begin(), key.end());
    }

    trie.build();
  }

  {
    thrust::device_vector<size_t> lookup_result(num_keys, -1lu);
    thrust::device_vector<LabelType> device_keys = keys;
    thrust::device_vector<size_t> device_offsets = offsets;

    trie.lookup(
      device_keys.begin(), device_offsets.begin(), device_offsets.end(), lookup_result.begin());

    REQUIRE(cuco::test::all_of(lookup_result.begin(), lookup_result.end(), valid_key(num_keys)));
  }
}
