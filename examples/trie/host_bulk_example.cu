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

#include <../tests/trie/utils.hpp>
#include <cuco/trie.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/scan.h>

using namespace cuco::utility;

/**
 * @file host_bulk_example.cu
 * @brief Demonstrates usage of the trie "bulk" host APIs.
 *
 * The bulk APIs are only invocable from the host and are used for doing operations like `insert` or
 * `lookup` on a set of keys.
 *
 */
int main(void)
{
  using LabelType = int;

  std::size_t num_keys       = 64 * 1024;
  std::size_t max_key_length = 6;

  thrust::host_vector<LabelType> labels;
  thrust::host_vector<size_t> offsets;

  distribution::unique lengths_dist;
  distribution::gaussian labels_dist{0.5};

  cuco::test::trie::generate_labels(
    labels, offsets, num_keys, max_key_length, lengths_dist, labels_dist);
  auto keys = cuco::test::trie::sorted_keys(labels, offsets);

  cuco::experimental::trie<LabelType> trie;
  for (auto key : keys) {
    trie.insert(key.begin(), key.end());
  }
  trie.build();

  thrust::device_vector<LabelType> d_labels = labels;
  thrust::device_vector<size_t> d_offsets   = offsets;
  thrust::device_vector<size_t> result(num_keys, -1lu);

  trie.lookup(d_labels.begin(), d_offsets.begin(), d_offsets.end(), result.begin());

  bool const all_keys_found =
    thrust::all_of(result.begin(), result.end(), cuco::test::trie::valid_key(num_keys));
  if (all_keys_found) { std::cout << "Success! Found all keys.\n"; }

  return 0;
}
