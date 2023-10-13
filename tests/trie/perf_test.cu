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

#include <iomanip>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <catch2/catch_test_macros.hpp>

#include "trie_utils.hpp"

TEST_CASE("Perf test", "")
{
  using LabelType = int;

  const char* input_filename = "trie_dataset.txt";
  auto keys = generate_split_keys<LabelType>(read_input_keys(input_filename, 45 * 1000 * 1000));
  size_t num_keys = keys.size();
  std::cout << "Num keys " << num_keys << std::endl;

  auto begin = current_time();
  cuco::experimental::trie<LabelType> trie;
  for (auto& key : keys) {
    trie.insert(key.begin(), key.end());
  }
  auto insert_msec = elapsed_milliseconds(begin);

  std::cout << "Insert " << std::setprecision(2) << insert_msec / 1000. << "s @ ";
  std::cout << std::setprecision(2) << (1. * num_keys / insert_msec) / 1000 << "M keys/sec"
            << std::endl;

  begin = current_time();
  trie.build();
  auto build_msec = elapsed_milliseconds(begin);

  std::cout << "Build " << build_msec << "ms @ ";
  std::cout << std::setprecision(3) << (1. * num_keys / build_msec) / 1000 << "M keys/sec"
            << std::endl;

  std::random_shuffle(keys.begin(), keys.end());

  thrust::host_vector<size_t> lookup_offsets(num_keys + 1);
  lookup_offsets[0] = 0;
#pragma omp parallel for
  for (size_t i = 0; i < num_keys; i++) {
    lookup_offsets[i + 1] = keys[i].size();
  }
  std::partial_sum(lookup_offsets.begin(), lookup_offsets.end(), lookup_offsets.begin());

  thrust::host_vector<size_t> lookup_inputs(lookup_offsets.back());
#pragma omp parallel for
  for (size_t i = 0; i < num_keys; i++) {
    for (size_t pos = 0; pos < keys[i].size(); pos++) {
      lookup_inputs[lookup_offsets[i] + pos] = keys[i][pos];
    }
  }

  // std::cout << "Average key length " << std::setprecision(2)
  //          << 1. * lookup_offsets.back() / num_keys << std::endl;

  thrust::device_vector<LabelType> d_lookup_inputs = lookup_inputs;
  thrust::device_vector<size_t> d_lookup_offsets   = lookup_offsets;
  thrust::device_vector<size_t> d_lookup_result(num_keys, -1lu);

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  begin = current_time();
  trie.lookup(d_lookup_inputs.begin(),
              d_lookup_offsets.begin(),
              d_lookup_offsets.end(),
              d_lookup_result.begin(),
              stream);
  cudaStreamSynchronize(stream);
  auto lookup_msec = elapsed_milliseconds(begin);

  std::cout << "Lookup " << lookup_msec << "ms @ ";
  std::cout << std::setprecision(2) << (1. * num_keys / lookup_msec) / 1000.0 << "M keys/sec"
            << std::endl;

  REQUIRE(cuco::test::all_of(d_lookup_result.begin(), d_lookup_result.end(), valid_key(num_keys)));
}
