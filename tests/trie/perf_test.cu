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

#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <unordered_set>

#include <catch2/catch_test_macros.hpp>

#include "common.hpp"

using namespace std;

vector<string> read_input_keys(const char* filename, size_t num_keys)
{
  ifstream input_file(filename);
  if (!input_file.is_open()) {
    std::cout << "Error opening file: " << filename << std::endl;
    exit(1);
  }
  vector<string> keys;
  string line;
  while (keys.size() < num_keys and getline(input_file, line)) {
    keys.push_back(line);
  }
  return keys;
}

template <typename KeyType>
vector<KeyType> split_str_into_ints(const string& key)
{
  stringstream ss(key);
  vector<KeyType> tokens;
  string buf;

  while (ss >> buf) {
    tokens.push_back(stoi(buf));
  }
  return tokens;
}

template <typename KeyType>
vector<vector<KeyType>> generate_split_keys(const vector<string>& keys)
{
  vector<vector<KeyType>> split_keys(keys.size());
#pragma omp parallel for
  for (size_t i = 0; i < keys.size(); i++) {
    split_keys[i] = split_str_into_ints<KeyType>(keys[i]);
  }
  return split_keys;
}

template <typename KeyType>
void find_pivots(const vector<vector<KeyType>>& keys,
                 std::vector<KeyType>& pivot_vals,
                 std::vector<size_t>& pivot_offsets)
{
  pivot_vals.push_back(keys[0][1]);
  pivot_offsets.push_back(0);

  for (size_t pos = 1; pos < keys.size(); pos++) {
    if (keys[pos][1] != keys[pos - 1][1]) {
      pivot_vals.push_back(keys[pos][1]);
      pivot_offsets.push_back(pos);
    }
  }
  pivot_offsets.push_back(keys.size());
}

TEST_CASE("Perf test", "")
{
  using KeyType = int;

  const char* input_filename = "trie_dataset.txt";
  auto keys       = generate_split_keys<KeyType>(read_input_keys(input_filename, 45 * 1000 * 1000));
  size_t num_keys = keys.size();
  std::cout << "Num keys " << num_keys << std::endl;

  auto begin = current_time();
  cuco::experimental::trie<KeyType> trie;
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

  thrust::device_vector<KeyType> d_lookup_inputs = lookup_inputs;
  thrust::device_vector<size_t> d_lookup_offsets = lookup_offsets;
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
