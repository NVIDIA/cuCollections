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

#include <defaults.hpp>
#include <utils.hpp>

#include "../../tests/trie/trie_utils.hpp"
#include <cuco/trie.cuh>
#include <cuco/utility/key_generator.hpp>

#include <nvbench/nvbench.cuh>

using namespace cuco::benchmark;
using namespace cuco::utility;

/**
 * @brief A benchmark evaluating `cuco::experimental::trie::insert` performance
 */
void trie_insert(nvbench::state& state)
{
  using LabelType = int;
  cuco::experimental::trie<LabelType> trie;

  auto const num_keys = 64 * 1024;
  std::vector<std::vector<LabelType>> keys;

  bool synthetic_dataset = true;
  if (synthetic_dataset) {
    thrust::host_vector<LabelType> labels;
    thrust::host_vector<size_t> offsets;
    auto const max_key_length = 6;
    generate_labels(labels, offsets, num_keys, max_key_length);
    keys = sorted_keys(labels, offsets);
  } else {
    keys = read_keys<LabelType>("trie_dataset.txt", num_keys);
  }

  state.add_element_count(num_keys);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    for (auto& key : keys) {
      trie.insert(key.begin(), key.end());
    }
  });
}

NVBENCH_BENCH(trie_insert).set_name("trie_insert").set_max_noise(defaults::MAX_NOISE);
