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

#include <cuco/detail/trie/dynamic_bitset/dynamic_bitset.cuh>
#include <cuco/utility/key_generator.hpp>

#include <nvbench/nvbench.cuh>

#include <thrust/host_vector.h>

using namespace cuco::benchmark;
using namespace cuco::utility;

/**
 * @brief A benchmark evaluating `cuco::experimental::detail::dynamic_bitset::size` performance
 */
template <typename Dist>
void dynamic_bitset_size(nvbench::state& state, nvbench::type_list<Dist>)
{
  auto const num_bits      = state.get_int64_or_default("NumInputs", defaults::N);
  using word_type          = typename cuco::experimental::detail::dynamic_bitset<>::word_type;
  auto const bits_per_word = cuco::experimental::detail::dynamic_bitset<>::bits_per_word;
  thrust::host_vector<word_type> keys((num_bits - 1) / bits_per_word + 1);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  cuco::experimental::detail::dynamic_bitset bitset;
  bitset.insert(keys.begin(), keys.end(), num_bits);

  state.add_element_count(1);
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto const size = bitset.size(); });
}

NVBENCH_BENCH_TYPES(dynamic_bitset_size,
                    NVBENCH_TYPE_AXES(nvbench::type_list<distribution::uniform>))
  .set_name("dynamic_bitset_size")
  .set_type_axes_names({"Distribution"})
  .set_max_noise(defaults::MAX_NOISE);
