/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <nvbench/nvbench.cuh>
#include <nvbench/test_kernels.cuh>

#include <thrust/device_vector.h>
#include <random>
#include <type_traits>

#include "cuco/static_multimap.cuh"

template <typename Key, typename OutputIt>
static void generate_keys(OutputIt output_begin, OutputIt output_end, const std::string& dist)
{
  auto num_keys = std::distance(output_begin, output_end);

  std::random_device rd;
  std::mt19937 gen{rd()};

  if (dist == "UNIQUE") {
    for (auto i = 0; i < num_keys; ++i) {
      output_begin[i] = i;
    }
  } else if (dist == "UNIFORM") {
    std::uniform_int_distribution<Key> distribution{0, std::numeric_limits<Key>::max()};
    for (auto i = 0; i < num_keys; ++i) {
      output_begin[i] = distribution(gen);
    }
  } else if (dist == "GAUSSIAN") {
    std::normal_distribution<> dg{1e9, 1e7};
    for (auto i = 0; i < num_keys; ++i) {
      output_begin[i] = std::abs(static_cast<Key>(dg(gen)));
    }
  }
}

template <typename Key, typename Value>
void nvbench_static_multimap_insert(nvbench::state& state, nvbench::type_list<Key, Value>)
{
  if (not std::is_same<Key, Value>::value) {
    state.skip("Key should be the same type as Value.");
    return;
  }

  using map_type = cuco::static_multimap<Key, Value>;

  const std::size_t num_keys = state.get_int64("NumInputs");
  auto occupancy             = state.get_float64("Occupancy");
  std::size_t size           = num_keys / occupancy;
  const auto dist            = state.get_string("Distribution");

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Key>(h_keys.begin(), h_keys.end(), dist);

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               map_type map{size, -1, -1};
               auto m_view = map.get_device_mutable_view();

               auto const block_size = 128;
               auto const stride     = 1;
               auto const tile_size  = 4;
               auto const grid_size =
                 (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

               using Hash     = cuco::detail::MurmurHash3_32<Key>;
               using KeyEqual = thrust::equal_to<Key>;

               Hash hash;
               KeyEqual key_equal;

               timer.start();
               cuco::detail::insert<block_size, tile_size>
                 <<<grid_size, block_size, 0, launch.get_stream()>>>(
                   d_pairs.begin(), d_pairs.end(), m_view, hash, key_equal);
               // CUCO_CUDA_TRY(cudaDeviceSynchronize());
               timer.stop();
             });
}

using key_type   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using value_type = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
NVBENCH_BENCH_TYPES(nvbench_static_multimap_insert, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_type_axes_names({"Key", "Value"})
  .add_int64_axis("NumInputs", {100'000'000})
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1))
  .add_string_axis("Distribution", {"UNIQUE", "UNIFORM", "GAUSSIAN"});
