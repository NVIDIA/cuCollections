/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <benchmark/benchmark.h>
#include <synchronization.hpp>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <fstream>
#include <iostream>
#include <random>
#include "cuco/static_reduction_map.cuh"

enum class dist_type { UNIQUE, UNIFORM, GAUSSIAN };

template <dist_type Dist, typename Key, typename OutputIt>
static void generate_keys(OutputIt output_begin, OutputIt output_end)
{
  auto num_keys = std::distance(output_begin, output_end);

  std::random_device rd;
  std::mt19937 gen{rd()};

  switch (Dist) {
    case dist_type::UNIQUE:
      for (auto i = 0; i < num_keys; ++i) {
        output_begin[i] = i;
      }
      break;
    case dist_type::UNIFORM:
      for (auto i = 0; i < num_keys; ++i) {
        output_begin[i] = std::abs(static_cast<Key>(gen()));
      }
      break;
    case dist_type::GAUSSIAN:
      std::normal_distribution<> dg{1e9, 1e7};
      for (auto i = 0; i < num_keys; ++i) {
        output_begin[i] = std::abs(static_cast<Key>(dg(gen)));
      }
      break;
  }
}

/**
 * @brief Generates input sizes and hash table occupancies
 *
 */
static void generate_size_and_occupancy(benchmark::internal::Benchmark* b)
{
  for (auto size = 4096; size <= 1 << 28; size *= 2) {
    for (auto occupancy = 60; occupancy <= 60; occupancy += 10) {
      b->Args({size, occupancy});
    }
  }
}

template <typename Key, typename Value, dist_type Dist, template <typename> typename ReductionOp>
static void BM_static_map_insert(::benchmark::State& state)
{
  using map_type = cuco::static_reduction_map<ReductionOp<Value>, Key, Value>;

  std::size_t num_keys = state.range(0);
  float occupancy      = state.range(1) / float{100};
  std::size_t size     = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<Value> d_values(h_keys);

  auto pairs_begin =
    thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_values.begin()));
  auto pairs_end = pairs_begin + num_keys;

  for (auto _ : state) {
    map_type map{size, -1};
    {
      cuda_event_timer raii{state};
      map.insert(pairs_begin, pairs_end);
    }
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) * int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

BENCHMARK_TEMPLATE(BM_static_map_insert, int32_t, int32_t, dist_type::UNIQUE, cuco::reduce_add)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime()
  ->Apply(generate_size_and_occupancy);

BENCHMARK_TEMPLATE(BM_static_map_insert, int32_t, int32_t, dist_type::UNIFORM, cuco::reduce_add)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime()
  ->Apply(generate_size_and_occupancy);

BENCHMARK_TEMPLATE(BM_static_map_insert, int32_t, int32_t, dist_type::GAUSSIAN, cuco::reduce_add)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime()
  ->Apply(generate_size_and_occupancy);

BENCHMARK_TEMPLATE(BM_static_map_insert, int64_t, int64_t, dist_type::UNIQUE, cuco::reduce_add)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime()
  ->Apply(generate_size_and_occupancy);

BENCHMARK_TEMPLATE(BM_static_map_insert, int64_t, int64_t, dist_type::UNIFORM, cuco::reduce_add)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime()
  ->Apply(generate_size_and_occupancy);

BENCHMARK_TEMPLATE(BM_static_map_insert, int64_t, int64_t, dist_type::GAUSSIAN, cuco::reduce_add)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime()
  ->Apply(generate_size_and_occupancy);