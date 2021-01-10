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
#include <cuco/dynamic_map.cuh>
#include <iostream>
#include <random>

enum class dist_type {
  UNIQUE,
  UNIFORM,
  GAUSSIAN
};

template<dist_type Dist, typename Key, typename OutputIt>
static void generate_keys(OutputIt output_begin, OutputIt output_end) {
  auto num_keys = std::distance(output_begin, output_end);
  
  std::random_device rd;
  std::mt19937 gen{rd()};

  switch(Dist) {
    case dist_type::UNIQUE:
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = i;
      }
      break;
    case dist_type::UNIFORM:
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = std::abs(static_cast<Key>(gen()));
      }
      break;
    case dist_type::GAUSSIAN:
      std::normal_distribution<> dg{1e9, 1e7};
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = std::abs(static_cast<Key>(dg(gen)));
      }
      break;
  }
}

static void gen_final_size(benchmark::internal::Benchmark* b) {
  for(auto size = 10'000'000; size <= 150'000'000; size += 20'000'000) {
    b->Args({size});
  }
}

template <typename Key, typename Value, dist_type Dist>
static void BM_dynamic_insert(::benchmark::State& state) {
  using map_type = cuco::dynamic_map<Key, Value>;
  
  std::size_t num_keys = state.range(0);
  std::size_t initial_size = 1<<27;
  
  std::vector<Key> h_keys( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for(auto i = 0; i < num_keys; ++i) {
    Key key = h_keys[i];
    Value val = h_keys[i];
    h_pairs[i].first = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  std::size_t batch_size = 10E6;
  for(auto _ : state) {
    map_type map{initial_size, -1, -1};
    {
      cuda_event_timer raii{state}; 
      for(auto i = 0; i < num_keys; i += batch_size) {
        map.insert(d_pairs.begin() + i, d_pairs.begin() + i + batch_size);
      }
    }
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_dynamic_search_all(::benchmark::State& state) {
  using map_type = cuco::dynamic_map<Key, Value>;
  
  std::size_t num_keys = state.range(0);
  std::size_t initial_size = 1<<27;

  std::vector<Key> h_keys( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  
  for(auto i = 0; i < num_keys; ++i) {
    Key key = h_keys[i];
    Value val = h_keys[i];
    h_pairs[i].first = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<Key> d_keys( h_keys );
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );
  thrust::device_vector<Value> d_results( num_keys );

  map_type map{initial_size, -1, -1};
  map.insert(d_pairs.begin(), d_pairs.end());

  for(auto _ : state) {
    cuda_event_timer raii{state};
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

BENCHMARK_TEMPLATE(BM_dynamic_insert, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_dynamic_search_all, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();
/*
BENCHMARK_TEMPLATE(BM_dynamic_insert, int32_t, int32_t, dist_type::UNIFORM)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_dynamic_search_all, int32_t, int32_t, dist_type::UNIFORM)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_dynamic_insert, int32_t, int32_t, dist_type::GAUSSIAN)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_dynamic_search_all, int32_t, int32_t, dist_type::GAUSSIAN)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();
  
BENCHMARK_TEMPLATE(BM_dynamic_insert, int64_t, int64_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_dynamic_search_all, int64_t, int64_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_dynamic_insert, int64_t, int64_t, dist_type::UNIFORM)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_dynamic_search_all, int64_t, int64_t, dist_type::UNIFORM)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_dynamic_insert, int64_t, int64_t, dist_type::GAUSSIAN)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_dynamic_search_all, int64_t, int64_t, dist_type::GAUSSIAN)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();
*/