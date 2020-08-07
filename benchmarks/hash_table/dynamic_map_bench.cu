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
  GAUSSIAN,
  SUM_TEST
};

template<dist_type Dist, typename Key, std::size_t num_sum_duplicates = 1, typename OutputIt>
static void generate_keys(OutputIt output_begin, OutputIt output_end) {
  auto num_keys = std::distance(output_begin, output_end);
  
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::normal_distribution<> dg{1e9, 1e7};

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
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = std::abs(static_cast<Key>(dg(gen)));
      }
      break;
    case dist_type::SUM_TEST:
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = i / num_sum_duplicates;
      }
      break;
  }
}

static void gen_final_size(benchmark::internal::Benchmark* b) {
  for(auto size = 10'000'000; size <= 310'000'000; size += 50'000'000) {
    b->Args({size});
  }
}

template<dist_type Dist, typename Value, typename InputIt, typename OutputIt>
static void generate_values(InputIt first, InputIt last, OutputIt output_begin) {
  auto num_keys = std::distance(first, last);

  switch(Dist) {
    case dist_type::SUM_TEST:
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = 1;
      }
      break;
    case dist_type::UNIQUE:
    case dist_type::UNIFORM:
    case dist_type::GAUSSIAN:
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = first[i];
      }
      break;
  }
}

template <typename KeyIt, typename ValueIt, typename OutputIt>
static void generate_pairs(KeyIt keys_begin, KeyIt keys_end, ValueIt values_begin, OutputIt output_begin) {
  auto num_keys = std::distance(keys_begin, keys_end);

  for(auto i = 0; i < num_keys; ++i) {
    output_begin[i].first = keys_begin[i];
    output_begin[i].second = values_begin[i];
  }
}

template <typename KeyIt, typename OutputIt>
static void generate_pairs(KeyIt keys_begin, KeyIt keys_end, OutputIt output_begin) {
  auto num_keys = std::distance(keys_begin, keys_end);

  for(auto i = 0; i < num_keys; ++i) {
    output_begin[i].first = keys_begin[i];
    output_begin[i].second = keys_begin[i];
  }
}



template <typename Key, typename Value, dist_type Dist>
static void BM_dynamic_insert(::benchmark::State& state) {
  using map_type = cuco::dynamic_map<Key, Value>;
  
  std::size_t const num_keys = state.range(0);
  std::size_t const initial_size = 1<<27;
  
  std::vector<Key> h_keys( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  generate_pairs(h_keys.begin(), h_keys.end(), h_pairs.begin());

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  std::size_t batch_size = 1E6;
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
static void BM_dynamic_insertSumReduce(::benchmark::State& state) {
  using map_type = cuco::dynamic_map<Key, Value>;
  
  std::size_t num_keys = state.range(0);
  std::size_t initial_size = 1<<26;
  std::size_t const num_sum_duplicates = 1;
  
  std::vector<Key> h_keys( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  
  generate_keys<Dist, Key, num_sum_duplicates>(h_keys.begin(), h_keys.end());
  generate_pairs(h_keys.begin(), h_keys.end(), h_pairs.begin());

  thrust::device_vector<Key> d_keys( h_keys );
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );
  thrust::device_vector<Value> d_results( num_keys );  

  std::size_t batch_size = 1E7;
  for(auto _ : state) {
    map_type map{initial_size, -1, -1};
    {
      cuda_event_timer raii{state}; 
      for(auto i = 0; i < num_keys / batch_size; ++i) {
        map.insertSumReduce(d_pairs.begin() + i * batch_size, d_pairs.begin() + (i + 1) * batch_size);
      }
    }
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}



template <typename Key, typename Value, dist_type Dist>
static void BM_static_insertSumReduce(::benchmark::State& state) {
  using map_type = cuco::static_map<Key, Value>;
  
  std::size_t num_keys = state.range(0);
  std::size_t initial_size = 1<<26;
  std::size_t const num_sum_duplicates = 1;
  
  std::vector<Key> h_keys( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  
  generate_keys<Dist, Key, num_sum_duplicates>(h_keys.begin(), h_keys.end());
  generate_pairs(h_keys.begin(), h_keys.end(), h_pairs.begin());
  
  thrust::device_vector<Key> d_keys( h_keys );
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );
  thrust::device_vector<Value> d_results( num_keys );  

  float resize_thresh = 0.60;
  std::size_t batch_size = 1E7;
  for(auto _ : state) {
    map_type map{initial_size, -1, -1};
    {
      cuda_event_timer raii{state};
      for(auto i = 0; i < num_keys / batch_size; ++i) {
        if(map.get_size() + batch_size > resize_thresh * map.get_capacity()) {
          map.resize();
          map.insertSumReduce(d_pairs.begin(), d_pairs.begin() + i * batch_size);
          //std::cout << "resizing at " << map.get_size() << " and reinserting " << i * batch_size << " keys" << std::endl;
        }
        map.insertSumReduce(d_pairs.begin() + i * batch_size, d_pairs.begin() + (i + 1) * batch_size);
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
  generate_pairs(h_keys.begin(), h_keys.end(), h_pairs.begin());
  
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


/*
BENCHMARK_TEMPLATE(BM_dynamic_insert, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_dynamic_search_all, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();

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
*/

BENCHMARK_TEMPLATE(BM_dynamic_insertSumReduce, int32_t, int32_t, dist_type::SUM_TEST)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_static_insertSumReduce, int32_t, int32_t, dist_type::SUM_TEST)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();