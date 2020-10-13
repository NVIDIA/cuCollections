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

#include <gpu_hash_table.cuh>
#include <cuco/legacy_static_map.cuh>
#include <single_value_hash_table.cuh>

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
        output_begin[i] = std::abs(static_cast<long>(gen()));
      }
      break;
    case dist_type::GAUSSIAN:
      std::normal_distribution<> dg{1e9, 1e7};
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = std::abs(static_cast<long>(dg(gen)));
      }
      break;
  }
}

static void gen_final_size(benchmark::internal::Benchmark* b) {
  for(auto size = 10'000'000; size <= 150'000'000; size += 20'000'000) {
    b->Args({size});
  }
}

/**
 * @brief Generates input sizes and hash table occupancies
 *
 */
static void gen_size_and_occupancy(benchmark::internal::Benchmark* b) {
  for (auto size = 100'000'000; size <= 100'000'000; size *= 10) {
    for (auto occupancy = 10; occupancy <= 90; occupancy += 10) {
      b->Args({size, occupancy});
    }
  }
}

static void gen_size_and_slab_count(benchmark::internal::Benchmark *b) {
  for (auto size = 10'000'000; size <= 10'000'000; size *= 2) {
    for(auto deciSPBAvg = 1; deciSPBAvg <= 20; ++deciSPBAvg) {
      b->Args({size, deciSPBAvg});
    }
  }
}

template <typename Key, typename Value, dist_type Dist>
static void BM_dynamic_insert(::benchmark::State& state) {

  using map_type = cuco::dynamic_map<Key, Value,
                                     cuda::thread_scope_device,
                                     cuco::legacy_static_map>;
  
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
static void BM_dynamic_find(::benchmark::State& state) {
  using map_type = cuco::dynamic_map<Key, Value, 
                                     cuda::thread_scope_device, 
                                     cuco::legacy_static_map>;
  
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

template <typename Key, typename Value, dist_type Dist>
static void BM_slabhash_insert(::benchmark::State& state) {

  using map_type = gpu_hash_table<Key, Value, SlabHashTypeT::ConcurrentMap>;
  
  std::size_t num_keys = state.range(0);
  std::size_t num_buckets = 1<<21;//6'291'496; // 732 MB buckets + 268 MB pool
  int64_t device_idx = 0;
  int64_t seed = 12;
  
  std::vector<Key> h_keys( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  std::vector<Value> h_values (h_keys);

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  std::size_t batch_size = 10E6;
  for(auto _ : state) {
    auto build_time = 0.0f;
    map_type map{batch_size, num_buckets, device_idx, seed};
    for(uint32_t i = 0; i < num_keys; i += batch_size) {
      build_time += map.hash_build_with_unique_keys(h_keys.data() + i, 
                                                    h_values.data() + i, batch_size);
    }
    state.SetIterationTime((float)build_time / 1000);
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_slabhash_insert_lf(::benchmark::State& state) {

  using map_type = gpu_hash_table<Key, Value, SlabHashTypeT::ConcurrentMap>;
  
  std::size_t num_keys = state.range(0);
  
  float num_slabs_per_bucket_avg = state.range(1) / float{10};
  uint32_t num_keys_per_slab = 15u;
  uint32_t num_keys_per_bucket_avg = num_slabs_per_bucket_avg * num_keys_per_slab;
  uint32_t num_buckets = 
    (num_keys + num_keys_per_bucket_avg - 1) / num_keys_per_bucket_avg;

  int64_t device_idx = 0;
  int64_t seed = 12;
  
  std::vector<Key> h_keys( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  std::vector<Value> h_values (h_keys);

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  bool lf_printed = false;

  for(auto _ : state) {
    auto build_time = 0.0f;
    map_type map{num_keys, num_buckets, device_idx, seed};
    build_time += map.hash_build_with_unique_keys(h_keys.data(), 
                                                  h_values.data(), num_keys);
    state.SetIterationTime((float)build_time / 1000);

    if(!lf_printed) {
      std::cout << "load_factor " << map.measureLoadFactor() << std::endl;
      lf_printed = true;
    }
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
*/

/*
BENCHMARK_TEMPLATE(BM_dynamic_search_all, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();
*/

/*
BENCHMARK_TEMPLATE(BM_dynamic_insert, int32_t, int32_t, dist_type::GAUSSIAN)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_dynamic_search_all, int32_t, int32_t, dist_type::GAUSSIAN)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();
*/

/*
BENCHMARK_TEMPLATE(BM_slabhash_insert, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_final_size)
  ->UseManualTime();
*/

BENCHMARK_TEMPLATE(BM_slabhash_insert_lf, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_slab_count)
  ->UseManualTime();