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

// Just to placate compiler!!
//#include <gpu_hash_table.cuh>
#include "/home/nico/Documents/hashmap/SlabHash/src/gpu_hash_table.cuh"
#include <cuco/legacy_static_map.cuh>
#include <cuco/static_map.cuh>
#include <single_value_hash_table.cuh>
#include <cuco/dynamic_map.cuh>

#include <thrust/device_vector.h>
#include <benchmark/benchmark.h>
#include <synchronization.hpp>
#include <iostream>
#include <random>

enum class dist_type {
  UNIQUE,
  UNIQUE_NONE,
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
        output_begin[i] = i + 1;
      }
      shuffle(output_begin, output_end, std::default_random_engine(14));
      break;
    case dist_type::UNIQUE_NONE:
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = i + 1 + num_keys;
      }
      shuffle(output_begin, output_end, std::default_random_engine(14));
      break;
    case dist_type::UNIFORM:
      // only works for Key = int32_t  
      for(auto i = 0; i < num_keys; ++i) {
        uint_fast32_t elem = gen();
        int32_t temp;
        std::memcpy(&temp, &elem, sizeof(int32_t)); // copy bits to int32_t
        temp = temp & 0x7FFFFFFF; // clear sign bit
        output_begin[i] = temp;
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
  for (auto size = 1<<27; size <= 1<<27; size *= 10) {
    for (auto occupancy = 40; occupancy <= 90; occupancy += 10) {
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
static void BM_warpcore_insert(::benchmark::State& state) {

  using map_type = warpcore::SingleValueHashTable<Key, Value>;
  
  std::size_t num_keys = state.range(0);
  float occupancy = state.range(1) / float{100};
  std::size_t size = num_keys / occupancy;
  
  std::vector<Key> h_keys( num_keys );
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  thrust::device_vector<Key> d_keys( h_keys );
  thrust::device_vector<Value> d_values( h_keys );


  for(auto _ : state) {
    map_type map{size};
    {
      cuda_event_timer raii{state}; 
      map.insert(d_keys.data().get(), d_values.data().get(), num_keys);
    }
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_warpcore_find_all(::benchmark::State& state) {

  using map_type = warpcore::SingleValueHashTable<Key, Value>;
  
  std::size_t num_keys = state.range(0);
  float occupancy = state.range(1) / float{100};
  std::size_t size = num_keys / occupancy;
  
  std::vector<Key> h_keys( num_keys );
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  thrust::device_vector<Key> d_keys( h_keys );
  thrust::device_vector<Value> d_values( h_keys );
  thrust::device_vector<Value> d_results( num_keys);

  map_type map{size};
  map.insert(d_keys.data().get(), d_values.data().get(), num_keys);

  for(auto _ : state) {
    {
      cuda_event_timer raii{state};
      map.retrieve(d_keys.data().get(), num_keys, d_results.data().get());
    }
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_warpcore_find_none(::benchmark::State& state) {

  using map_type = warpcore::SingleValueHashTable<Key, Value>;
  
  std::size_t num_keys = state.range(0);
  float occupancy = state.range(1) / float{100};
  std::size_t size = num_keys / occupancy;
  
  std::vector<Key> h_keys( num_keys );
  std::vector<Key> h_search_keys( num_keys );
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  generate_keys<dist_type::UNIQUE_NONE, Key>(h_search_keys.begin(), h_search_keys.end());
  thrust::device_vector<Key> d_keys( h_keys );
  thrust::device_vector<Key> d_search_keys( h_search_keys );
  thrust::device_vector<Value> d_values( h_keys );
  thrust::device_vector<Value> d_results( num_keys);

  map_type map{size};
  map.insert(d_keys.data().get(), d_values.data().get(), num_keys);

  for(auto _ : state) {
    {
      cuda_event_timer raii{state};
      map.retrieve(d_search_keys.data().get(), num_keys, d_results.data().get());
    }
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_dynamic_map_insert(::benchmark::State& state) {
  using map_type = cuco::dynamic_map<Key, Value,
                                     cuda::thread_scope_device,
                                     cuco::legacy_static_map>;
  
  std::size_t num_keys = state.range(0);
  float occupancy = state.range(1) / float{100};
  std::size_t size = num_keys / occupancy;

  std::vector<Key> h_keys( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs( num_keys );
  
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  
  for(auto i = 0; i < num_keys; ++i) {
    Key key = h_keys[i];
    Value val = h_keys[i];
    h_pairs[i].first = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  for(auto _ : state) {
    map_type map{size, -1, -1};
    {
      cuda_event_timer raii{state};
      map.insert(d_pairs.begin(), d_pairs.end());
    }
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_dynamic_map_find_all(::benchmark::State& state) {
  using map_type = cuco::dynamic_map<Key, Value,
                                     cuda::thread_scope_device,
                                     cuco::legacy_static_map>;
  
  std::size_t num_keys = state.range(0);
  float occupancy = state.range(1) / float{100};
  std::size_t size = num_keys / occupancy;

  map_type map{size, -1, -1};

  std::vector<Key> h_keys( num_keys );
  std::vector<Value> h_values( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  std::vector<Value> h_results (num_keys);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  
  for(auto i = 0; i < num_keys; ++i) {
    Key key = h_keys[i];
    Value val = h_keys[i];
    h_pairs[i].first = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<Key> d_keys( h_keys ); 
  thrust::device_vector<Value> d_results( num_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  map.insert(d_pairs.begin(), d_pairs.end());
  
  for(auto _ : state) {
    {
      cuda_event_timer raii{state};
      map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    }
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) * int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_dynamic_map_find_none(::benchmark::State& state) {
  using map_type = cuco::dynamic_map<Key, Value,
                                     cuda::thread_scope_device,
                                     cuco::legacy_static_map>;
  
  std::size_t num_keys = state.range(0);
  float occupancy = state.range(1) / float{100};
  std::size_t size = num_keys / occupancy;

  map_type map{size, -1, -1};

  std::vector<Key> h_keys( num_keys );
  std::vector<Key> h_search_keys( num_keys );
  std::vector<Value> h_values( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  std::vector<Value> h_results (num_keys);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  generate_keys<dist_type::UNIQUE_NONE, Key>(h_search_keys.begin(), h_search_keys.end());
  
  for(auto i = 0; i < num_keys; ++i) {
    Key key = h_keys[i];
    Value val = h_keys[i];
    h_pairs[i].first = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<Key> d_keys( h_keys );
  thrust::device_vector<Key> d_search_keys( h_search_keys );
  thrust::device_vector<Value> d_results( num_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  map.insert(d_pairs.begin(), d_pairs.end());
  
  for(auto _ : state) {
    {
      cuda_event_timer raii{state};
      map.find(d_search_keys.begin(), d_search_keys.end(), d_results.begin());
    }
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) * int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_legacy_static_map_insert(::benchmark::State& state) {
  using map_type = cuco::legacy_static_map<Key, Value>;
  
  std::size_t num_keys = state.range(0);
  float occupancy = state.range(1) / float{100};
  std::size_t size = num_keys / occupancy;

  std::vector<Key> h_keys( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs( num_keys );
  
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  
  for(auto i = 0; i < num_keys; ++i) {
    Key key = h_keys[i];
    Value val = h_keys[i];
    h_pairs[i].first = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  for(auto _ : state) {
    map_type map{size, -1, -1};
    {
      cuda_event_timer raii{state};
      map.insert(d_pairs.begin(), d_pairs.end());
    }
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_legacy_static_map_find(::benchmark::State& state) {
  using map_type = cuco::legacy_static_map<Key, Value>;
  
  std::size_t num_keys = state.range(0);
  float occupancy = state.range(1) / float{100};
  std::size_t size = num_keys / occupancy;

  map_type map{size, -1, -1};
  auto view = map.get_device_mutable_view();

  std::vector<Key> h_keys( num_keys );
  std::vector<Value> h_values( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  std::vector<Value> h_results( num_keys );

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  
  for(auto i = 0; i < num_keys; ++i) {
    Key key = h_keys[i];
    Value val = h_keys[i];
    h_pairs[i].first = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<Key> d_keys( h_keys ); 
  thrust::device_vector<Value> d_results( num_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  map.insert(d_pairs.begin(), d_pairs.end());
  
  for(auto _ : state) {
    {
      cuda_event_timer raii{state};
      map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    }
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) * int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_slabhash_insert_lf(::benchmark::State& state) {

  using map_type = gpu_hash_table<Key, Value, SlabHashTypeT::ConcurrentMap>;
  
  std::size_t num_keys = state.range(0);
  uint32_t base_size = 0.069 * (1<<30);
  std::size_t slab_size = 128;
  std::size_t num_buckets = base_size / slab_size;
  int64_t device_idx = 0;
  int64_t seed = 12;
  
  std::vector<Key> h_keys( num_keys );
  
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  std::vector<Value> h_values (h_keys);

  for(auto _ : state) {
    auto build_time = 0.0f;
    map_type map{num_keys, num_buckets, device_idx, seed};
    build_time += map.hash_build_with_unique_keys(h_keys.data(), 
                                                  h_values.data(), num_keys);
    state.SetIterationTime((float)build_time / 1000);
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_slabhash_find_all_lf(::benchmark::State& state) {

  using map_type = gpu_hash_table<Key, Value, SlabHashTypeT::ConcurrentMap>;
  
  std::size_t num_keys = state.range(0);

  std::size_t base_slabs_size = 0.069 * (1<<30);
  std::size_t slab_size = 128;
  std::size_t num_buckets = base_slabs_size / slab_size;
  //std::cout << "num_buckets: " << num_buckets << std::endl;
  int64_t device_idx = 0;
  int64_t seed = 12;
  
  std::vector<Key> h_keys( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  std::vector<Value> h_values (h_keys);
  std::vector<Value> h_results(num_keys);

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  map_type map{num_keys, num_buckets, device_idx, seed};
  map.hash_build_with_unique_keys(h_keys.data(), 
                                  h_values.data(), num_keys);
  
  for(auto _ : state) {
    auto find_time = 0.0f;
    find_time = map.hash_search(h_keys.data(), h_results.data(), num_keys);
    state.SetIterationTime((float)find_time / 1000);
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_slabhash_find_none_lf(::benchmark::State& state) {

  using map_type = gpu_hash_table<Key, Value, SlabHashTypeT::ConcurrentMap>;
  
  std::size_t num_keys = state.range(0);

  std::size_t base_slabs_size = 0.069 * (1<<30);
  std::size_t slab_size = 128;
  std::size_t num_buckets = base_slabs_size / slab_size;
  //std::cout << "num_buckets: " << num_buckets << std::endl;
  int64_t device_idx = 0;
  int64_t seed = 12;
  
  std::vector<Key> h_keys( num_keys );
  std::vector<Key> h_search_keys( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs ( num_keys );
  
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  generate_keys<dist_type::UNIQUE_NONE, Key>(h_search_keys.begin(), h_search_keys.end());
  std::vector<Value> h_values (h_keys);
  std::vector<Value> h_results(num_keys);

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  map_type map{num_keys, num_buckets, device_idx, seed};
  map.hash_build_with_unique_keys(h_keys.data(), 
                                  h_values.data(), num_keys);
  
  for(auto _ : state) {
    auto find_time = 0.0f;
    find_time = map.hash_search(h_search_keys.data(), h_results.data(), num_keys);
    state.SetIterationTime((float)find_time / 1000);
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

/*
BENCHMARK_TEMPLATE(BM_warpcore_insert, std::uint32_t, std::uint32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_warpcore_find_all, std::uint32_t, std::uint32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();
  
BENCHMARK_TEMPLATE(BM_warpcore_find_none, std::uint32_t, std::uint32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();
//*/

/*
BENCHMARK_TEMPLATE(BM_warpcore_insert, std::uint32_t, std::uint32_t, dist_type::UNIFORM)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();
//*/

/*
BENCHMARK_TEMPLATE(BM_warpcore_find_all, std::uint32_t, std::uint32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();
//*/

///*
BENCHMARK_TEMPLATE(BM_dynamic_map_insert, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();
//*/

/*
BENCHMARK_TEMPLATE(BM_dynamic_map_find_none, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();
//*/

///*
BENCHMARK_TEMPLATE(BM_dynamic_map_find_all, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();
//*/

/*
BENCHMARK_TEMPLATE(BM_dynamic_map_find_all, int32_t, int32_t, dist_type::UNIFORM)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();
//*/

/*
BENCHMARK_TEMPLATE(BM_slabhash_insert_lf, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_slabhash_find_all_lf, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_slabhash_find_none_lf, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();
*/

/*
BENCHMARK_TEMPLATE(BM_slabhash_insert_lf, int32_t, int32_t, dist_type::UNIFORM)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_slabhash_find_all_lf, int32_t, int32_t, dist_type::UNIFORM)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_slabhash_find_none_lf, int32_t, int32_t, dist_type::UNIFORM)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();
*/