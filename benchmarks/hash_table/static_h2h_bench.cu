#include <gpu_hash_table.cuh>
#include "cudf/concurrent_unordered_map.cuh"
#include <cuco/legacy_static_map.cuh>
#include <single_value_hash_table.cuh>

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
        output_begin[i] = i + 1;
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
static void BM_cudf_insert_thrust(::benchmark::State& state) {
  using map_type = concurrent_unordered_map<Key, Value>;

  std::size_t num_keys = state.range(0);
  float occupancy = state.range(1) / float{100};
  std::size_t size = num_keys / occupancy;
  
  std::vector<Key> h_keys( num_keys );
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  thrust::device_vector<Key> d_keys( h_keys );
  thrust::device_vector<Key> d_values( h_keys );

  auto key_iterator = d_keys.begin();
  auto value_iterator = d_values.begin();
  auto zip_counter = thrust::make_zip_iterator(
      thrust::make_tuple(key_iterator, value_iterator));
  
  for (auto _ : state) {
    auto map = map_type::create(size);
    auto view = *map;
    {
      cuda_event_timer raii{state};
      thrust::for_each(
          thrust::device, zip_counter, zip_counter + num_keys,
          [view] __device__(auto const& p) mutable {
            view.insert(thrust::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
          });
      cudaDeviceSynchronize();
    }
  }
  
  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
                          
}

template <typename Key, typename Value, dist_type Dist>
static void BM_legacy_static_map_insert_thrust(::benchmark::State& state) {
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
    auto m_view = map.get_device_mutable_view();
    {
      cuda_event_timer raii{state};
      thrust::for_each(thrust::device,
                        d_pairs.begin(),
                        d_pairs.end(),
                        [m_view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
                         m_view.insert(pair);
                        });
      cudaDeviceSynchronize();
    }
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_static_map_insert_thrust(::benchmark::State& state) {
  using map_type = cuco::static_map<Key, Value>;
  
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
    auto m_view = map.get_device_mutable_view();
    {
      cuda_event_timer raii{state};
      thrust::for_each(thrust::device,
                        d_pairs.begin(),
                        d_pairs.end(),
                        [m_view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
                         m_view.insert(pair);
                        });
      cudaDeviceSynchronize();
    }
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_cudf_find_thrust(::benchmark::State& state) {
  using map_type = concurrent_unordered_map<Key, Value>;

  std::size_t num_keys = state.range(0);
  float occupancy = state.range(1) / float{100};
  std::size_t size = num_keys / occupancy;
  
  std::vector<Key> h_keys( num_keys );
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  thrust::device_vector<Key> d_keys( h_keys );
  thrust::device_vector<Value> d_values( h_keys );
  thrust::device_vector<Value> d_results( num_keys );

  auto counter = thrust::make_counting_iterator(0);
  auto key_counter = thrust::make_zip_iterator(
    thrust::make_tuple(counter, d_keys.begin()));

  auto key_iterator = d_keys.begin();
  auto value_iterator = d_values.begin();
  auto zip_counter = thrust::make_zip_iterator(
      thrust::make_tuple(key_iterator, value_iterator));
  
  for (auto _ : state) {
    auto map = map_type::create(size);
    auto view = *map;
    thrust::for_each(
        thrust::device, zip_counter, zip_counter + num_keys,
        [view] __device__(auto const& p) mutable {
          view.insert(thrust::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
        });
    {
      cuda_event_timer raii{state};
      thrust::for_each(
          thrust::device, key_counter, key_counter + num_keys,
          [view, res = d_values.begin()] __device__(auto const& pair) mutable {
            view.find(thrust::get<1>(pair));
            *(res + thrust::get<0>(pair)) = 1;
          });
      cudaDeviceSynchronize();
    }
  }
  
  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
                          
}

template <typename Key, typename Value, dist_type Dist>
static void BM_legacy_static_map_find_thrust(::benchmark::State& state) {
  using map_type = cuco::legacy_static_map<Key, Value>;
  
  std::size_t num_keys = state.range(0);
  float occupancy = state.range(1) / float{100};
  std::size_t size = num_keys / occupancy;

  std::vector<Key> h_keys( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs( num_keys );
  
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  thrust::device_vector<Key> d_keys( h_keys );
  auto counter = thrust::make_counting_iterator(0);
  auto key_counter = thrust::make_zip_iterator(
    thrust::make_tuple(counter, d_keys.begin()));
  thrust::device_vector<Value> d_values( num_keys );
  
  for(auto i = 0; i < num_keys; ++i) {
    Key key = h_keys[i];
    Value val = h_keys[i];
    h_pairs[i].first = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  for(auto _ : state) {
    map_type map{size, -1, -1};
    map.insert(d_pairs.begin(), d_pairs.end());
    auto view = map.get_device_view();
    {
      cuda_event_timer raii{state};
      thrust::for_each(thrust::device,
                        key_counter,
                        key_counter + num_keys,
                        [view, res = d_values.begin()] __device__(auto const& pair) mutable {
                          *(res + thrust::get<0>(pair)) = view.find(thrust::get<1>(pair))->second;
                        });
      cudaDeviceSynchronize();
    }
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_static_map_find_thrust(::benchmark::State& state) {
  using map_type = cuco::static_map<Key, Value>;
  
  std::size_t num_keys = state.range(0);
  float occupancy = state.range(1) / float{100};
  std::size_t size = num_keys / occupancy;

  std::vector<Key> h_keys( num_keys );
  std::vector<cuco::pair_type<Key, Value>> h_pairs( num_keys );
  
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  thrust::device_vector<Key> d_keys( h_keys );
  auto counter = thrust::make_counting_iterator(0);
  auto key_counter = thrust::make_zip_iterator(
    thrust::make_tuple(counter, d_keys.begin()));
  thrust::device_vector<Value> d_values( num_keys );
  
  for(auto i = 0; i < num_keys; ++i) {
    Key key = h_keys[i];
    Value val = h_keys[i];
    h_pairs[i].first = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs( h_pairs );

  for(auto _ : state) {
    map_type map{size, -1, -1};
    map.insert(d_pairs.begin(), d_pairs.end());
    auto view = map.get_device_view();
    {
      cuda_event_timer raii{state};
      thrust::for_each(thrust::device,
                        key_counter,
                        key_counter + num_keys,
                        [view, res = d_values.begin()] __device__(auto const& pair) mutable {
                          *(res + thrust::get<0>(pair)) = view.find(thrust::get<1>(pair))->second;
                        });
      cudaDeviceSynchronize();
    }
  }
  
  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
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
static void BM_warpcore_find(::benchmark::State& state) {

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
static void BM_static_map_insert(::benchmark::State& state) {
  using map_type = cuco::static_map<Key, Value>;
  
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
static void BM_static_map_find(::benchmark::State& state) {
  using map_type = cuco::static_map<Key, Value>;
  
  std::size_t num_keys = state.range(0);
  float occupancy = state.range(1) / float{100};
  std::size_t size = num_keys / occupancy;

  map_type map{size, -1, -1};
  auto view = map.get_device_mutable_view();

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

BENCHMARK_TEMPLATE(BM_warpcore_insert, std::uint32_t, std::uint32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_legacy_static_map_insert, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_static_map_insert, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_warpcore_find, std::uint32_t, std::uint32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_legacy_static_map_find, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_static_map_find, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(gen_size_and_occupancy)
  ->UseManualTime();