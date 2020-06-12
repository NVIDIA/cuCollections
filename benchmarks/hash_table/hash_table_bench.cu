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

#include <cuco/insert_only_hash_array.cuh>
#include "../nvtx3.hpp"
#include "cudf/concurrent_unordered_map.cuh"
#include "SlabHash/src/gpu_hash_table.cuh"

#include <thrust/for_each.h>
#include <iostream>



/**
 * @brief Generates input sizes and hash table occupancies
 *
 */
static void cuSweepLoad(benchmark::internal::Benchmark* b) {
  for (auto occupancy = 40; occupancy <= 90; occupancy += 10) {
    for (auto size = 10'000'000; size <= 10'000'000; size *= 2) {
      b->Args({size, occupancy});
    }
  }
}

/**
 * @brief Generates input sizes and hash table occupancies
 *
 */
static void cuSweepSize(benchmark::internal::Benchmark* b) {
  for (auto occupancy = 55; occupancy <= 55; occupancy += 10) {
    for (auto size = 100'000; size <= 100'000'000; size *= 2) {
      b->Args({size, occupancy});
    }
  }
}

/**
 * @brief Generates input sizes and number of buckets for SlabHash
 */
static void SlabSweepSize(benchmark::internal::Benchmark *b) {
  for (auto size = 100'000; size <= 100'000'000; size *= 2) {
    for(auto deciSPBAvg = 6; deciSPBAvg <= 6; ++deciSPBAvg) {
      b->Args({size, deciSPBAvg});
    }
  }
}

/**
 * @brief Generates input sizes and number of buckets for SlabHash
 */
static void SlabSweepLoad(benchmark::internal::Benchmark *b) {
  for (auto size = 10'000'000; size <= 10'000'000; size *= 2) {
    for(auto deciSPBAvg = 1; deciSPBAvg <= 20; ++deciSPBAvg) {
      b->Args({size, deciSPBAvg});
    }
  }
}

static void ResizeSweep(benchmark::internal::Benchmark *b) {
  for(auto size = 300'000'000; size <= 300'000'000; size *= 2) {
    b->Args({size});
  }
}



/**
 * @brief Benchmark inserting all unique keys of a given number with specified
 * hash table occupancy
 */
template <typename Key, typename Value>
static void BM_cuco_insert_random_keys(::benchmark::State& state) {
  using map_type =
      cuco::insert_only_hash_array<Key, Value, cuda::thread_scope_device>;

  auto occupancy = (state.range(1) / double{100});
  auto capacity = state.range(0) / occupancy;
  auto numKeys = state.range(0);
    
  std::mt19937 rng(/* seed = */ 12);
  std::vector<Key> h_keys(numKeys);
  std::vector<Value> h_values(numKeys);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = int32_t(rng() & 0x7FFFFFFF);
    h_values[i] = ~h_keys[i];
  }

  thrust::device_vector<Key> d_keys(numKeys);
  thrust::device_vector<Key> d_values(numKeys);
  thrust::copy(h_keys.begin(), h_keys.end(), d_keys.begin());
  thrust::copy(h_values.begin(), h_values.end(), d_values.begin());

  auto key_iterator = d_keys.begin();
  auto value_iterator = d_values.begin();
  auto zip_counter = thrust::make_zip_iterator(
  thrust::make_tuple(key_iterator, value_iterator));


  for (auto _ : state) {
    
    state.ResumeTiming();
    state.PauseTiming();
    map_type map{capacity, -1};
    auto view = map.get_device_view();
    state.ResumeTiming();
  
    thrust::for_each(
      thrust::device, zip_counter, zip_counter + state.range(0),
      [view] __device__(auto const& p) mutable {
        view.insert(cuco::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
      });
    
    state.PauseTiming();
  }
  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}



/**
 * @brief Benchmark inserting all unique keys of a given number with specified
 * hash table occupancy
 */
template <typename Key, typename Value>
static void BM_cuco_search_all(::benchmark::State& state) {
  using map_type =
      cuco::insert_only_hash_array<Key, Value, cuda::thread_scope_device>;

  auto occupancy = (state.range(1) / double{100});
  auto capacity = state.range(0) / occupancy;
  auto numKeys = state.range(0);
  
  map_type map{capacity, -1};
  auto view = map.get_device_view();

  std::mt19937 rng(/* seed = */ 12);
  std::vector<Key> h_keys(numKeys);
  std::vector<Value> h_values(numKeys);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = int32_t(rng() & 0x7FFFFFFF);
    h_values[i] = ~h_keys[i];
  }

  thrust::device_vector<Key> d_keys(numKeys);
  thrust::device_vector<Key> d_values(numKeys);
  thrust::copy(h_keys.begin(), h_keys.end(), d_keys.begin());
  thrust::copy(h_values.begin(), h_values.end(), d_values.begin());

  auto key_iterator = d_keys.begin();
  auto value_iterator = d_values.begin();
  auto zip_counter = thrust::make_zip_iterator(
      thrust::make_tuple(key_iterator, value_iterator));

  thrust::for_each(
      thrust::device, zip_counter, zip_counter + state.range(0),
      [view] __device__(auto const& p) mutable {
        view.insert(cuco::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
      });

  for (auto _ : state) {
    thrust::for_each(
        thrust::device, key_iterator, key_iterator + state.range(0),
        [view] __device__(auto const& p) mutable {
          view.find(p);
        });
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}



template <typename Key, typename Value>
static void BM_cudf_insert_random_keys(::benchmark::State& state) {
  using map_type = concurrent_unordered_map<Key, Value>;

  auto occupancy = (state.range(1) / double{100});
  auto capacity = state.range(0) / occupancy;
  auto numKeys = state.range(0);
  
  std::mt19937 rng(/* seed = */ 12);
  std::vector<Key> h_keys(numKeys);
  std::vector<Value> h_values(numKeys);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = int32_t(rng() & 0x7FFFFFFF);
    h_values[i] = ~h_keys[i];
  }

  thrust::device_vector<Key> d_keys(numKeys);
  thrust::device_vector<Key> d_values(numKeys);
  thrust::copy(h_keys.begin(), h_keys.end(), d_keys.begin());
  thrust::copy(h_values.begin(), h_values.end(), d_values.begin());

  auto key_iterator = d_keys.begin();
  auto value_iterator = d_values.begin();
  auto zip_counter = thrust::make_zip_iterator(
      thrust::make_tuple(key_iterator, value_iterator));
  
  for (auto _ : state) {
    
    state.ResumeTiming();
    state.PauseTiming();
    auto map = map_type::create(capacity);
    auto view = *map;
    state.ResumeTiming();
    
    thrust::for_each(
        thrust::device, zip_counter, zip_counter + state.range(0),
        [view] __device__(auto const& p) mutable {
          view.insert(thrust::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
        });

    state.PauseTiming();
  }

  
  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
                          
}



template <typename Key, typename Value>
static void BM_cudf_search_all(::benchmark::State& state) {
  using map_type = concurrent_unordered_map<Key, Value>;

  auto occupancy = (state.range(1) / double{100});
  auto capacity = state.range(0) / occupancy;
  auto numKeys = state.range(0);
  
  auto map = map_type::create(capacity);
  auto view = *map;
  
  std::mt19937 rng(/* seed = */ 12);
  std::vector<Key> h_keys(numKeys);
  std::vector<Value> h_values(numKeys);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = int32_t(rng() & 0x7FFFFFFF);
    h_values[i] = ~h_keys[i];

  }

  thrust::device_vector<Key> d_keys(numKeys);
  thrust::device_vector<Key> d_values(numKeys);
  thrust::copy(h_keys.begin(), h_keys.end(), d_keys.begin());
  thrust::copy(h_values.begin(), h_values.end(), d_values.begin());

  auto key_iterator = d_keys.begin();
  auto value_iterator = d_values.begin();
  auto zip_counter = thrust::make_zip_iterator(
      thrust::make_tuple(key_iterator, value_iterator));

  thrust::for_each(
      thrust::device, zip_counter, zip_counter + state.range(0),
      [view] __device__(auto const& p) mutable {
        view.insert(thrust::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
      });

  for (auto _ : state) {
    thrust::for_each(
      thrust::device, key_iterator, key_iterator + state.range(0),
      [view] __device__(auto const& p) mutable {
        view.find(p);
      });
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}



template <typename Key, typename Value>
static void BM_cudf_insert_resize(::benchmark::State& state) {
  using map_type = concurrent_unordered_map<Key, Value>;

  auto numKeys = 300'000'000;
  auto resizeOccupancy = 0.60;
  
  std::mt19937 rng(/* seed = */ 12);
  std::vector<Key> h_keys(numKeys);
  std::vector<Value> h_values(numKeys);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = int32_t(rng() & 0x7FFFFFFF);
    h_values[i] = ~h_keys[i];
  }

  thrust::device_vector<Key> d_keys(numKeys);
  thrust::device_vector<Key> d_values(numKeys);
  thrust::copy(h_keys.begin(), h_keys.end(), d_keys.begin());
  thrust::copy(h_values.begin(), h_values.end(), d_values.begin());

  auto key_iterator = d_keys.begin();
  auto value_iterator = d_values.begin();
  auto zip_counter = thrust::make_zip_iterator(
      thrust::make_tuple(key_iterator, value_iterator));
    
  for (auto _ : state) {
    
    state.ResumeTiming();
    state.PauseTiming();
    auto map = map_type::create(294'000'000);
    auto view = *map;
    state.ResumeTiming();
    
    thrust::for_each(
        thrust::device, zip_counter, zip_counter + (uint32_t) (resizeOccupancy * numKeys),
        [view] __device__(auto const& p) mutable {
          view.insert(thrust::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
        });

    state.PauseTiming();
    map = map_type::create(428'000'000);
    view = *map;
    state.ResumeTiming();

    thrust::for_each(
        thrust::device, zip_counter, zip_counter + numKeys,
        [view] __device__(auto const& p) mutable {
          view.insert(thrust::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
        });

    state.PauseTiming();
  }
}



template <typename Key, typename Value>
static void BM_slabhash_insert_random_keys(::benchmark::State& state) {
  /* 
   * To adjust the occupancy, we fix the average number of slabs per bucket and
   * instead manipulate the total number of buckets
   */
  
  using map_type = gpu_hash_table<Key, Value, SlabHashTypeT::ConcurrentMap>;

  uint32_t numKeys = state.range(0);
  float numSlabsPerBucketAvg = state.range(1) / float{10};
  uint32_t numKeysPerSlab = 15u;
  uint32_t numKeysPerBucketAvg = numSlabsPerBucketAvg * numKeysPerSlab;
  uint32_t numBuckets = 
    (numKeys + numKeysPerBucketAvg - 1) / numKeysPerBucketAvg;
  const uint32_t deviceIdx = 0;
  const int64_t seed = 1;

  
  // initialize key-value pairs for insertion
  std::mt19937 rng(/* seed = */ 12);
  Key *keys = new Key[numKeys];
  Value *values = new Value[numKeys];
  for(auto i = 0; i < numKeys; ++i) {
    keys[i] = int32_t(rng() & 0x7FFFFFFF);
    values[i] = ~keys[i];
  }

  auto avgLoadFactor = 0.0;
  auto buildTime = 0.0;

  for(auto _ : state) {
    state.PauseTiming();
    map_type map{numKeys, numBuckets, deviceIdx, seed};
    state.ResumeTiming();

    buildTime += map.hash_build(keys, values, numKeys);

    state.PauseTiming();
    avgLoadFactor += map.measureLoadFactor();
    state.ResumeTiming();
  }
  
  buildTime /= state.iterations();
  avgLoadFactor /= state.iterations();

  state.counters["buildTime"] = buildTime;
  state.counters["loadFactor"] = avgLoadFactor;
  state.counters["GBytesPerSecond"] = (sizeof(Key) + sizeof(Value)) *
                                     int64_t(state.range(0)) / (1'000'000 * buildTime);
}



template <typename Key, typename Value>
static void BM_slabhash_search_all(::benchmark::State& state) {
  /* 
   * To adjust the occupancy, we fix the average number of slabs per bucket and
   * instead manipulate the total number of buckets
   */
  
  using map_type = gpu_hash_table<Key, Value, SlabHashTypeT::ConcurrentMap>;

  uint32_t numKeys = state.range(0);
  float numSlabsPerBucketAvg = state.range(1) / float{10};
  uint32_t numKeysPerSlab = 15u;
  uint32_t numKeysPerBucketAvg = numSlabsPerBucketAvg * numKeysPerSlab;
  uint32_t numBuckets = 
    (numKeys + numKeysPerBucketAvg - 1) / numKeysPerBucketAvg;
  const uint32_t deviceIdx = 0;
  const int64_t seed = 1;

  
  // initialize key-value pairs for insertion
  std::mt19937 rng(/* seed = */ 12);
  Key *keys = new Key[numKeys];
  Value *values = new Value[numKeys];
  for(auto i = 0; i < numKeys; ++i) {
    keys[i] = int32_t(rng() & 0x7FFFFFFF);
    values[i] = ~keys[i];
  }
  Value *results = new Value[numKeys];

  auto avgLoadFactor = 0.0;
  auto searchTime = 0.0;

  for(auto _ : state) {
    state.PauseTiming();
    map_type map{numKeys, numBuckets, deviceIdx, seed};
    map.hash_build(keys, values, numKeys);
    state.ResumeTiming();

    searchTime += map.hash_search(keys, results, numKeys);

    state.PauseTiming();
    avgLoadFactor += map.measureLoadFactor();
    state.ResumeTiming();
  }
  
  searchTime /= state.iterations();
  avgLoadFactor /= state.iterations();

  state.counters["searchTime"] = searchTime;
  state.counters["loadFactor"] = avgLoadFactor;
  state.counters["GBytesPerSecond"] = (sizeof(Key) + sizeof(Value)) *
                                     int64_t(state.range(0)) / (1'000'000 * searchTime);
}



template <typename Key, typename Value>
static void BM_slabhash_insert_resize(::benchmark::State& state) {
  /* 
   * To adjust the occupancy, we fix the average number of slabs per bucket and
   * instead manipulate the total number of buckets
   */
  
  using map_type = gpu_hash_table<Key, Value, SlabHashTypeT::ConcurrentMap>;

  uint32_t numKeys = 300'000'000;
  float numSlabsPerBucketAvg = 20 / float{10};
  uint32_t numKeysPerSlab = 15u;
  uint32_t numKeysPerBucketAvg = numSlabsPerBucketAvg * numKeysPerSlab;
  uint32_t numBuckets = 
    (numKeys + numKeysPerBucketAvg - 1) / numKeysPerBucketAvg;
  const uint32_t deviceIdx = 0;
  const int64_t seed = 1;

  
  // initialize key-value pairs for insertion
  std::mt19937 rng(/* seed = */ 12);
  Key *keys = new Key[numKeys];
  Value *values = new Value[numKeys];
  for(auto i = 0; i < numKeys; ++i) {
    keys[i] = int32_t(rng() & 0x7FFFFFFF);
    values[i] = ~keys[i];
  }

  auto buildTime = 0.0;

  for(auto _ : state) {
    state.PauseTiming();
    map_type map{numKeys, numBuckets, deviceIdx, seed};
    state.ResumeTiming();

    buildTime = map.hash_build(keys, values, numKeys);
    state.SetIterationTime((float)buildTime / 1000);
  }
}



/*
// cuCo tests
BENCHMARK_TEMPLATE(BM_cuco_insert_random_keys, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(cuSweepSize);

BENCHMARK_TEMPLATE(BM_cuco_search_all, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(cuSweepLoad);

BENCHMARK_TEMPLATE(BM_cuco_search_all, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(cuSweepSize);

// cuDF tests
BENCHMARK_TEMPLATE(BM_cudf_insert_random_keys, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(cuSweepSize);

BENCHMARK_TEMPLATE(BM_cudf_search_all, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(cuSweepLoad);

BENCHMARK_TEMPLATE(BM_cudf_search_all, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(cuSweepSize);
*/

BENCHMARK_TEMPLATE(BM_cudf_insert_resize, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond);

/*
// SlabHash tests
BENCHMARK_TEMPLATE(BM_slabhash_insert_random_keys, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(SlabSweepLoad);

BENCHMARK_TEMPLATE(BM_slabhash_search_all, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(SlabSweepLoad);
*/

BENCHMARK_TEMPLATE(BM_slabhash_insert_resize, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();