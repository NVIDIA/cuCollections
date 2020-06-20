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
#include "cudf_chain/concurrent_unordered_map_chain.cuh"
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
  for(auto size = 10'000'000; size <= 310'000'000; size += 20'000'000) {
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

  auto numKeys = state.range(0);
  auto resizeOccupancy = 0.6;
  auto initSize = 134'217'728;
  
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
    
    auto size = 134'217'728;
    auto numToInsert =  std::min((uint32_t)(resizeOccupancy * size), (uint32_t) numKeys);
    auto finished = false;
    
    auto t1 = 0;
    auto t2 = 0;
    auto buildTime = 0;

    do {
      std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
      auto map = map_type::create(size);
      auto view = *map;
      std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
      
      if(size == initSize) {
        t1 = 0;
      }
      else { // only time the creation time after the map is resized
        t1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1).count();
      }
      
      std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
      thrust::for_each(
          thrust::device, zip_counter, zip_counter + numToInsert,
          [view] __device__(auto const& p) mutable {
            view.insert(thrust::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
          });
     
      finished = (numToInsert == numKeys);
      size *= 2;
      numToInsert = std::min((uint32_t) (resizeOccupancy * size), (uint32_t) numKeys);
      std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
      
      t2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin2).count();

      buildTime += (t1 + t2);

    } while(!finished);



    state.SetIterationTime((float) buildTime / 1000);
  }
  
  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}



template <typename Key, typename Value>
static void BM_cudf_insert_resize_search(::benchmark::State& state) {
  using map_type = concurrent_unordered_map<Key, Value>;

  auto numKeys = state.range(0);
  auto resizeOccupancy = 0.6;
  auto initSize = 134'217'728;
  
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
    
  auto size = 134'217'728;
  auto numToInsert =  std::min((uint32_t)(resizeOccupancy * size), (uint32_t) numKeys);
  auto finished = false;
  
  auto t1 = 0;
  auto t2 = 0;
  auto buildTime = 0;
  
  // dummy initializers so that variables are accessible outside of do-while
  auto map = map_type::create(1);
  auto view = *map;

  do {
    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
    map = map_type::create(size);
    view = *map;
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    
    if(size == initSize) {
      t1 = 0;
    }
    else { // only time the creation after the map is resized
      t1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1).count();
    }
    
    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    thrust::for_each(
        thrust::device, zip_counter, zip_counter + numToInsert,
        [view] __device__(auto const& p) mutable {
          view.insert(thrust::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
        });
    
    finished = (numToInsert == numKeys);
    size *= 2;
    numToInsert = std::min((uint32_t) (resizeOccupancy * size), (uint32_t) numKeys);
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    
    t2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin2).count();

    buildTime += (t1 + t2);

  } while(!finished);
  
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
  
  delete keys;
  delete values;
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

  map_type map{numKeys, numBuckets, deviceIdx, seed};
  map.hash_build(keys, values, numKeys);
  
  for(auto _ : state) {
    searchTime += map.hash_search(keys, results, numKeys);
    avgLoadFactor += map.measureLoadFactor();
  }
  
  searchTime /= state.iterations();
  avgLoadFactor /= state.iterations();

  state.counters["searchTime"] = searchTime;
  state.counters["loadFactor"] = avgLoadFactor;
  state.counters["GBytesPerSecond"] = (sizeof(Key) + sizeof(Value)) *
                                     int64_t(state.range(0)) / (1'000'000 * searchTime);
  
  delete keys;
  delete values;
  delete results;
}



template <typename Key, typename Value>
static void BM_slabhash_insert_resize(::benchmark::State& state) {
  /* 
   * To adjust the occupancy, we fix the average number of slabs per bucket and
   * instead manipulate the total number of buckets
   */
  
  using map_type = gpu_hash_table<Key, Value, SlabHashTypeT::ConcurrentMap>;

  uint32_t numKeys = state.range(0);
  /*
  float numSlabsPerBucketAvg = 6 / float{10};
  uint32_t numKeysPerSlab = 15u;
  uint32_t numKeysPerBucketAvg = numSlabsPerBucketAvg * numKeysPerSlab;
  */
  uint32_t numBuckets = 4'194'304;
    /*(numKeys + numKeysPerBucketAvg - 1) / numKeysPerBucketAvg; */
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
    map_type map{numKeys, numBuckets, deviceIdx, seed};
    buildTime = map.hash_build_with_unique_keys(keys, values, numKeys);
    state.SetIterationTime((float)buildTime / 1000);
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));

  delete keys;
  delete values;
}



template <typename Key, typename Value>
static void BM_slabhash_insert_resize_search(::benchmark::State& state) {
  /* 
   * To adjust the occupancy, we fix the average number of slabs per bucket and
   * instead manipulate the total number of buckets
   */
  
  using map_type = gpu_hash_table<Key, Value, SlabHashTypeT::ConcurrentMap>;

  uint32_t numKeys = state.range(0);
  /*
  float numSlabsPerBucketAvg = 6 / float{10};
  uint32_t numKeysPerSlab = 15u;
  uint32_t numKeysPerBucketAvg = numSlabsPerBucketAvg * numKeysPerSlab;
  */
  uint32_t numBuckets = 4'194'304;
    /*(numKeys + numKeysPerBucketAvg - 1) / numKeysPerBucketAvg; */
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
  
  map_type map{numKeys, numBuckets, deviceIdx, seed};
  map.hash_build_with_unique_keys(keys, values, numKeys);
  
  auto searchTime = 0.0;
  for(auto _ : state) {
    searchTime = map.hash_search(keys, results, numKeys);
    state.SetIterationTime((float)searchTime / 1000);
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));

  delete keys;
  delete values;
  delete results;
}



template <typename Key, typename Value>
static void BM_cudf_chain_insert_resize(::benchmark::State& state) {
  using map_type = concurrent_unordered_map_chain<Key, Value>;

  auto numKeys = state.range(0);
  
  std::mt19937 rng(/* seed = */ 12);
  std::vector<Key> h_keys(numKeys);
  std::vector<Value> h_values(numKeys);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = int32_t(rng() & 0x7FFFFFFF);
    h_values[i] = ~h_keys[i];
  }
 
  float buildTime = 0.0f;
  for (auto _ : state) {
    auto map = map_type::create(134'217'728);
    auto view = *map;

    buildTime = view.bulkInsert(h_keys, h_values, numKeys);
    state.SetIterationTime(buildTime / 1000);

    view.freeSubmaps();
  }
  
  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
                          
}



template <typename Key, typename Value>
static void BM_cudf_chain_search_resize(::benchmark::State& state) {
  using map_type = concurrent_unordered_map_chain<Key, Value>;

  auto numKeys = state.range(0);
  
  auto map = map_type::create(134'217'728);
  auto view = *map;
  
  std::mt19937 rng(/* seed = */ 12);
  std::vector<Key> h_keys(numKeys);
  std::vector<Value> h_values(numKeys);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = int32_t(rng() & 0x7FFFFFFF);
    h_values[i] = ~h_keys[i];

  }

  // insert keys
  view.bulkInsert(h_keys, h_values, numKeys);
  
  // search for keys
  Value* d_results;
  cudaMalloc((void**)&d_results, numKeys * sizeof(Value*));
  Value* h_results = (Value*) malloc(numKeys * sizeof(Value*));
  
  thrust::device_vector<Key> d_keys( h_keys );
  auto idx_iterator = thrust::make_counting_iterator<uint32_t>(0);
  auto key_iterator = d_keys.begin();
  auto key_counter = thrust::make_zip_iterator(
      thrust::make_tuple(idx_iterator, key_iterator));
  
  for(auto _ : state) {
    thrust::for_each(
      thrust::device, key_iterator, key_iterator + numKeys,
      [view] __device__(auto const& p) mutable {
        view.find(p);
        //d_results[thrust::get<0>(p)] = found->second;
      });
  }
  
  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
  // cleanup 
  view.freeSubmaps();
  cudaFree(d_results);
  free(h_results);
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

BENCHMARK_TEMPLATE(BM_cudf_insert_resize, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Apply(ResizeSweep);
BENCHMARK_TEMPLATE(BM_cudf_insert_resize_search, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(ResizeSweep);
*/
/*
// SlabHash tests
BENCHMARK_TEMPLATE(BM_slabhash_insert_random_keys, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(SlabSweepLoad);


BENCHMARK_TEMPLATE(BM_slabhash_search_all, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(SlabSweepLoad);

BENCHMARK_TEMPLATE(BM_slabhash_search_all, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(SlabSweepSize);

BENCHMARK_TEMPLATE(BM_slabhash_insert_resize, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Apply(ResizeSweep);

BENCHMARK_TEMPLATE(BM_slabhash_insert_resize_search, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Apply(ResizeSweep);
*/

// chaining cuDF tests
/*
BENCHMARK_TEMPLATE(BM_cudf_chain_insert_resize, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Apply(ResizeSweep);
*/
BENCHMARK_TEMPLATE(BM_cudf_chain_search_resize, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(ResizeSweep);
