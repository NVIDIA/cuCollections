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

//#include <cuco/insert_only_hash_array.cuh>
#include "cuco/static_map.cuh"
#include "../nvtx3.hpp"
#include "cudf/concurrent_unordered_map.cuh"
#include "cudf_chain/concurrent_unordered_map_chain.cuh"
#include "SlabHash/src/gpu_hash_table.cuh"

#include <thrust/for_each.h>
#include <iostream>
#include <fstream>



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


static void OccupancySweep(benchmark::internal::Benchmark *b) {
  for(auto occ = 40; occ <= 90; occ += 10) {
    b->Args({occ});
  }
}



/**
 * @brief Benchmark inserting all unique keys of a given number with specified
 * hash table occupancy
 */
#if 0
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
#endif


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
  
  std::vector<Key> h_keys(numKeys);
  std::vector<Value> h_values(numKeys);
  
  ///*
  std::mt19937 rng(12);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = int32_t(rng() & 0x7FFFFFFF);
    h_values[i] = ~h_keys[i];
  }
  //*/

  /*
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{1e9, 1e7};

  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = (int32_t)(std::round(d(gen))) & 0x7FFFFFFF;
    h_values[i] = ~h_keys[i];
  }
  //*/
  
  // geometrically distributed keys
  /*
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::geometric_distribution<uint32_t> d(0.01);
  
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = (int32_t)(std::round(d(gen))) & 0x7FFFFFFF;
    h_values[i] = ~h_keys[i];
  }
  //*/


  thrust::device_vector<Key> d_keys(numKeys);
  thrust::device_vector<Key> d_values(numKeys);
  thrust::copy(h_keys.begin(), h_keys.end(), d_keys.begin());
  thrust::copy(h_values.begin(), h_values.end(), d_values.begin());

  auto key_iterator = d_keys.begin();
  auto value_iterator = d_values.begin();
  auto zip_counter = thrust::make_zip_iterator(
      thrust::make_tuple(key_iterator, value_iterator));
  
  auto batchSize = 1e6;
  for (auto _ : state) {
    auto buildTime = 0;
    auto size = initSize;
    auto map = map_type::create(size);
    auto numKeysInserted = 0;
    
    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
    for(auto i = 0; i < numKeys / batchSize; ++i) {
      auto capacity = resizeOccupancy * size;
      if(numKeysInserted + batchSize <= capacity) {
        auto view = *map;
        thrust::for_each(
          thrust::device, zip_counter + i * batchSize, zip_counter + (i + 1) * batchSize,
          [view] __device__(auto const& p) mutable {
            view.insert(thrust::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
          });
        cudaDeviceSynchronize();
        numKeysInserted += batchSize;
      }
      else { // if the map needs to be resized, resize and reinsert all of the old keys
        size *= 2;
        map = map_type::create(size);
        auto view = *map;
        thrust::for_each(
          thrust::device, zip_counter, zip_counter + i * batchSize,
          [view] __device__(auto const& p) mutable {
            view.insert(thrust::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
          });
        --i;
      }
    }
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1).count();
    
    // correctness check
    /*
    Value* h_results = (Value*) malloc(numKeys * sizeof(Value)); 
    Value* d_results;
    cudaMalloc((void**)&d_results, numKeys * sizeof(Value));

    auto idxIterator = thrust::make_counting_iterator<Key>(0);
    auto key_counter = thrust::make_zip_iterator(thrust::make_tuple(idxIterator, key_iterator));
    auto view = *map;
    
    thrust::for_each(
      thrust::device, key_counter, key_counter + numKeys,
      [view, d_results] __device__(auto const& p) mutable {
        auto found = view.find(thrust::get<1>(p));
        if(found != view.end()) {
          d_results[thrust::get<0>(p)] = found->second;
        }
      });
    cudaMemcpy(h_results, d_results, numKeys * sizeof(Value), cudaMemcpyDeviceToHost);
    for(auto i = 0; i < numKeys; ++i) {
      if(h_results[i] != h_values[i]) {
        std::cout << "key value mismatch at index " << i << std::endl;
        break;
      }
    }

    free(h_results);
    cudaFree(d_results);
    */

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
  
  std::vector<Key> h_keys(numKeys);
  std::vector<Value> h_values(numKeys);
  
  ///*
  std::mt19937 rng(12);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = int32_t(rng() & 0x7FFFFFFF);
    h_values[i] = ~h_keys[i];
  }
  //*/
  
  /*
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{1e9, 1e7};

  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = (int32_t)(std::round(d(gen))) & 0x7FFFFFFF;
    h_values[i] = ~h_keys[i];
  }
  //*/
  
  // geometrically distributed keys
  /*
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::geometric_distribution<uint32_t> d(0.01);
  
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = (int32_t)(std::round(d(gen))) & 0x7FFFFFFF;
    h_values[i] = ~h_keys[i];
  }
  //*/

  thrust::device_vector<Key> d_keys(numKeys);
  thrust::device_vector<Key> d_values(numKeys);
  thrust::copy(h_keys.begin(), h_keys.end(), d_keys.begin());
  thrust::copy(h_values.begin(), h_values.end(), d_values.begin());

  auto key_iterator = d_keys.begin();
  auto value_iterator = d_values.begin();
  auto zip_counter = thrust::make_zip_iterator(
      thrust::make_tuple(key_iterator, value_iterator));
    
  auto batchSize = 1e6;
  auto size = initSize;
  auto map = map_type::create(size);
  auto numKeysInserted = 0;
  
  // insert keys
  for(auto i = 0; i < numKeys / batchSize; ++i) {
    auto capacity = resizeOccupancy * size;
    if(numKeysInserted + batchSize <= capacity) {
      auto view = *map;
      thrust::for_each(
        thrust::device, zip_counter + i * batchSize, zip_counter + (i + 1) * batchSize,
        [view] __device__(auto const& p) mutable {
          view.insert(thrust::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
        });
      cudaDeviceSynchronize();
      numKeysInserted += batchSize;
    }
    else { // if the map needs to be resized, resize and reinsert all of the old keys
      size *= 2;
      map = map_type::create(size);
      auto view = *map;
      thrust::for_each(
        thrust::device, zip_counter, zip_counter + i * batchSize,
        [view] __device__(auto const& p) mutable {
          view.insert(thrust::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
        });
      --i;
    }
  }
  
  // search for inserted keys
  auto view = *map;
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

    buildTime += map.hash_build_with_unique_keys(keys, values, numKeys);

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
  map.hash_build_with_unique_keys(keys, values, numKeys);
  
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
  uint32_t numBuckets = 7'340'032;
    /*(numKeys + numKeysPerBucketAvg - 1) / numKeysPerBucketAvg; */
  const uint32_t deviceIdx = 0;
  const int64_t seed = 1;

  
  // initialize key-value pairs for insertion
  Key *h_keys = new Key[numKeys];
  Value *h_values = new Value[numKeys];
  
  ///*
  std::mt19937 rng(12);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = int32_t(rng() & 0x7FFFFFFF);
    h_values[i] = ~h_keys[i];
  }
  //*/
  
  /*
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{1e9, 1e7};

  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = (int32_t)(std::round(d(gen))) & 0x7FFFFFFF;
    h_values[i] = ~h_keys[i];
  }
  //*/
  
  // geometrically distributed keys
  /*
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::geometric_distribution<uint32_t> d(0.01);
  
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = (int32_t)(std::round(d(gen))) & 0x7FFFFFFF;
    h_values[i] = ~h_keys[i];
  }
  //*/
  
  uint32_t batchSize = 1e6;
  for(auto _ : state) {
    auto buildTime = 0.0f;
    map_type map{batchSize, numBuckets, deviceIdx, seed};
    for(uint32_t i = 0; i < numKeys / batchSize; ++i) {
      buildTime += map.hash_build_with_unique_keys(h_keys + i * batchSize, h_values + i * batchSize, batchSize);
    }
    state.SetIterationTime((float)buildTime / 1000);
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));

  delete h_keys;
  delete h_values;
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
  Key *h_keys = new Key[numKeys];
  Value *h_values = new Value[numKeys];
  
  ///*
  std::mt19937 rng(12);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = int32_t(rng() & 0x7FFFFFFF);
    h_values[i] = ~h_keys[i];
  }
  //*/
  
  /*
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{1e9, 1e7};

  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = (int32_t)(std::round(d(gen))) & 0x7FFFFFFF;
    h_values[i] = ~h_keys[i];
  }
  //*/
 
  // geometrically distributed keys
  /*
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::geometric_distribution<uint32_t> d(0.01);
  
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = (int32_t)(std::round(d(gen))) & 0x7FFFFFFF;
    h_values[i] = ~h_keys[i];
  }
  //*/

  Value *results = new Value[numKeys];

  uint32_t batchSize = 1e6;
  map_type map{numKeys, numBuckets, deviceIdx, seed};
  for(uint32_t i = 0; i < numKeys / batchSize; ++i) {
    map.hash_build_with_unique_keys(h_keys + i * batchSize, h_values + i * batchSize, batchSize);
  }
  
  auto searchTime = 0.0;
  for(auto _ : state) {
    searchTime = map.hash_search(h_keys, results, numKeys);
    state.SetIterationTime((float)searchTime / 1000);
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));

  delete h_keys;
  delete h_values;
  delete results;
}



template <typename Key, typename Value>
static void BM_cudfChain_insert_resize(::benchmark::State& state) {
  using map_type = concurrent_unordered_map_chain<Key, Value>;

  auto numKeys = state.range(0);

  std::vector<Key> h_keys(numKeys);
  std::vector<Value> h_values(numKeys);

  // uniformly random keys
  ///*
  std::mt19937 rng(12);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = int32_t(rng() & 0x7FFFFFFF);
    h_values[i] = ~h_keys[i];
  }
  //*/
  
  // normally distributed keys
  /*
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{1e9, 1e7};

  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = (int32_t)(std::round(d(gen))) & 0x7FFFFFFF;
    h_values[i] = ~h_keys[i];
  }
  //*/

  // geometrically distributed keys
  /*
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::geometric_distribution<uint32_t> d(0.01);
  
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = (int32_t)(std::round(d(gen))) & 0x7FFFFFFF;
    h_values[i] = ~h_keys[i];
  }
  //*/

  auto batchSize = 1e6;
  for (auto _ : state) {
    auto buildTime = 0.0f;
    auto map = map_type::create(134'217'728);
    auto view = *map;

    for(auto i = 0; i < numKeys / batchSize; ++i) {
      std::vector<Key> keyBatch (h_keys.begin() + i * batchSize, h_keys.begin() + (i + 1) * batchSize);
      std::vector<Value> valueBatch (h_values.begin() + i * batchSize, h_values.begin() + (i + 1) * batchSize);
      buildTime += view.bulkInsert(keyBatch, valueBatch, batchSize);
    }
    state.SetIterationTime(buildTime / 1000);

    view.freeSubmaps();
  }
 
  uint64_t nBytes = (sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0));
  state.SetBytesProcessed(nBytes);
}



template <typename Key, typename Value>
static void BM_cudfChain_search_resize(::benchmark::State& state) {
  using map_type = concurrent_unordered_map_chain<Key, Value>;

  auto numKeys = state.range(0);
  
  auto map = map_type::create(1<<28);
  auto view = *map;
  
  std::vector<Key> h_keys(numKeys);
  std::vector<Value> h_values(numKeys);
  ///*
  std::mt19937 rng(12);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = int32_t(rng() & 0x7FFFFFFF);
    h_values[i] = ~h_keys[i];

  }
  //*/

  /*
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{1e9, 1e7};

  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = (int32_t)(std::round(d(gen))) & 0x7FFFFFFF;
    h_values[i] = ~h_keys[i];
  }
  //*/
  
  // geometrically distributed keys
  /*
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::geometric_distribution<uint32_t> d(0.01);
  
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = (int32_t)(std::round(d(gen))) & 0x7FFFFFFF;
    h_values[i] = ~h_keys[i];
  }
  //*/

  // insert keys
  auto batchSize = 1e6;
  for(auto i = 0; i < numKeys / batchSize; ++i) {
    std::vector<Key> keyBatch (h_keys.begin() + i * batchSize, h_keys.begin() + (i + 1) * batchSize);
    std::vector<Value> valueBatch (h_values.begin() + i * batchSize, h_values.begin() + (i + 1) * batchSize);
    view.bulkInsert(keyBatch, valueBatch, batchSize);
  }

  // search for keys
  std::vector<Value> h_results(numKeys);
  float searchTime = 0.0f;

  for(auto _ : state) {
    searchTime = view.bulkSearch(h_keys, h_results, numKeys);
    state.SetIterationTime(searchTime / 1000);
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
  // cleanup 
  view.freeSubmaps();
}



template <typename Key, typename Value>
static void BM_cudfChain_search_none(::benchmark::State& state) {
  using map_type = concurrent_unordered_map_chain<Key, Value>;

  uint32_t numKeys = 100'000'000;
  auto numQueries = 100'000'000;
  auto occupancy = (float)state.range(0) / 100;
  uint32_t capacity = (float)numKeys / occupancy;
  
  auto map = map_type::create(capacity);
  auto view = *map;
  
  
  std::vector<Key> h_keys(numKeys);
  std::vector<Value> h_values(numKeys);
  ///*
  std::mt19937 rng(12);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = i;
    h_values[i] = i;

  }

  std::vector<Key> h_search_keys(numQueries);
  for(auto i = 0; i < numQueries; ++i) {
    h_search_keys[i] = numKeys + i;
  }
  
  // insert keys
  auto batchSize = 1e6;
  for(auto i = 0; i < numKeys / batchSize; ++i) {
    std::vector<Key> keyBatch (h_keys.begin() + i * batchSize, h_keys.begin() + (i + 1) * batchSize);
    std::vector<Value> valueBatch (h_values.begin() + i * batchSize, h_values.begin() + (i + 1) * batchSize);
    view.bulkInsert(keyBatch, valueBatch, batchSize);
  }

  // search for keys
  std::vector<Value> h_results(numQueries);
  float searchTime = 0.0f;

  for(auto _ : state) {
    searchTime = view.bulkSearch(h_search_keys, h_results, numQueries);
    state.SetIterationTime(searchTime / 1000);
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(numQueries));
  // cleanup 
  view.freeSubmaps();
}



template <typename Key, typename Value>
static void BM_cudfChain_search_all(::benchmark::State& state) {
  using map_type = concurrent_unordered_map_chain<Key, Value>;

  uint32_t numKeys = 100'000'000;
  auto numQueries = 100'000'000;
  auto occupancy = (float)state.range(0) / 100;
  uint32_t capacity = (float)numKeys / occupancy;
  
  auto map = map_type::create(capacity);
  auto view = *map;
  
  std::vector<Key> h_keys(numKeys);
  std::vector<Value> h_values(numKeys);
  ///*
  std::mt19937 rng(12);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = i;
    h_values[i] = i;

  }

  std::vector<Key> h_search_keys(numQueries);
  for(auto i = 0; i < numQueries; ++i) {
    h_search_keys[i] = i;
  }
  
  // insert keys
  auto batchSize = 1e6;
  for(auto i = 0; i < numKeys / batchSize; ++i) {
    std::vector<Key> keyBatch (h_keys.begin() + i * batchSize, h_keys.begin() + (i + 1) * batchSize);
    std::vector<Value> valueBatch (h_values.begin() + i * batchSize, h_values.begin() + (i + 1) * batchSize);
    view.bulkInsert(keyBatch, valueBatch, batchSize);
  }

  // search for keys
  std::vector<Value> h_results(numQueries);
  float searchTime = 0.0f;

  for(auto _ : state) {
    searchTime = view.bulkSearch(h_search_keys, h_results, numQueries);
    state.SetIterationTime(searchTime / 1000);
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(numQueries));
  // cleanup 
  view.freeSubmaps();
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
BENCHMARK_TEMPLATE(BM_cudfChain_insert_resize, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Apply(ResizeSweep);

BENCHMARK_TEMPLATE(BM_cudfChain_search_resize, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Apply(ResizeSweep);

BENCHMARK_TEMPLATE(BM_cudfChain_search_none, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Apply(OccupancySweep);

BENCHMARK_TEMPLATE(BM_cudfChain_search_all, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Apply(OccupancySweep);
*/
