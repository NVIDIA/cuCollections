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
 * @brief Benchmarks the time it takes to construct a cuDF hash map of a given
 * size.
 */
template <typename Key, typename Value>
static void BM_cudf_construction(::benchmark::State& state) {
  using map_type = concurrent_unordered_map<Key, Value>;

  std::string msg{"cudf construction: " + std::to_string(state.range(0))};
  nvtx3::thread_range r{"msg"};

  for (auto _ : state) {
    nvtx3::thread_range l;
    auto map = map_type::create(state.range(0));
    cudaDeviceSynchronize();
  }

  state.SetBytesProcessed(
      ((sizeof(Key) + sizeof(Value)) * int64_t(state.iterations()) *
       int64_t(state.range(0))));
}

/*
BENCHMARK_TEMPLATE(BM_cudf_construction, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(10)
    ->Range(10'000, 1'000'000'000);

BENCHMARK_TEMPLATE(BM_cudf_construction, int64_t, int64_t)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(10)
    ->Range(10'000, 100'000'000);
*/

/**
 * @brief Benchmarks the time it takes to construct a cuco hash map of a given
 * size.
 */
template <typename Key, typename Value>
static void BM_cuco_construction(::benchmark::State& state) {
  using map_type =
      cuco::insert_only_hash_array<Key, Value, cuda::thread_scope_device>;
  nvtx3::thread_range r{"cuco construction"};
  for (auto _ : state) {
    nvtx3::thread_range l;
    map_type map{state.range(0), -1};
    cudaDeviceSynchronize();
  }

  state.SetBytesProcessed((sizeof(typename map_type::atomic_value_type) *
                           int64_t(state.iterations()) *
                           int64_t(state.range(0))));
}

/*
BENCHMARK_TEMPLATE(BM_cuco_construction, int32_t, int32_t)
    //->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(10)
    ->Range(10'000, 1'000'000'000);

BENCHMARK_TEMPLATE(BM_cuco_construction, int64_t, int64_t)
    //->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(10)
    ->Range(10'000, 100'000'000);
*/



/**
 * @brief Generates input sizes and number of buckets for SlabHash
 */
static void genNumKeysAndNumSlabs32(benchmark::internal::Benchmark *b) {
  for(auto i = 10'000; i < 1'000'000'000; i *= 10) {
    for(auto j = 1; j < 2; ++j) {
      b->Args({i, j});
    }    
  }
}

/**
 * @brief Benchmarks the time it takes to construct a SlabHash hash map of a
 * given size.
 */
template <typename Key, typename Value>
static void BM_slabhash_construction(::benchmark::State& state) {
  using map_type = gpu_hash_table<Key, Value, SlabHashTypeT::ConcurrentMap>;

  uint32_t numKeys = state.range(0);
  uint32_t numSlabsPerBucketAvg = state.range(1);
  uint32_t numKeysPerSlab = 15u;//ConcurrentMapT<Key, Value>::SlabTypeT.NUM_ELEMENTS_PER_SLAB;
  uint32_t numKeysPerBucketAvg = numSlabsPerBucketAvg * numKeysPerSlab;
  uint32_t numBuckets = (numKeys + numKeysPerBucketAvg - 1) / numKeysPerBucketAvg;
  const uint32_t deviceIdx = 0;
  const int64_t seed = 1;
  
  for(auto _ : state)  {
    map_type map{numKeys, numBuckets, deviceIdx, seed};
    cudaDeviceSynchronize();
  }
  
  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                           int64_t(state.iterations()) *
                           int64_t(numKeys));
}
/*
BENCHMARK_TEMPLATE(BM_slabhash_construction, int32_t, int32_t)
    //->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->Apply(genNumKeysAndNumSlabs32);
*/

/**
 * @brief Generates input sizes and hash table occupancies
 *
 */
static void generate_size_and_occupancy(benchmark::internal::Benchmark* b) {
  for (auto occupancy = 40; occupancy <= 90; occupancy += 10) {
    for (auto size = 100'000; size <= 100'000'000; size *= 10) {
      b->Args({size, occupancy});
    }
  }
}

/**
 * @brief Benchmark inserting all unique keys of a given number with specified
 * hash table occupancy
 */
template <typename Key, typename Value>
static void BM_cuco_insert_unique_keys(::benchmark::State& state) {
  using map_type =
      cuco::insert_only_hash_array<Key, Value, cuda::thread_scope_device>;

  auto occupancy = (state.range(1) / double{100});
  auto capacity = state.range(0) / occupancy;
  for (auto _ : state) {
    state.PauseTiming();
    map_type map{capacity, -1};
    auto view = map.get_device_view();
    auto key_iterator = thrust::make_counting_iterator<Key>(0);
    auto value_iterator = thrust::make_counting_iterator<Value>(0);
    auto zip_counter = thrust::make_zip_iterator(
        thrust::make_tuple(key_iterator, value_iterator));
    state.ResumeTiming();
    thrust::for_each(
        thrust::device, zip_counter, zip_counter + state.range(0),
        [view] __device__(auto const& p) mutable {
          view.insert(cuco::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
        });
  }
  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}


BENCHMARK_TEMPLATE(BM_cuco_insert_unique_keys, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(generate_size_and_occupancy);

/*
BENCHMARK_TEMPLATE(BM_cuco_insert_unique_keys, int64_t, int64_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(generate_size_and_occupancy);
*/

template <typename Key, typename Value>
static void BM_cudf_insert_unique_keys(::benchmark::State& state) {
  using map_type = concurrent_unordered_map<Key, Value>;

  auto occupancy = (state.range(1) / double{100});
  auto capacity = state.range(0) / occupancy;
  for (auto _ : state) {
    state.PauseTiming();
    auto map = map_type::create(capacity);
    auto view = *map;
    auto key_iterator = thrust::make_counting_iterator<Key>(0);
    auto value_iterator = thrust::make_counting_iterator<Value>(0);
    auto zip_counter = thrust::make_zip_iterator(
        thrust::make_tuple(key_iterator, value_iterator));

    state.ResumeTiming();
    thrust::for_each(
        thrust::device, zip_counter, zip_counter + state.range(0),
        [view] __device__(auto const& p) mutable {
          view.insert(thrust::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
        });
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

BENCHMARK_TEMPLATE(BM_cudf_insert_unique_keys, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(generate_size_and_occupancy);

/*
// native atomicCAS doesn't have overloads for (u)int64_t
BENCHMARK_TEMPLATE(BM_cudf_insert_unique_keys, unsigned long long int,
                   unsigned long long int)
    ->Unit(benchmark::kMillisecond)
    ->Apply(generate_size_and_occupancy);
*/

/**
 * @brief Generates input sizes and number of buckets for SlabHash
 */
static void genSizeSlabs(benchmark::internal::Benchmark *b) {
  for (auto size = 100'000; size <= 10'000'000; size *= 10) {
    for(auto deciSPBAvg = 1; deciSPBAvg < 20; ++deciSPBAvg) {
      b->Args({size, deciSPBAvg});
    }
  }
}

template <typename Key, typename Value>
static void BM_slabhash_insert_unique_keys(::benchmark::State& state) {
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
  Key *keys = new Key[numKeys];
  Value *values = new Value[numKeys];
  for(auto i = 0; i < numKeys; ++i) {
    keys[i] = i;
    values[i] = i;
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

BENCHMARK_TEMPLATE(BM_slabhash_insert_unique_keys, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(genSizeSlabs);