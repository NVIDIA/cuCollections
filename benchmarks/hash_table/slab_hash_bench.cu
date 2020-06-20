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



template <typename Key, typename Value>
static void cuco_search_all() {
  using map_type =
      cuco::insert_only_hash_array<Key, Value, cuda::thread_scope_device>;

  auto numKeys = 100'000'000;
  auto occupancy = 0.55;
  auto capacity = numKeys / occupancy;

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
  
  {    
    nvtx3::thread_range b{"cuco build"};
    thrust::for_each(
        thrust::device, zip_counter, zip_counter + numKeys,
        [view] __device__(auto const& p) mutable {
          view.insert(cuco::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
        });
  }

  {
    nvtx3::thread_range s{"cuco search"};
    thrust::for_each(
        thrust::device, key_iterator, key_iterator + numKeys,
        [view] __device__(auto const& p) mutable {
          view.find(p);
        });
  }
}



template <typename Key, typename Value>
static void cudf_search_all() {
  using map_type = concurrent_unordered_map<Key, Value>;
  
  auto numKeys = 100'000'000;
  auto occupancy = 0.55;
  auto capacity = numKeys / occupancy;

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
  
  { 
    nvtx3::thread_range b{"cudf build"};
    thrust::for_each(
        thrust::device, zip_counter, zip_counter + numKeys,
        [view] __device__(auto const& p) mutable {
          view.insert(thrust::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
        });
  }
  
  {
    nvtx3::thread_range s{"cudf search"};
    thrust::for_each(
      thrust::device, key_iterator, key_iterator + numKeys,
      [view] __device__(auto const& p) mutable {
        view.find(p);
      });
  }
}



template <typename Key, typename Value>
static void slabhash_search_all() {
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
  Value *results = new Value[numKeys];

  map_type map{numKeys, numBuckets, deviceIdx, seed};
  
  map.hash_build(keys, values, numKeys);
  map.hash_search(keys, results, numKeys);
  
  for(auto i = 0; i < numKeys; ++i) {
    if(keys[i] != ~results[i]) {
      std::cout << "Key-value mismatch at index " << i << std::endl;
      break;
    }
  }
  
}



template <typename Key, typename Value>
static void slabhash_insert_resize() {
  /* 
   * To adjust the occupancy, we fix the average number of slabs per bucket and
   * instead manipulate the total number of buckets
   */
  
  using map_type = gpu_hash_table<Key, Value, SlabHashTypeT::ConcurrentMap>;

  uint32_t numKeys = 170'000'000; // this number of keys ensures resizing once
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
  
  map_type map{numKeys, numBuckets, deviceIdx, seed};
  map.hash_build_with_unique_keys(keys, values, numKeys);

  delete keys;
  delete values;
}



template <typename Key, typename Value>
static void cudf_chain_search_all() {
  using map_type = concurrent_unordered_map_chain<Key, Value>;

  auto occupancy = 0.55;
  auto numKeys = 2 * 10'000'000;
  auto capacity = (numKeys / 2) / occupancy;
  
  auto map = map_type::create(capacity);
  auto view = *map;
  
  std::mt19937 rng(/* seed = */ 12);
  std::vector<Key> h_keys(numKeys);
  std::vector<Value> h_values(numKeys);
  for(auto i = 0; i < numKeys; ++i) {
    h_keys[i] = int32_t(rng() & 0x7FFFFFFF);
    h_values[i] = ~h_keys[i];
  }

  
  // insert host keys
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
  
  thrust::for_each(
    thrust::device, key_counter, key_counter + numKeys,
    [view, d_results] __device__(auto const& p) mutable {
      auto found = view.find(thrust::get<1>(p));
      d_results[thrust::get<0>(p)] = found->second;
    });
  

  // check results from search
  cudaMemcpy(h_results, d_results, numKeys * sizeof(Value*), cudaMemcpyDeviceToHost);

  for(auto i = 0; i < numKeys; ++i) {
    if(h_results[i] != h_values[i]) {
      std::cout << "Value-result mismatch at index " << i  << std::endl;
      break;
    }
  }
  
  // cleanup 
  view.freeSubmaps();
  cudaFree(d_results);
  free(h_results);
}



int main() {
  
  for(auto i = 0; i < 1; ++i) {
    //cudf_search_all<int32_t, int32_t>();
    //cuco_search_all<int32_t, int32_t>();
    //slabhash_search_all<int32_t, int32_t>();
    //slabhash_insert_resize<int32_t, int32_t>();
    cudf_chain_search_all<int32_t, int32_t>();
  }

  return 0;
}