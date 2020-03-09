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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <cuco/insert_only_hash_array.cuh>
#include "../nvtx3.hpp"

template <typename KeyRandomIterator, typename ValueRandomIterator>
void thrust_reduce_by_key(KeyRandomIterator keys_begin,
                          KeyRandomIterator keys_end,
                          ValueRandomIterator values_begin) {
  nvtx3::thread_range l;
  using Key = typename thrust::iterator_traits<KeyRandomIterator>::value_type;
  using Value =
      typename thrust::iterator_traits<ValueRandomIterator>::value_type;

  // Exact size of output is unknown (number of unique keys), but upper bounded
  // by the number of keys
  auto maximum_output_size = thrust::distance(keys_begin, keys_end);
  thrust::device_vector<Key> output_keys(maximum_output_size);
  thrust::device_vector<Value> output_values(maximum_output_size);

  thrust::sort_by_key(thrust::device, keys_begin, keys_end, values_begin);
  thrust::reduce_by_key(thrust::device, keys_begin, keys_end, values_begin,
                        output_keys.begin(), output_values.end());
}

template <typename Key, typename Value>
static void BM_thrust(::benchmark::State& state) {
  std::string msg{"thrust rbk: "};
  msg += std::to_string(state.range(0));
  nvtx3::thread_range r{msg};
  for (auto _ : state) {
    state.PauseTiming();
    thrust::device_vector<Key> keys(state.range(0));
    thrust::device_vector<Value> values(state.range(0));
    state.ResumeTiming();
    thrust_reduce_by_key(keys.begin(), keys.end(), values.begin());
    cudaDeviceSynchronize();
  }
}
BENCHMARK_TEMPLATE(BM_thrust, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(10)
    ->Range(1'000'000, 1'000'000'000);

/*
BENCHMARK_TEMPLATE(BM_thrust, int64_t, int64_t)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(10)
    ->Range(100'000, 100'000'000);
    */

template <typename KeyRandomIterator, typename ValueRandomIterator>
void cuco_reduce_by_key(KeyRandomIterator keys_begin,
                        KeyRandomIterator keys_end,
                        ValueRandomIterator values_begin) {
  nvtx3::thread_range l{};
  using Key = typename thrust::iterator_traits<KeyRandomIterator>::value_type;
  using Value =
      typename thrust::iterator_traits<ValueRandomIterator>::value_type;

  auto const input_size = thrust::distance(keys_begin, keys_end);
  auto const occupancy = 0.9;
  std::size_t const map_capacity = input_size / occupancy;
  cuco::insert_only_hash_array<Key, Value, cuda::thread_scope_device> map{
      map_capacity, -1};
}

template <typename Key, typename Value>
static void BM_cuco(::benchmark::State& state) {
  std::string msg{"cuco rbk: "};
  msg += std::to_string(state.range(0));
  nvtx3::thread_range r{msg};
  for (auto _ : state) {
    state.PauseTiming();
    thrust::device_vector<Key> keys(state.range(0));
    thrust::device_vector<Value> values(state.range(0));
    state.ResumeTiming();
    cuco_reduce_by_key(keys.begin(), keys.end(), values.begin());
    cudaDeviceSynchronize();
  }
}
BENCHMARK_TEMPLATE(BM_cuco, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(10)
    ->Range(1'000'000, 1'000'000'000);
