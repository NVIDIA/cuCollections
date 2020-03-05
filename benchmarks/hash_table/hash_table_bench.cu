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
#include "../synchronization/synchronization.hpp"
#include "cudf/concurrent_unordered_map.cuh"

#include <thrust/for_each.h>
#include <iostream>

static void BM_cudf_construction(::benchmark::State& state) {
  using map_type = concurrent_unordered_map<int32_t, int32_t>;

  for (auto _ : state) {
    cuda_event_timer t{state, true};
    auto map = map_type::create(state.range(0));
  }
}
BENCHMARK(BM_cudf_construction)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(10)
    ->Range(10'000, 100'000'000);

static void BM_cuco_construction(::benchmark::State& state) {
  using map_type =
      cuco::insert_only_hash_array<int32_t, int32_t, cuda::thread_scope_device>;
  for (auto _ : state) {
    cuda_event_timer t{state, true};
    map_type map{state.range(0), -1};
  }
}
BENCHMARK(BM_cuco_construction)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(10)
    ->Range(10'000, 100'000'000);

static void BM_cuco_insert_unique_keys(::benchmark::State& state) {
  using map_type =
      cuco::insert_only_hash_array<int32_t, int32_t, cuda::thread_scope_device>;

  auto fill_factor = 0.5;

  for (auto _ : state) {
    map_type map{state.range(0) / fill_factor, -1};
    auto view = map.get_device_view();
    auto counting_iterator = thrust::make_counting_iterator(0);
    auto zip_counter = thrust::make_zip_iterator(
        thrust::make_tuple(counting_iterator, counting_iterator));

    {
      // Only time the kernel
      cuda_event_timer t{state, true};
      thrust::for_each(
          thrust::device, zip_counter, zip_counter + state.range(0),
          [view] __device__(auto const& p) mutable {
            view.insert(cuco::make_pair(thrust::get<0>(p), thrust::get<1>(p)));
          });
    }
  }
}
BENCHMARK(BM_cuco_insert_unique_keys)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(10)
    ->Range(10'000, 100'000'000);

static void BM_cudf_insert_unique_keys(::benchmark::State& state) {
  using map_type = concurrent_unordered_map<int32_t, int32_t>;

  auto fill_factor = 0.5;

  for (auto _ : state) {
    auto map = map_type::create(state.range(0) / fill_factor);
    auto view = *map;
    auto counting_iterator = thrust::make_counting_iterator(0);
    auto zip_counter = thrust::make_zip_iterator(
        thrust::make_tuple(counting_iterator, counting_iterator));

    {
      // Only time the kernel
      cuda_event_timer t{state, true};
      thrust::for_each(thrust::device, zip_counter,
                       zip_counter + state.range(0),
                       [view] __device__(auto const& p) mutable {
                         view.insert(thrust::make_pair(thrust::get<0>(p),
                                                       thrust::get<1>(p)));
                       });
    }
  }
}
BENCHMARK(BM_cudf_insert_unique_keys)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(10)
    ->Range(10'000, 100'000'000);