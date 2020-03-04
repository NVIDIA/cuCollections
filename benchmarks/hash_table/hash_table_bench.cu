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
#include "cudf/concurrent_unordered_map.cuh"
#include "../synchronization/synchronization.hpp"

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
  using map_type = insert_only_hash_array<int32_t, int32_t>;
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

/*
// Define another benchmark
static void BM_StringCopy(benchmark::State& state) {
  std::string x = "hello";
  for (auto _ : state) std::string copy(x);
}
BENCHMARK(BM_StringCopy);

static void BM_StringCompare(benchmark::State& state) {
  std::string s1(state.range(0), '-');
  std::string s2(state.range(0), '-');
  for (auto _ : state) {
    benchmark::DoNotOptimize(s1.compare(s2));
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_StringCompare)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 18)
    ->Complexity(benchmark::oN);

template <class Q>
void BM_Sequential(benchmark::State& state) {
  Q q;
  typename Q::value_type v(0);
  for (auto _ : state) {
    for (int i = state.range(0); i--;) q.push_back(v);
  }
  // actually messages, not bytes:
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          state.range(0));
}
BENCHMARK_TEMPLATE(BM_Sequential, std::vector<int>)->Range(1 << 0, 1 << 10);
*/