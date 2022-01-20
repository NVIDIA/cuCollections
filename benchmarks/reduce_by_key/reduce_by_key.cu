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

#include <thrust/device_vector.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

/**
 * @brief Generates input sizes and number of unique keys
 *
 */
static void generate_size_and_num_unique(benchmark::internal::Benchmark* b)
{
  for (auto num_unique = 64; num_unique <= 1 << 20; num_unique <<= 1) {
    for (auto size = 10'000'000; size <= 10'000'000; size *= 10) {
      b->Args({size, num_unique});
    }
  }
}

template <typename KeyRandomIterator, typename ValueRandomIterator>
void thrust_reduce_by_key(KeyRandomIterator keys_begin,
                          KeyRandomIterator keys_end,
                          ValueRandomIterator values_begin)
{
  using Key   = typename thrust::iterator_traits<KeyRandomIterator>::value_type;
  using Value = typename thrust::iterator_traits<ValueRandomIterator>::value_type;

  // Exact size of output is unknown (number of unique keys), but upper bounded
  // by the number of keys
  auto maximum_output_size = thrust::distance(keys_begin, keys_end);
  thrust::device_vector<Key> output_keys(maximum_output_size);
  thrust::device_vector<Value> output_values(maximum_output_size);

  thrust::sort_by_key(thrust::device, keys_begin, keys_end, values_begin);
  thrust::reduce_by_key(
    thrust::device, keys_begin, keys_end, values_begin, output_keys.begin(), output_values.end());
}

template <typename Key, typename Value>
static void BM_thrust(::benchmark::State& state)
{
  auto const num_unique_keys = state.range(1);
  for (auto _ : state) {
    state.PauseTiming();
    thrust::device_vector<Key> keys(state.range(0));
    auto begin = thrust::make_counting_iterator(0);
    thrust::transform(
      begin, begin + state.range(0), keys.begin(), [num_unique_keys] __device__(auto i) {
        return i % num_unique_keys;
      });

    thrust::device_vector<Value> values(state.range(0));
    state.ResumeTiming();
    thrust_reduce_by_key(keys.begin(), keys.end(), values.begin());
    cudaDeviceSynchronize();
  }
}
BENCHMARK_TEMPLATE(BM_thrust, int32_t, int32_t)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_num_unique);

BENCHMARK_TEMPLATE(BM_thrust, int64_t, int64_t)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_num_unique);

// TODO: Hash based reduce by key benchmark
