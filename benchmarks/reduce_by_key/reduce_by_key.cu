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
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cuco/insert_only_hash_array.cuh>

#include "../hash_table/cudf/concurrent_unordered_map.cuh"

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
/*
BENCHMARK_TEMPLATE(BM_thrust, int32_t, int32_t)
    ->Unit(benchmark::kMillisecond)
    ->Apply(generate_size_and_num_unique);
    */

/*
BENCHMARK_TEMPLATE(BM_thrust, int64_t, int64_t)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(10)
    ->Range(100'000, 100'000'000);
    */
/*

template <typename KeyRandomIterator, typename ValueRandomIterator>
void cuco_reduce_by_key(KeyRandomIterator keys_begin,
                     KeyRandomIterator keys_end,
                     ValueRandomIterator values_begin) {
using Key = typename thrust::iterator_traits<KeyRandomIterator>::value_type;
using Value =
   typename thrust::iterator_traits<ValueRandomIterator>::value_type;

auto const input_size = thrust::distance(keys_begin, keys_end);
auto const occupancy = 0.9;
std::size_t const map_capacity = input_size / occupancy;

using map_type =
   cuco::insert_only_hash_array<Key, Value, cuda::thread_scope_device>;

map_type map{map_capacity, -1};
}

template <typename Key, typename Value>
static void BM_cuco(::benchmark::State& state) {
std::string msg{"cuco rbk: "};
msg += std::to_string(state.range(0));
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
 */

template <typename KeyRandomIterator, typename ValueRandomIterator>
void cudf_reduce_by_key(KeyRandomIterator keys_begin,
                        KeyRandomIterator keys_end,
                        ValueRandomIterator values_begin)
{
  using Key   = typename thrust::iterator_traits<KeyRandomIterator>::value_type;
  using Value = typename thrust::iterator_traits<ValueRandomIterator>::value_type;

  auto const input_size          = thrust::distance(keys_begin, keys_end);
  auto const occupancy           = 0.5;
  std::size_t const map_capacity = input_size / occupancy;

  using map_type = concurrent_unordered_map<Key, Value>;
  auto map       = map_type::create(map_capacity);

  // Concurrent insert/find are supported
  if (sizeof(Key) + sizeof(Value) <= 8) {
    thrust::transform(thrust::device,
                      keys_begin,
                      keys_end,
                      values_begin,
                      thrust::make_discard_iterator(),
                      [view = *map] __device__(Key const& k, Value const& v) mutable {
                        auto found = view.find(k);

                        if (view.end() == found) {
                          auto result = view.insert(thrust::make_pair(k, 0));
                          found       = result.first;
                        }

                        atomicAdd(&(found->second), v);
                        return 0;
                      });
  } else {
    thrust::transform(thrust::device,
                      keys_begin,
                      keys_end,
                      values_begin,
                      thrust::make_discard_iterator(),
                      [view = *map] __device__(Key const& k, Value const& v) mutable {
                        auto result                          = view.insert(thrust::make_pair(k, 0));
                        auto found                           = result.first;
                        thrust::pair<Key, Value>& found_pair = *found;
                        atomicAdd(&(found_pair.second), v);
                        return 0;
                      });
  }
}

template <typename Key, typename Value>
static void BM_cudf(::benchmark::State& state)
{
  std::string msg{"cudf rbk: "};
  msg += std::to_string(state.range(0));
  auto const num_unique_keys = state.range(1);

  for (auto _ : state) {
    state.PauseTiming();
    thrust::device_vector<Key> keys(state.range(0));
    auto begin = thrust::make_counting_iterator(0);
    thrust::transform(
      begin, begin + state.range(0), keys.begin(), [num_unique_keys] __device__(auto i) {
        return i % num_unique_keys;
      });
    auto values = thrust::counting_iterator<int32_t>(0);
    state.ResumeTiming();
    cudf_reduce_by_key(keys.begin(), keys.end(), values);
    cudaDeviceSynchronize();
  }
}
BENCHMARK_TEMPLATE(BM_cudf, unsigned long long int, unsigned long long int)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_num_unique);

/*
template <typename KeyRandomIterator, typename ValueRandomIterator>
void cuco_cas_reduce_by_key(KeyRandomIterator keys_begin,
                            KeyRandomIterator keys_end,
                            ValueRandomIterator values_begin)
{
  using Key   = typename thrust::iterator_traits<KeyRandomIterator>::value_type;
  using Value = typename thrust::iterator_traits<ValueRandomIterator>::value_type;

  auto const input_size          = thrust::distance(keys_begin, keys_end);
  auto const occupancy           = 0.5;
  std::size_t const map_capacity = input_size / occupancy;

  using map_type = cuco::insert_only_hash_array<Key, Value, cuda::thread_scope_device>;

  map_type map{map_capacity, -1, -1};

  auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(keys_begin, values_begin));
  auto zipped_end   = zipped_begin + thrust::distance(keys_begin, keys_end);

  std::cout << "hello\n";

  thrust::for_each(thrust::device,
                   zipped_begin,
                   zipped_end,
                   [view = map.get_device_view()] __device__(auto const& key_value) mutable {
                     auto k = thrust::get<0>(key_value);
                     auto v = thrust::get<1>(key_value);

      auto found = view.find(k);

      if (view.end() == found) {
        auto result = view.insert(cuco::make_pair(k, 0));
        found       = result.first;
      }

                     // TODO: JH: This is currently causing a hang for some reason.

                     found->second.fetch_add(v, cuda::std::memory_order_relaxed);
    });
}

template <typename Key, typename Value>
static void BM_cuco_cas(::benchmark::State& state)
{
  std::string msg{"cudf rbk: "};
  msg += std::to_string(state.range(0));
  auto const num_unique_keys = state.range(1);

  for (auto _ : state) {
    state.PauseTiming();
    thrust::device_vector<Key> keys(state.range(0));
    auto begin = thrust::make_counting_iterator(1);
    thrust::transform(
      begin, begin + state.range(0), keys.begin(), [num_unique_keys] __device__(auto i) {
        return i % num_unique_keys;
      });
    auto values = thrust::counting_iterator<int32_t>(0);
    state.ResumeTiming();
    cuco_cas_reduce_by_key(keys.begin(), keys.end(), values);
    cudaDeviceSynchronize();
  }
}

BENCHMARK_TEMPLATE(BM_cuco_cas, int64_t, int64_t)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_num_unique);
  */