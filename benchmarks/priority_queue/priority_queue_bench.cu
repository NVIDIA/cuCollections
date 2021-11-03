#include <vector>
#include <cstdint>
#include <random>

#include <benchmark/benchmark.h>

#include <cuco/priority_queue.cuh>
#include <cuco/detail/pair.cuh>

#include <thrust/device_vector.h>

using namespace cuco;

template <typename T>
struct pair_less {
  __host__ __device__ bool operator()(const T& a, const T& b) const {
    return a.first < b.first;
  }
};

template<typename Key, typename Value, typename OutputIt>
static void generate_keys_uniform(OutputIt output_begin, OutputIt output_end) {
  std::random_device rd;
  std::mt19937 gen{rd()};

  auto num_keys = std::distance(output_begin, output_end);

  for (auto i = 0; i < num_keys; ++i) {
    output_begin[i] = {static_cast<Key>(gen()), static_cast<Value>(gen())};
  }
}

template <typename Key, typename Value, int NumKeys>
static void BM_insert(::benchmark::State& state)
{
  for (auto _ : state) {
    state.PauseTiming();

    priority_queue<pair<Key, Value>, pair_less<pair<Key, Value>>> pq(NumKeys);

    std::vector<pair<Key, Value>> h_pairs(NumKeys);
    generate_keys_uniform<Key, Value>(h_pairs.begin(), h_pairs.end());
    thrust::device_vector<pair<Key, Value>> d_pairs(h_pairs);

    state.ResumeTiming();
    pq.push(d_pairs.begin(), d_pairs.end());
    cudaDeviceSynchronize();
  }
  
}

template <typename Key, typename Value, int NumKeys>
static void BM_delete(::benchmark::State& state)
{
  for (auto _ : state) {
    state.PauseTiming();

    priority_queue<pair<Key, Value>, pair_less<pair<Key, Value>>> pq(NumKeys);

    std::vector<pair<Key, Value>> h_pairs(NumKeys);
    generate_keys_uniform<Key, Value>(h_pairs.begin(), h_pairs.end());
    thrust::device_vector<pair<Key, Value>> d_pairs(h_pairs);

    pq.push(d_pairs.begin(), d_pairs.end());
    cudaDeviceSynchronize();

    state.ResumeTiming();
    pq.pop(d_pairs.begin(), d_pairs.end());
    cudaDeviceSynchronize();
  }
  
}

BENCHMARK_TEMPLATE(BM_insert, int, int, 128'000'000)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_delete, int, int, 128'000'000)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_insert, int, int, 256'000'000)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_delete, int, int, 256'000'000)
  ->Unit(benchmark::kMillisecond);
