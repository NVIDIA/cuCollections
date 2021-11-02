#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <cuco/priority_queue.cuh>
#include <cuco/detail/error.hpp>
#include <cuco/detail/pair.cuh>

#include <thrust/device_vector.h>

using namespace cuco;

template <typename T>
struct pair_less {
  __host__ __device__ bool operator()(const T& a, const T& b) const {
    return a.first < b.first;
  }
};

constexpr int NUM_KEYS = 128e6;

template <typename Key, typename Value>
static void BM_insert(::benchmark::State& state)
{
  srand(0);
  for (auto _ : state) {
    state.PauseTiming();
    priority_queue<pair<Key, Value>, pair_less<pair<Key, Value>>> pq(NUM_KEYS);
    std::vector<pair<Key, Value>> h_pairs(NUM_KEYS);
    for (auto &p : h_pairs) {
      p = {rand(), rand()};
    }
    thrust::device_vector<pair<Key, Value>> d_pairs(h_pairs);
    state.ResumeTiming();
    pq.push(d_pairs.begin(), d_pairs.end());
    cudaDeviceSynchronize();
  }
  
}

template <typename Key, typename Value>
static void BM_delete(::benchmark::State& state)
{
  srand(0);
  for (auto _ : state) {
    state.PauseTiming();
    priority_queue<pair<Key, Value>, pair_less<pair<Key, Value>>> pq(NUM_KEYS);
    std::vector<pair<Key, Value>> h_pairs(NUM_KEYS);
    for (auto &p : h_pairs) {
      p = {rand(), rand()};
    }
    thrust::device_vector<pair<Key, Value>> d_pairs(h_pairs);
    pq.push(d_pairs.begin(), d_pairs.end());
    cudaDeviceSynchronize();
    state.ResumeTiming();
    pq.pop(d_pairs.begin(), d_pairs.end());
    cudaDeviceSynchronize();
  }
  
}

BENCHMARK_TEMPLATE(BM_insert, int, int)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_delete, int, int)
  ->Unit(benchmark::kMillisecond);


/*
int main() {

  InsertThenDelete();

  return 0;
}*/
