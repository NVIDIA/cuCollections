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

#pragma once

// Google Benchmark library
#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#define BENCH_CUDA_TRY(call)                                                         \
  do {                                                                               \
    auto const status = (call);                                                      \
    if (cudaSuccess != status) { throw std::runtime_error("CUDA error detected."); } \
  } while (0);

/**
 * @brief  This class serves as a wrapper for using `cudaEvent_t` as the user
 * defined timer within the framework of google benchmark
 * (https://github.com/google/benchmark).
 *
 * It is built on top of the idea of Resource acquisition is initialization
 * (RAII). In the following we show a minimal example of how to use this class.
 *
 * \code{cpp}
 * #include <benchmark/benchmark.h>
 *
 * static void sample_cuda_benchmark(benchmark::State& state) {
 *
 *   for (auto _ : state){
 *     cudaStream_t stream = 0;
 *
 *     // Create (Construct) an object of this class. You HAVE to pass in the
 *     // benchmark::State object you are using. It measures the time from its
 *     // creation to its destruction that is spent on the specified CUDA stream.
 *     // It also clears the L2 cache by cudaMemset'ing a device buffer that is of
 *     // the size of the L2 cache (if flush_l2_cache is set to true and there is
 *     // an L2 cache on the current device).
 *     cuda_event_timer raii(state, true, stream); // flush_l2_cache = true
 *
 *     // Now perform the operations that is to be benchmarked
 *     sample_kernel<<<1, 256, 0, stream>>>(); // Possibly launching a CUDA kernel
 *
 *   }
 * }
 *
 * // Register the function as a benchmark. You will need to set the `UseManualTime()`
 * // flag in order to use the timer embeded in this class.
 * BENCHMARK(sample_cuda_benchmark)->UseManualTime();
 * \endcode
 *
 *
 */
class cuda_event_timer {
 public:
  /**
   * @brief Constructs a `cuda_event_timer` beginning a manual timing range.
   *
   * Optionally flushes L2 cache.
   *
   * @param[in,out] state  This is the benchmark::State whose timer we are going
   * to update.
   * @param[in] flush_l2_cache_ whether or not to flush the L2 cache before
   *                            every iteration.
   * @param[in] stream_ The CUDA stream we are measuring time on.
   */
  cuda_event_timer(benchmark::State &state, bool flush_l2_cache = false, cudaStream_t stream = 0)
    : p_state(&state), stream_(stream)
  {
    // flush all of L2$
    if (flush_l2_cache) {
      int current_device = 0;
      BENCH_CUDA_TRY(cudaGetDevice(&current_device));

      int l2_cache_bytes = 0;
      BENCH_CUDA_TRY(
        cudaDeviceGetAttribute(&l2_cache_bytes, cudaDevAttrL2CacheSize, current_device));

      if (l2_cache_bytes > 0) {
        const int memset_value = 0;
        int *l2_cache_buffer   = nullptr;
        BENCH_CUDA_TRY(cudaMalloc(&l2_cache_buffer, l2_cache_bytes));
        BENCH_CUDA_TRY(cudaMemsetAsync(l2_cache_buffer, memset_value, l2_cache_bytes, stream_));
        BENCH_CUDA_TRY(cudaFree(l2_cache_buffer));
      }
    }

    BENCH_CUDA_TRY(cudaEventCreate(&start_));
    BENCH_CUDA_TRY(cudaEventCreate(&stop_));
    BENCH_CUDA_TRY(cudaEventRecord(start_, stream_));
  }

  cuda_event_timer() = delete;

  /**
   * @brief Destroy the `cuda_event_timer` and ending the manual time range.
   *
   */
  ~cuda_event_timer()
  {
    BENCH_CUDA_TRY(cudaEventRecord(stop_, stream_));
    BENCH_CUDA_TRY(cudaEventSynchronize(stop_));
    float milliseconds = 0.0f;
    BENCH_CUDA_TRY(cudaEventElapsedTime(&milliseconds, start_, stop_));
    p_state->SetIterationTime(milliseconds / (1000.0f));
    BENCH_CUDA_TRY(cudaEventDestroy(start_));
    BENCH_CUDA_TRY(cudaEventDestroy(stop_));
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  cudaStream_t stream_;
  benchmark::State *p_state;
};