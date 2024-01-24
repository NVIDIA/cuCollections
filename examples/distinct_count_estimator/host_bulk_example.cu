/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cuco/distinct_count_estimator.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cstddef>
#include <iostream>

int main()
{
  using T                         = int;
  std::size_t constexpr num_items = 1ull << 30;  // 4GB

  thrust::device_vector<T> items(num_items);
  // create a vector of distinct items
  thrust::sequence(items.begin(), items.end(), 0);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cuco::distinct_count_estimator<T> estimator;
  cudaEventRecord(start);
  // add all items to the estimator
  estimator.add(items.begin(), items.end());
  // after the estimator has seen all items, we can calculate the cardinality
  std::size_t const estimated_cardinality = estimator.estimate();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float input_size_gb = num_items * sizeof(T) / 1073741824.0f;
  float throughput    = input_size_gb / (milliseconds / 1000.0f);

  std::cout << "True cardinality:\t" << num_items << "\nEstimated cardinality:\t"
            << estimated_cardinality << "\nRelative error:\t"
            << abs(static_cast<double>(num_items) - static_cast<double>(estimated_cardinality)) /
                 num_items
            << "\nData size:\t" << input_size_gb << "GB"
            << "\nElapsed time:\t" << milliseconds << "ms"
            << "\nMemory throughput\t" << throughput << "GB/s" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}