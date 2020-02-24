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

#include <cu_collections/utilities/error.hpp>
#include <insert_only_hash_array.cuh>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <catch.hpp>

template <typename Map, typename Pairs>
__global__ void insert_kernel(Map map, Pairs const* pairs, size_t size) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) map.insert(pairs[tid]);
}

TEST_CASE("The first test") {
  insert_only_hash_array<int32_t, int32_t> a{1000, -1, -1};

  CUDA_TRY(cudaDeviceSynchronize());

  auto view = a.get_device_view();

  std::vector<thrust::pair<int32_t, int32_t>> pairs(100);
  std::generate(pairs.begin(), pairs.end(), []() {
    static int32_t counter{};
    ++counter;
    return thrust::make_pair(counter, counter);
  });

  thrust::device_vector<thrust::pair<int32_t, int32_t>> d_pairs(pairs);

  CUDA_TRY(cudaDeviceSynchronize());

  // thrust::for_each(
  //    d_pairs.begin(), d_pairs.end(),
  //    [view] __device__(
  //        thrust::pair<const int32_t, int32_t> const& pair) mutable {
  //      view.insert(pair);
  //    });

  insert_kernel<<<5, 256>>>(view, d_pairs.data().get(), d_pairs.size());

  CUDA_TRY(cudaDeviceSynchronize());
}
