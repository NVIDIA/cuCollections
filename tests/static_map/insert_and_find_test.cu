/*
 * Copyright (c) 2022, Jonas Hahnfeld, CERN.
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <utils.hpp>

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

#include <catch2/catch.hpp>

static constexpr int Iters = 10'000;

template <typename View>
__global__ void parallel_sum(View v)
{
  for (int i = 0; i < Iters; i++) {
#if __CUDA_ARCH__ < 700
    if constexpr (cuco::detail::is_packable<View::value_type>())
#endif
    {
      auto [iter, inserted] = v.insert_and_find(thrust::make_pair(i, 1));
      // for debugging...
      // if (iter->second < 0) {
      //   asm("trap;");
      // }
      if (!inserted) { iter->second += 1; }
    }
#if __CUDA_ARCH__ < 700
    else {
      v.insert(thrust::make_pair(i, gridDim.x * blockDim.x));
    }
#endif
  }
}

TEMPLATE_TEST_CASE_SIG("Parallel insert-or-update",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t),
                       (int32_t, int64_t),
                       (int64_t, int32_t),
                       (int64_t, int64_t))
{
  cuco::empty_key<Key> empty_key_sentinel{-1};
  cuco::empty_value<Value> empty_value_sentinel{-1};
  cuco::static_map<Key, Value> m(10 * Iters, empty_key_sentinel, empty_value_sentinel);

  static constexpr int Blocks  = 1024;
  static constexpr int Threads = 128;
  parallel_sum<<<Blocks, Threads>>>(m.get_device_mutable_view());
  cudaDeviceSynchronize();

  thrust::device_vector<Key> d_keys(Iters);
  thrust::device_vector<Value> d_values(Iters);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  m.find(d_keys.begin(), d_keys.end(), d_values.begin());

  REQUIRE(cuco::test::all_of(
    d_values.begin(), d_values.end(), [] __device__(Value v) { return v == Blocks * Threads; }));
}
