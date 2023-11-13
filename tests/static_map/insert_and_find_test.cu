/*
 * Copyright (c) 2022, Jonas Hahnfeld, CERN.
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <catch2/catch_template_test_macros.hpp>

static constexpr int Iters = 10'000;

template <typename Ref>
__global__ void parallel_sum(Ref v)
{
  for (int i = 0; i < Iters; i++) {
#if __CUDA_ARCH__ < 700
    if constexpr (cuco::detail::is_packable<Ref::value_type>())
#endif
    {
      auto constexpr cg_size = Ref::cg_size;
      if constexpr (cg_size == 1) {
        auto [iter, inserted] = v.insert_and_find(cuco::pair{i, 1});
        // for debugging...
        // if (iter->second < 0) {
        //   asm("trap;");
        // }
        if (!inserted) {
          auto ref =
            cuda::atomic_ref<typename Ref::mapped_type, cuda::thread_scope_device>{iter->second};
          ref.fetch_add(1);
        }
      } else {
        auto const tile =
          cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());
        auto [iter, inserted] = v.insert_and_find(tile, cuco::pair{i, 1});
        if (!inserted and tile.thread_rank() == 0) {
          auto ref =
            cuda::atomic_ref<typename Ref::mapped_type, cuda::thread_scope_device>{iter->second};
          ref.fetch_add(1);
        }
      }
    }
#if __CUDA_ARCH__ < 700
    else {
      v.insert(cuco::pair{i, gridDim.x * blockDim.x});
    }
#endif
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_map insert_and_find tests",
  "",
  ((typename Key, typename Value, cuco::test::probe_sequence Probe, int CGSize),
   Key,
   Value,
   Probe,
   CGSize),
  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int32_t, int64_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, int64_t, cuco::test::probe_sequence::linear_probing, 2))
{
  using probe =
    std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                       cuco::experimental::linear_probing<CGSize, cuco::murmurhash3_32<Key>>,
                       cuco::experimental::double_hashing<CGSize,
                                                          cuco::murmurhash3_32<Key>,
                                                          cuco::murmurhash3_32<Key>>>;

  auto map = cuco::experimental::static_map<Key,
                                            Value,
                                            cuco::experimental::extent<std::size_t>,
                                            cuda::thread_scope_device,
                                            thrust::equal_to<Key>,
                                            probe,
                                            cuco::cuda_allocator<std::byte>,
                                            cuco::experimental::storage<2>>{
    10 * Iters, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};

  static constexpr int Blocks  = 1024;
  static constexpr int Threads = 128;

  parallel_sum<<<Blocks, Threads>>>(
    map.ref(cuco::experimental::op::insert, cuco::experimental::op::insert_and_find));
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  thrust::device_vector<Key> d_keys(Iters);
  thrust::device_vector<Value> d_values(Iters);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  map.find(d_keys.begin(), d_keys.end(), d_values.begin());

  REQUIRE(cuco::test::all_of(d_values.begin(), d_values.end(), [] __device__(Value v) {
    return v == (Blocks * Threads) / CGSize;
  }));
}
