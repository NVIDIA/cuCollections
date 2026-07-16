/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// `PersistingL2Access` is a code-generation hint for global-memory filter words. It must preserve
// observable Bloom-filter results, and it must be safe to use with a `bloom_filter_ref` backed by
// shared memory where L2 access properties are intentionally ignored.

#include <test_utils.hpp>

#include <cuco/bloom_filter.cuh>
#include <cuco/bloom_filter_ref.cuh>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cooperative_groups.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>

namespace cg = cooperative_groups;

namespace {

using key_type = int32_t;

using shared_ref_type = cuco::bloom_filter_ref<key_type,
                                               cuco::extent<uint32_t>,
                                               cuda::thread_scope_block,
                                               cuco::bloom_filter_policy<key_type,
                                                                         cuco::xxhash_64<key_type>,
                                                                         uint32_t,
                                                                         8,
                                                                         8,
                                                                         8,
                                                                         1,
                                                                         1,
                                                                         8,
                                                                         false,
                                                                         false,
                                                                         true>>;

__global__ void shared_memory_persisting_l2_smoke_kernel(int* results)
{
  __shared__ typename shared_ref_type::filter_block_type storage[1];

  auto block = cg::this_thread_block();
  shared_ref_type filter{storage, cuco::extent<uint32_t>{1}, {}, {}};

  filter.clear(block);
  block.sync();

  if (threadIdx.x == 0) {
    filter.add(42);
    filter.add(1337);
  }
  block.sync();

  if (threadIdx.x == 0) {
    results[0] = filter.contains(42) ? 1 : 0;
    results[1] = filter.contains(1337) ? 1 : 0;
  }
}

}  // namespace

TEST_CASE("bloom_filter: PersistingL2Access preserves global-memory results", "")
{
  using normal_filter_type = cuco::bloom_filter<key_type,
                                                cuco::extent<std::size_t>,
                                                cuda::thread_scope_device,
                                                cuco::bloom_filter_policy<key_type>>;
  using persisting_l2_filter_type =
    cuco::bloom_filter<key_type,
                       cuco::extent<std::size_t>,
                       cuda::thread_scope_device,
                       cuco::bloom_filter_policy<key_type,
                                                 cuco::xxhash_64<key_type>,
                                                 uint32_t,
                                                 8,
                                                 8,
                                                 8,
                                                 1,
                                                 1,
                                                 8,
                                                 false,
                                                 false,
                                                 true>>;

  STATIC_REQUIRE_FALSE(cuco::bloom_filter_policy<key_type>::persisting_l2_access);
  STATIC_REQUIRE_FALSE((cuco::bloom_filter_policy<key_type>::persisting_l2_access));
  STATIC_REQUIRE((cuco::bloom_filter_policy<key_type,
                                            cuco::xxhash_64<key_type>,
                                            uint32_t,
                                            8,
                                            8,
                                            8,
                                            1,
                                            1,
                                            8,
                                            false,
                                            false,
                                            true>::persisting_l2_access));

  constexpr int32_t num_blocks = 1'000;
  constexpr int32_t num_keys   = 400;
  constexpr int32_t num_probe  = 800;

  auto normal_filter        = normal_filter_type{num_blocks};
  auto persisting_l2_filter = persisting_l2_filter_type{num_blocks};

  thrust::device_vector<key_type> insert_keys(num_keys);
  thrust::sequence(thrust::device, insert_keys.begin(), insert_keys.end());

  normal_filter.add(insert_keys.begin(), insert_keys.end());
  persisting_l2_filter.add(insert_keys.begin(), insert_keys.end());

  auto const total_words =
    static_cast<std::size_t>(normal_filter.block_extent()) * normal_filter_type::words_per_block;
  REQUIRE(thrust::equal(thrust::device,
                        normal_filter.data(),
                        normal_filter.data() + total_words,
                        persisting_l2_filter.data()));

  thrust::device_vector<key_type> probe_keys(num_probe);
  thrust::sequence(thrust::device, probe_keys.begin(), probe_keys.end());

  thrust::device_vector<bool> normal_results(num_probe);
  thrust::device_vector<bool> persisting_l2_results(num_probe);
  normal_filter.contains(probe_keys.begin(), probe_keys.end(), normal_results.begin());
  persisting_l2_filter.contains(
    probe_keys.begin(), probe_keys.end(), persisting_l2_results.begin());

  REQUIRE(thrust::equal(
    thrust::device, normal_results.begin(), normal_results.end(), persisting_l2_results.begin()));
}

TEST_CASE("bloom_filter: PersistingL2Access is safe for shared-memory refs", "")
{
  thrust::device_vector<int> results(2, 0);

  shared_memory_persisting_l2_smoke_kernel<<<1, 32>>>(thrust::raw_pointer_cast(results.data()));
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  REQUIRE(static_cast<int>(results[0]) == 1);
  REQUIRE(static_cast<int>(results[1]) == 1);
}
