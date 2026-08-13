/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/bloom_filter.cuh>
#include <cuco/detail/error.hpp>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cstddef>
#include <cstdint>

int main(void)
{
  /**
   * This example shows how to create a Bloom filter that marks its filter-word accesses as
   * persisting in L2 and how to reserve/reset the corresponding L2 set-aside with the CUDA
   * runtime API.
   *
   * The `PersistingL2Access=true` policy parameter makes `cuco::bloom_filter` emit per-access L2
   * persisting cache hints for global-memory filter words. The policy does not reserve cache
   * capacity by itself; applications should reserve a persisting-L2 set-aside before the
   * Bloom-filter phase with `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...)` and reset
   * persisting lines afterwards with `cudaCtxResetPersistingL2Cache()`.
   *
   * Use this only when the filter fits in the set-aside/L2 capacity, or when the key stream has
   * enough locality that persisted filter lines are reused. If the filter is much larger than L2,
   * random accesses can continually mark cold lines as persisting, thrash the persisting region,
   * and slow unrelated kernels until the persisting cache is reset.
   *
   * NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 128 MiB L2, 200M input items:
   *
   * filter size | add, default | add, persisting L2 | contains, default | contains, persisting L2
   * ----------- | ------------ | ------------------ | ----------------- | ------------------------
   * 64 MiB      | 4.880 ms     | 3.405 ms           | 2.444 ms          | 2.427 ms
   * 80 MiB      | 8.276 ms     | 3.380 ms           | 4.706 ms          | 2.443 ms
   * 96 MiB      | 11.004 ms    | 3.476 ms           | 4.874 ms          | 2.575 ms
   * 128 MiB     | 22.602 ms    | 9.097 ms           | 5.545 ms          | 5.371 ms
   */

  using key_type = int;

  int constexpr num_keys = 10'000;

  // Create a Bloom filter policy with persisting L2 access enabled. This is the same shape as the
  // default policy, except the final `PersistingL2Access` parameter is `true`.
  using policy_type              = cuco::bloom_filter_policy<key_type,
                                                             cuco::xxhash_64<key_type>,
                                                             4,
                                                             8,
                                                             8,
                                                             8,
                                                             1,
                                                             1,
                                                             8,
                                                             false,
                                                             false,
                                                             true>;  ///< Persisting L2 access enabled.
  auto constexpr bytes_per_block = sizeof(policy_type::word_type) * policy_type::words_per_block;

  int device = 0;
  CUCO_CUDA_TRY(cudaGetDevice(&device));

  int max_persisting_l2_cache_size = 0;
  CUCO_CUDA_TRY(cudaDeviceGetAttribute(
    &max_persisting_l2_cache_size, cudaDevAttrMaxPersistingL2CacheSize, device));

  // Create a filter that occupies 80% of the maximum supported persisting-L2 set-aside, then
  // reserve enough persisting L2 for that filter before launching Bloom-filter kernels.
  auto constexpr l2_set_aside_fraction = 0.8;
  auto const target_filter_size =
    static_cast<std::size_t>(l2_set_aside_fraction * max_persisting_l2_cache_size);
  CUCO_CUDA_TRY(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, target_filter_size));

  auto const sub_filters = target_filter_size / bytes_per_block;

  // This filter marks global-memory filter-word loads and atomics as persisting-L2 accesses.
  cuco::bloom_filter<key_type, cuco::extent<std::size_t>, cuda::thread_scope_device, policy_type>
    filter{sub_filters};

  thrust::device_vector<key_type> keys(num_keys);
  thrust::sequence(keys.begin(), keys.end(), 1);

  // Insert all keys into the filter.
  filter.add(keys.begin(), keys.end());

  thrust::device_vector<bool> result(num_keys, false);

  // Query the same keys.
  filter.contains(keys.begin(), keys.end(), result.begin());

  // Evict persisting lines so unrelated later kernels do not inherit stale cache residency.
  CUCO_CUDA_TRY(cudaCtxResetPersistingL2Cache());

  // Release the persisting-L2 set-aside.
  CUCO_CUDA_TRY(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));

  return 0;
}
