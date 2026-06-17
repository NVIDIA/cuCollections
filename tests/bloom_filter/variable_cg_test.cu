/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

// Exercises `cuco::bloom_filter_ref` device-side APIs (`ref.add(key)`, `ref.contains(key)`,
// `ref.contains(group, key)`) directly from custom kernels, varying the cooperative group size.
// The bulk host-side APIs route through the warp-cooperative kernel which already had correct
// CG-reduction semantics; the scalar and CG ref methods are separate code paths that need their
// own coverage.

#include <test_utils.hpp>

#include <cuco/bloom_filter.cuh>

#include <cuda/std/functional>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cooperative_groups.h>

#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>

using size_type = int32_t;

namespace cg = cooperative_groups;

template <class Ref, class Key>
__global__ void scalar_add_kernel(Ref ref, Key const* keys, size_type n)
{
  auto const i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) { ref.add(keys[i]); }
}

template <class Ref, class Key>
__global__ void scalar_contains_kernel(Ref ref, Key const* keys, size_type n, bool* out)
{
  auto const i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = ref.contains(keys[i]); }
}

template <int CGSize, class Ref, class Key>
__global__ void cg_contains_consistency_kernel(Ref ref,
                                               Key const* keys,
                                               size_type n,
                                               int* mismatches)
{
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<CGSize>(block);
  auto const idx   = static_cast<size_type>((blockIdx.x * blockDim.x + threadIdx.x) / CGSize);
  if (idx >= n) { return; }
  bool const got       = ref.contains(tile, keys[idx]);
  bool const all_agree = tile.all(got);
  bool const any_agree = tile.any(got);
  if (tile.thread_rank() == 0 && all_agree != any_agree) { atomicAdd(mismatches, 1); }
}

template <int CGSize, class Ref>
__global__ void cooperative_clear_kernel(Ref ref)
{
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<CGSize>(block);
  ref.clear(tile);
}

TEMPLATE_TEST_CASE_SIG(
  "bloom_filter device ref scalar add and contains",
  "",
  ((class Key, class Policy), Key, Policy),
  (int32_t, cuco::default_filter_policy<int32_t>),
  (int32_t, cuco::parametric_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 1, 1, 1, 1, 1, 1>),
  (int32_t, cuco::parametric_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 8, 8, 4, 2, 4, 2>))
{
  using filter_type =
    cuco::bloom_filter<Key, cuco::extent<size_t>, cuda::thread_scope_device, Policy>;
  constexpr size_type num_keys{400};

  auto filter = filter_type{1000};

  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end());
  thrust::device_vector<bool> contained(num_keys, false);

  auto ref = filter.ref();

  constexpr int block_size = 128;
  int const grid_size      = (num_keys + block_size - 1) / block_size;

  scalar_add_kernel<<<grid_size, block_size>>>(
    ref, thrust::raw_pointer_cast(keys.data()), num_keys);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  scalar_contains_kernel<<<grid_size, block_size>>>(ref,
                                                    thrust::raw_pointer_cast(keys.data()),
                                                    num_keys,
                                                    thrust::raw_pointer_cast(contained.data()));
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  REQUIRE(cuco::test::all_of(contained.begin(), contained.end(), cuda::std::identity{}));
}

TEMPLATE_TEST_CASE_SIG(
  "bloom_filter device ref CG contains is reduced across the group",
  "",
  ((int32_t CGSize, class Key, class Policy), CGSize, Key, Policy),
  (4,
   int32_t,
   cuco::parametric_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 8, 8, 4, 2, 4, 2>),
  (8,
   int32_t,
   cuco::parametric_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 8, 8, 8, 1, 8, 1>))
{
  using filter_type =
    cuco::bloom_filter<Key, cuco::extent<size_t>, cuda::thread_scope_device, Policy>;
  // Small filter + many probes drive non-inserted keys into the partial-match regime where
  // different lanes in the tile see different per-slice match results. Without a group reduction
  // in `ref.contains(group, key)`, each lane returns its slice's partial result and the tile
  // disagrees on the answer.
  constexpr size_type num_inserted{200};
  constexpr size_type num_probed{2000};
  constexpr size_type num_blocks{16};

  auto filter = filter_type{num_blocks};
  auto ref    = filter.ref();

  thrust::device_vector<Key> insert_keys(num_inserted);
  thrust::sequence(thrust::device, insert_keys.begin(), insert_keys.end());
  filter.add(insert_keys.begin(), insert_keys.end());

  thrust::device_vector<Key> probe_keys(num_probed);
  thrust::sequence(thrust::device, probe_keys.begin(), probe_keys.end());

  thrust::device_vector<int> mismatches(1, 0);

  constexpr int block_size = 128;
  int const grid_size      = (num_probed * CGSize + block_size - 1) / block_size;

  cg_contains_consistency_kernel<CGSize>
    <<<grid_size, block_size>>>(ref,
                                thrust::raw_pointer_cast(probe_keys.data()),
                                num_probed,
                                thrust::raw_pointer_cast(mismatches.data()));
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  REQUIRE(static_cast<int>(mismatches[0]) == 0);
}

TEMPLATE_TEST_CASE_SIG("bloom_filter device ref cooperative clear",
                       "",
                       ((int32_t CGSize, class Key, class Policy), CGSize, Key, Policy),
                       (1, int32_t, cuco::default_filter_policy<int32_t>),
                       (4, int32_t, cuco::default_filter_policy<int32_t>),
                       (8, int32_t, cuco::default_filter_policy<int32_t>),
                       (32, int32_t, cuco::default_filter_policy<int32_t>))
{
  using filter_type =
    cuco::bloom_filter<Key, cuco::extent<size_t>, cuda::thread_scope_device, Policy>;
  constexpr size_type num_keys{400};

  auto filter = filter_type{1000};

  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end());
  filter.add(keys.begin(), keys.end());

  thrust::device_vector<bool> contained(num_keys, false);
  filter.contains(keys.begin(), keys.end(), contained.begin());
  REQUIRE(cuco::test::all_of(contained.begin(), contained.end(), cuda::std::identity{}));

  // Device cooperative clear via a single tile that iterates over all filter words.
  cooperative_clear_kernel<CGSize><<<1, CGSize>>>(filter.ref());
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  filter.contains(keys.begin(), keys.end(), contained.begin());
  REQUIRE(cuco::test::none_of(contained.begin(), contained.end(), cuda::std::identity{}));
}
