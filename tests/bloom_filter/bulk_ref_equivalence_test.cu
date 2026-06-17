/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

// Host-bulk vs device-ref equivalence. The host-bulk APIs route through `add_n` /
// `contains_n` in `kernels.cuh`; the device-ref APIs route through scalar / CG / CG-range
// methods on `bloom_filter_ref` directly. Both paths share `policy_.array_pattern(...)`,
// so given the same input keys both must produce byte-identical filter bitsets (add side)
// and byte-identical query results (contains side). Catches regressions like the
// `add_coop(group, first, idx, is_valid)` iterator-formation UB fix where one path
// silently diverged from the other.

#include <test_utils.hpp>

#include <cuco/bloom_filter.cuh>
#include <cuco/detail/error.hpp>

#include <cuda/functional>
#include <cuda/std/functional>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cooperative_groups.h>

#include <catch2/catch_template_test_macros.hpp>

#include <cstddef>
#include <cstdint>

namespace cg = cooperative_groups;

using size_type = int32_t;

template <class Ref, class Key>
__global__ void scalar_add_kernel(Ref ref, Key const* keys, size_type n)
{
  auto const i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) { ref.add(keys[i]); }
}

template <int CGSize, class Ref, class Key>
__global__ void cg_add_kernel(Ref ref, Key const* keys, size_type n)
{
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<CGSize>(block);
  auto const idx   = static_cast<size_type>((blockIdx.x * blockDim.x + threadIdx.x) / CGSize);
  if (idx < n) { ref.add(tile, keys[idx]); }
}

template <int CGSize, class Ref, class Key>
__global__ void cg_range_add_kernel(Ref ref, Key const* first, Key const* last)
{
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<CGSize>(block);
  ref.add(tile, first, last);
}

template <class Ref, class Key>
__global__ void scalar_contains_kernel(Ref ref, Key const* keys, size_type n, bool* out)
{
  auto const i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = ref.contains(keys[i]); }
}

template <int CGSize, class Ref, class Key>
__global__ void cg_contains_kernel(Ref ref, Key const* keys, size_type n, bool* out)
{
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<CGSize>(block);
  auto const idx   = static_cast<size_type>((blockIdx.x * blockDim.x + threadIdx.x) / CGSize);
  if (idx < n) {
    auto const found = ref.contains(tile, keys[idx]);
    if (tile.thread_rank() == 0) { out[idx] = found; }
  }
}

template <int CGSize, class Ref, class Key>
__global__ void cg_range_contains_kernel(Ref ref, Key const* first, Key const* last, bool* out)
{
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<CGSize>(block);
  ref.contains(tile, first, last, out);
}

TEMPLATE_TEST_CASE_SIG(
  "bloom_filter: host bulk add equals device ref add",
  "",
  ((class Key, class Policy), Key, Policy),
  (int32_t, cuco::default_filter_policy<int32_t>),
  (int32_t, cuco::parametric_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 1, 1, 1, 1, 1, 1>),
  (uint64_t,
   cuco::parametric_filter_policy<cuco::xxhash_64<uint64_t>, uint32_t, 8, 12, 8, 1, 4, 2>),
  (float, cuco::parametric_filter_policy<cuco::xxhash_64<float>, uint64_t, 4, 4, 2, 2, 1, 2>),
  (int32_t, cuco::parametric_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 8, 8, 2, 2, 1, 8>))
{
  using filter_type =
    cuco::bloom_filter<Key, cuco::extent<std::size_t>, cuda::thread_scope_device, Policy>;
  // Prime: forces a partial boundary tile (mix of valid + invalid lanes) on every
  // CGSize > 1 path, exercising the cooperative is_valid mask logic.
  constexpr size_type num_keys = 397;

  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end());
  auto const keys_raw = thrust::raw_pointer_cast(keys.data());

  SECTION("scalar ref.add")
  {
    auto filter_a = filter_type{1000};
    auto filter_b = filter_type{1000};

    filter_a.add(keys.begin(), keys.end());

    auto ref                 = filter_b.ref();
    constexpr int block_size = 128;
    int const grid_size      = (num_keys + block_size - 1) / block_size;
    scalar_add_kernel<<<grid_size, block_size>>>(ref, keys_raw, num_keys);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    auto const total_words =
      static_cast<std::size_t>(filter_a.block_extent()) * filter_type::words_per_block;
    REQUIRE(thrust::equal(
      thrust::device, filter_a.data(), filter_a.data() + total_words, filter_b.data()));
  }

  SECTION("CG ref.add(group, key)")
  {
    constexpr int CGSize = Policy::add_horizontal_layout;
    auto filter_a        = filter_type{1000};
    auto filter_b        = filter_type{1000};

    filter_a.add(keys.begin(), keys.end());

    auto ref                 = filter_b.ref();
    constexpr int block_size = 128;
    int const grid_size      = (num_keys * CGSize + block_size - 1) / block_size;
    cg_add_kernel<CGSize><<<grid_size, block_size>>>(ref, keys_raw, num_keys);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    auto const total_words =
      static_cast<std::size_t>(filter_a.block_extent()) * filter_type::words_per_block;
    REQUIRE(thrust::equal(
      thrust::device, filter_a.data(), filter_a.data() + total_words, filter_b.data()));
  }

  SECTION("CG ref.add(group, first, last)")
  {
    constexpr int CGSize = Policy::add_horizontal_layout;
    auto filter_a        = filter_type{1000};
    auto filter_b        = filter_type{1000};

    filter_a.add(keys.begin(), keys.end());

    auto ref = filter_b.ref();
    // Single tile processes the entire range cooperatively.
    cg_range_add_kernel<CGSize><<<1, CGSize>>>(ref, keys_raw, keys_raw + num_keys);
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    auto const total_words =
      static_cast<std::size_t>(filter_a.block_extent()) * filter_type::words_per_block;
    REQUIRE(thrust::equal(
      thrust::device, filter_a.data(), filter_a.data() + total_words, filter_b.data()));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "bloom_filter: host bulk contains equals device ref contains",
  "",
  ((class Key, class Policy), Key, Policy),
  (int32_t, cuco::default_filter_policy<int32_t>),
  (int32_t, cuco::parametric_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 1, 1, 1, 1, 1, 1>),
  (uint64_t,
   cuco::parametric_filter_policy<cuco::xxhash_64<uint64_t>, uint32_t, 8, 12, 8, 1, 4, 2>),
  (float, cuco::parametric_filter_policy<cuco::xxhash_64<float>, uint64_t, 4, 4, 2, 2, 1, 2>),
  (int32_t, cuco::parametric_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 8, 8, 2, 2, 1, 8>))
{
  using filter_type =
    cuco::bloom_filter<Key, cuco::extent<std::size_t>, cuda::thread_scope_device, Policy>;
  // Primes: force partial boundary tiles on both insert and probe ranges across every
  // CGSize > 1 path.
  constexpr size_type num_keys  = 397;
  constexpr size_type num_probe = 797;  // mix of inserted (first ~half) and non-inserted (rest)

  auto filter = filter_type{1000};

  thrust::device_vector<Key> insert_keys(num_keys);
  thrust::sequence(thrust::device, insert_keys.begin(), insert_keys.end());
  filter.add(insert_keys.begin(), insert_keys.end());

  thrust::device_vector<Key> probe_keys(num_probe);
  thrust::sequence(thrust::device, probe_keys.begin(), probe_keys.end());
  auto const probe_raw = thrust::raw_pointer_cast(probe_keys.data());

  thrust::device_vector<bool> bulk_result(num_probe);
  filter.contains(probe_keys.begin(), probe_keys.end(), bulk_result.begin());

  SECTION("scalar ref.contains")
  {
    thrust::device_vector<bool> ref_result(num_probe);
    auto ref                 = filter.ref();
    constexpr int block_size = 128;
    int const grid_size      = (num_probe + block_size - 1) / block_size;
    scalar_contains_kernel<<<grid_size, block_size>>>(
      ref, probe_raw, num_probe, thrust::raw_pointer_cast(ref_result.data()));
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    REQUIRE(cuco::test::equal(
      bulk_result.begin(), bulk_result.end(), ref_result.begin(), cuda::std::equal_to<bool>{}));
  }

  SECTION("CG ref.contains(group, key)")
  {
    constexpr int CGSize = Policy::contains_horizontal_layout;
    thrust::device_vector<bool> ref_result(num_probe);
    auto ref                 = filter.ref();
    constexpr int block_size = 128;
    int const grid_size      = (num_probe * CGSize + block_size - 1) / block_size;
    cg_contains_kernel<CGSize><<<grid_size, block_size>>>(
      ref, probe_raw, num_probe, thrust::raw_pointer_cast(ref_result.data()));
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    REQUIRE(cuco::test::equal(
      bulk_result.begin(), bulk_result.end(), ref_result.begin(), cuda::std::equal_to<bool>{}));
  }

  SECTION("device-range CG ref.contains(group, first, last, out)")
  {
    constexpr int CGSize = Policy::contains_horizontal_layout;
    thrust::device_vector<bool> ref_result(num_probe);
    auto ref = filter.ref();
    // Single tile processes the entire range cooperatively.
    cg_range_contains_kernel<CGSize><<<1, CGSize>>>(
      ref, probe_raw, probe_raw + num_probe, thrust::raw_pointer_cast(ref_result.data()));
    CUCO_CUDA_TRY(cudaDeviceSynchronize());

    REQUIRE(cuco::test::equal(
      bulk_result.begin(), bulk_result.end(), ref_result.begin(), cuda::std::equal_to<bool>{}));
  }
}
