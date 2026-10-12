/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/bloom_filter.cuh>
#include <cuco/utility/error.hpp>

#include <thrust/device_vector.h>

#include <cooperative_groups.h>

#include <catch2/catch_template_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

struct identity_hash {
  __host__ __device__ constexpr std::uint64_t operator()(std::uint64_t key) const { return key; }
};

using horizontal_add_policy = cuco::bloom_filter_policy<std::uint64_t, identity_hash>;
using vertical_add_policy =
  cuco::bloom_filter_policy<std::uint64_t, identity_hash, 4, 8, 8, 1, 2, 4, 1>;

template <bool Cooperative, int CGSize, class Ref>
__global__ void add_last_block(Ref ref, std::uint64_t key)
{
  if constexpr (Cooperative) {
    auto tile =
      cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
    ref.add(tile, key);
  } else {
    if (threadIdx.x == 0) { ref.add(key); }
  }
}

template <int CGSize, class Ref>
__global__ void contains_last_block(Ref ref, std::uint64_t key, bool* result)
{
  auto tile = cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
  auto const contained = ref.contains(tile, key);
  if (threadIdx.x == 0) {
    result[0] = ref.contains(key);
    result[1] = contained;
  }
}

template <class Ref>
__global__ void clear_large_filter(Ref ref)
{
  ref.clear(cooperative_groups::this_thread_block());
}

}  // namespace

TEMPLATE_TEST_CASE("bloom_filter large word indices",
                   "[large]",
                   horizontal_add_policy,
                   vertical_add_policy)
{
  using policy_type              = TestType;
  using filter_type              = cuco::bloom_filter<std::uint64_t,
                                                      cuco::extent<std::uint32_t>,
                                                      cuda::thread_scope_device,
                                                      policy_type>;
  using word_type                = typename filter_type::word_type;
  constexpr auto words_per_block = filter_type::words_per_block;
  // The last block begins at word 2^32. A wrapped offset would alias the first block.
  constexpr std::uint32_t num_blocks = (std::size_t{1} << 32) / words_per_block + 1;
  constexpr std::size_t num_words    = std::size_t{num_blocks} * words_per_block;
  constexpr std::size_t bytes        = num_words * sizeof(word_type);
  constexpr std::uint64_t key =
    (std::uint64_t{std::numeric_limits<std::uint32_t>::max()} << 32) | 42;

  std::size_t free_bytes, total_bytes;
  CUCO_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (free_bytes < 2 * bytes + (std::size_t{1} << 30)) {
    SKIP("Large Bloom filter regression requires 33 GiB of free device memory");
  }

  auto filter      = filter_type{num_blocks};
  auto other       = filter_type{num_blocks};
  auto small       = filter_type{1};
  auto* data       = const_cast<word_type*>(filter.data());
  auto* tail       = data + num_words - words_per_block;
  auto* other_tail = const_cast<word_type*>(other.data()) + num_words - words_per_block;
  thrust::device_vector<std::uint64_t> keys(1, key);
  thrust::device_vector<bool> contained(2, false);

  // Obtain the expected fingerprint from a one-block filter with the same lower hash.
  small.add(keys.begin(), keys.end());
  std::array<word_type, words_per_block> expected, zero{};
  CUCO_CUDA_TRY(
    cudaMemcpy(expected.data(), small.data(), sizeof(expected), cudaMemcpyDeviceToHost));
  auto read_block = [](word_type const* ptr) {
    std::array<word_type, words_per_block> result;
    CUCO_CUDA_TRY(cudaMemcpy(result.data(), ptr, sizeof(result), cudaMemcpyDeviceToHost));
    return result;
  };
  auto reset_blocks = [&] {
    CUCO_CUDA_TRY(cudaMemset(data, 0, sizeof(expected)));
    CUCO_CUDA_TRY(cudaMemset(tail, 0, sizeof(expected)));
  };

  // Inspect the actual storage: add and contains can agree even when both offsets wrap.
  reset_blocks();
  add_last_block<false, policy_type::add_horizontal_layout>
    <<<1, words_per_block>>>(filter.ref(), key);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  CHECK(read_block(tail) == expected);
  CHECK(read_block(data) == zero);

  reset_blocks();
  add_last_block<true, policy_type::add_horizontal_layout>
    <<<1, policy_type::add_horizontal_layout>>>(filter.ref(), key);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  CHECK(read_block(tail) == expected);
  CHECK(read_block(data) == zero);

  reset_blocks();
  filter.add(keys.begin(), keys.end());
  CHECK(read_block(tail) == expected);
  CHECK(read_block(data) == zero);

  // Populate the correct block independently before testing scalar, CG and bulk contains.
  reset_blocks();
  CUCO_CUDA_TRY(cudaMemcpy(tail, expected.data(), sizeof(expected), cudaMemcpyHostToDevice));
  contains_last_block<policy_type::contains_horizontal_layout>
    <<<1, policy_type::contains_horizontal_layout>>>(
      filter.ref(), key, thrust::raw_pointer_cast(contained.data()));
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  CHECK(static_cast<bool>(contained[0]));
  CHECK(static_cast<bool>(contained[1]));
  filter.contains(keys.begin(), keys.end(), contained.begin());
  CHECK(static_cast<bool>(contained[0]));

  // Host clear and transforms must also process more than UINT32_MAX words.
  filter.clear();
  CHECK(read_block(tail) == zero);
  CUCO_CUDA_TRY(cudaMemcpy(other_tail, expected.data(), sizeof(expected), cudaMemcpyHostToDevice));
  CUCO_CUDA_TRY(cudaMemset(tail, 0, sizeof(expected)));
  filter.merge(other);
  CHECK(read_block(tail) == expected);
  CUCO_CUDA_TRY(cudaMemset(other_tail, 0, sizeof(expected)));
  CUCO_CUDA_TRY(cudaMemcpy(tail, expected.data(), sizeof(expected), cudaMemcpyHostToDevice));
  filter.intersect(other);
  CHECK(read_block(tail) == zero);

  // A size_t extent isolates the int loop-counter bug from the 32-bit word-count overflow.
  using wide_ref_type = cuco::bloom_filter_ref<std::uint64_t,
                                               cuco::extent<std::size_t>,
                                               cuda::thread_scope_device,
                                               policy_type>;
  auto wide_ref = wide_ref_type{data, cuco::extent<std::size_t>{num_blocks}, {}, policy_type{}};
  CUCO_CUDA_TRY(cudaMemcpy(tail, expected.data(), sizeof(expected), cudaMemcpyHostToDevice));
  clear_large_filter<<<1, 256>>>(wide_ref);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  CHECK(read_block(tail) == zero);

  CUCO_CUDA_TRY(cudaMemcpy(tail, expected.data(), sizeof(expected), cudaMemcpyHostToDevice));
  clear_large_filter<<<1, 256>>>(filter.ref());
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  CHECK(read_block(tail) == zero);
}
