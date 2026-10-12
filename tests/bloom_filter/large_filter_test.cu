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
  // The last block starts at word 2^32, which wraps to the first block in uint32_t.
  constexpr std::uint32_t num_blocks = (std::size_t{1} << 32) / words_per_block + 1;
  constexpr std::size_t num_words    = std::size_t{num_blocks} * words_per_block;
  constexpr std::size_t bytes        = num_words * sizeof(word_type);
  constexpr std::uint64_t key =
    (std::uint64_t{std::numeric_limits<std::uint32_t>::max()} << 32) | 42;

  std::size_t free_bytes, total_bytes;
  CUCO_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (free_bytes < 2 * bytes) { SKIP("Insufficient device memory for two large filters"); }

  auto filter      = filter_type{num_blocks};
  auto other       = filter_type{num_blocks};
  auto* data       = const_cast<word_type*>(filter.data());
  auto* tail       = data + num_words - words_per_block;
  auto* other_tail = const_cast<word_type*>(other.data()) + num_words - words_per_block;
  thrust::device_vector<std::uint64_t> keys(1, key);
  thrust::device_vector<bool> contained(1, false);

  std::array<word_type, words_per_block> zero{}, full;
  full.fill(~word_type{0});
  auto read_block = [](word_type const* ptr) {
    std::array<word_type, words_per_block> result;
    CUCO_CUDA_TRY(cudaMemcpy(result.data(), ptr, sizeof(result), cudaMemcpyDeviceToHost));
    return result;
  };

  CUCO_CUDA_TRY(cudaMemset(data, 0, sizeof(zero)));
  CUCO_CUDA_TRY(cudaMemset(tail, 0, sizeof(zero)));
  filter.add(keys.begin(), keys.end());
  CHECK(read_block(tail) != zero);
  CHECK(read_block(data) == zero);

  CUCO_CUDA_TRY(cudaMemset(data, 0, sizeof(zero)));
  CUCO_CUDA_TRY(cudaMemset(tail, 0xff, sizeof(full)));
  filter.contains(keys.begin(), keys.end(), contained.begin());
  CHECK(static_cast<bool>(contained[0]));

  filter.clear();
  CHECK(read_block(tail) == zero);
  CUCO_CUDA_TRY(cudaMemset(other_tail, 0xff, sizeof(full)));
  CUCO_CUDA_TRY(cudaMemset(tail, 0, sizeof(zero)));
  filter.merge(other);
  CHECK(read_block(tail) == full);
  CUCO_CUDA_TRY(cudaMemset(other_tail, 0, sizeof(zero)));
  CUCO_CUDA_TRY(cudaMemset(tail, 0xff, sizeof(full)));
  filter.intersect(other);
  CHECK(read_block(tail) == zero);

  CUCO_CUDA_TRY(cudaMemset(tail, 0xff, sizeof(full)));
  clear_large_filter<<<1, 256>>>(filter.ref());
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  CHECK(read_block(tail) == zero);

  // A wide extent isolates the int loop counter from the word-count overflow.
  using wide_ref_type = cuco::bloom_filter_ref<std::uint64_t,
                                               cuco::extent<std::size_t>,
                                               cuda::thread_scope_device,
                                               policy_type>;
  auto wide_ref = wide_ref_type{data, cuco::extent<std::size_t>{num_blocks}, {}, policy_type{}};
  CUCO_CUDA_TRY(cudaMemset(tail, 0xff, sizeof(full)));
  clear_large_filter<<<1, 256>>>(wide_ref);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  CHECK(read_block(tail) == zero);
}
