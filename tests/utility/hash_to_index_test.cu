/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/utils.cuh>

#include <cuda/std/array>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <vector>

template <typename Hash, typename Size>
__global__ void hash_to_indices(Hash const* hashes, Size const* bounds, Size* output, int count)
{
  auto const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) { output[i] = cuco::detail::hash_to_index<Size>(hashes[i], bounds[i]); }
}

TEST_CASE("hash_to_index is constexpr and preserves unsigned hash bits", "")
{
  using cuco::detail::hash_to_index;
  STATIC_REQUIRE(hash_to_index<std::int32_t>(std::uint32_t{0x80000000}, 10) == 5);
  STATIC_REQUIRE(hash_to_index<std::int32_t>(std::int32_t{-1}, 10) == 9);
  STATIC_REQUIRE(hash_to_index<std::int64_t>(std::uint32_t{0x80000000}, 10) == 5);
  STATIC_REQUIRE(hash_to_index<std::uint64_t>(std::uint32_t{0x80000000}, 0x200000002ull) ==
                 0x100000001ull);
  STATIC_REQUIRE(hash_to_index<std::uint32_t>(std::uint64_t{0x8000000000000000ull}, 10) == 5);
  STATIC_REQUIRE(hash_to_index<std::uint64_t>(std::uint64_t{0xffffffffffffffffull},
                                              0xffffffffffffffffull) == 0xfffffffffffffffeull);
  STATIC_REQUIRE(hash_to_index<std::uint32_t>(std::uint8_t{128}, 10) == 5);
}

#if defined(CUCO_HAS_INT128)
TEMPLATE_TEST_CASE("hash_to_index narrows scalar 128-bit hashes to the index width",
                   "",
                   std::int32_t,
                   std::uint32_t,
                   std::int64_t,
                   std::uint64_t)
{
  constexpr auto hash =
    (static_cast<unsigned __int128>(0x123456789abcdef0ull) << 64) | 0x4000000080000000ull;
  constexpr TestType expected = sizeof(TestType) == 4 ? 5 : 2;
  STATIC_REQUIRE(cuco::detail::hash_to_index<TestType>(hash, 10) == expected);
  STATIC_REQUIRE(cuco::detail::hash_to_index<TestType>(static_cast<__int128>(-1), 10) == 9);
  thrust::device_vector<unsigned __int128> hashes(1, hash);
  thrust::device_vector<TestType> bounds(1, 10);
  thrust::device_vector<TestType> output(1);
  hash_to_indices<<<1, 1>>>(thrust::raw_pointer_cast(hashes.data()),
                            thrust::raw_pointer_cast(bounds.data()),
                            thrust::raw_pointer_cast(output.data()),
                            1);
  CUCO_CUDA_TRY(cudaGetLastError());
  REQUIRE(output[0] == expected);
}

TEMPLATE_TEST_CASE_SIG("hash_to_index matches a wide product on host and device",
                       "",
                       ((typename Hash, typename Size), Hash, Size),
                       (std::uint32_t, std::int32_t),
                       (std::uint32_t, std::uint32_t),
                       (std::uint32_t, std::int64_t),
                       (std::uint32_t, std::uint64_t),
                       (std::uint64_t, std::int32_t),
                       (std::uint64_t, std::uint32_t),
                       (std::uint64_t, std::int64_t),
                       (std::uint64_t, std::uint64_t),
                       (std::int32_t, std::int64_t),
                       (std::int64_t, std::int32_t))
{
  using unsigned_hash     = cuda::std::make_unsigned_t<Hash>;
  constexpr auto bits     = cuda::std::numeric_limits<unsigned_hash>::digits;
  constexpr auto max_hash = cuda::std::numeric_limits<unsigned_hash>::max();
  constexpr auto max_size = cuda::std::numeric_limits<Size>::max();
  std::vector<Hash> hashes;
  std::vector<Size> bounds;
  std::vector<Size> expected;
  auto append = [&](unsigned_hash hash, Size bound) {
    hashes.push_back(static_cast<Hash>(hash));
    bounds.push_back(bound);
    expected.push_back(static_cast<Size>((static_cast<unsigned __int128>(hash) * bound) >> bits));
  };
  for (auto hash : {unsigned_hash{0},
                    unsigned_hash{1},
                    unsigned_hash{max_hash / 2},
                    unsigned_hash{max_hash / 2 + 1},
                    unsigned_hash{max_hash - 1},
                    max_hash}) {
    for (auto bound : {Size{1},
                       Size{2},
                       Size{3},
                       Size{10},
                       Size{65537},
                       Size{max_size / 2},
                       Size{max_size - 1},
                       max_size}) {
      append(hash, bound);
    }
    if constexpr (sizeof(Size) == 8) {
      for (auto bound : {Size{0xffffffffull}, Size{0x100000000ull}, Size{0x100000001ull}}) {
        append(hash, bound);
      }
    }
  }
  std::mt19937_64 rng{24331};
  for (int i = 0; i < 4096; ++i) {
    append(static_cast<unsigned_hash>(rng()), static_cast<Size>(rng() % max_size + 1));
  }
  for (std::size_t i = 0; i < hashes.size(); ++i) {
    REQUIRE(cuco::detail::hash_to_index<Size>(hashes[i], bounds[i]) == expected[i]);
  }
  thrust::device_vector<Hash> d_hashes(hashes);
  thrust::device_vector<Size> d_bounds(bounds);
  thrust::device_vector<Size> d_output(hashes.size());
  hash_to_indices<<<(hashes.size() + 127) / 128, 128>>>(thrust::raw_pointer_cast(d_hashes.data()),
                                                        thrust::raw_pointer_cast(d_bounds.data()),
                                                        thrust::raw_pointer_cast(d_output.data()),
                                                        static_cast<int>(hashes.size()));
  CUCO_CUDA_TRY(cudaGetLastError());
  thrust::host_vector<Size> output(d_output);
  for (std::size_t i = 0; i < hashes.size(); ++i) {
    REQUIRE(output[i] == expected[i]);
  }
}
#endif

TEMPLATE_TEST_CASE("hash_to_index narrows two-word 128-bit hashes to the index width",
                   "",
                   std::int32_t,
                   std::uint32_t,
                   std::int64_t,
                   std::uint64_t)
{
  using hash_type = cuda::std::array<std::uint64_t, 2>;
  constexpr hash_type hash{0x4000000080000000ull, 0xffffffffffffffffull};
  constexpr TestType expected = sizeof(TestType) == 4 ? 5 : 2;
  STATIC_REQUIRE(cuco::detail::hash_to_index<TestType>(hash, 10) == expected);
  STATIC_REQUIRE(cuco::detail::hash_to_index<TestType>(hash_type{0, 0xffffffffffffffffull}, 10) ==
                 0);
  thrust::device_vector<hash_type> hashes(1, hash);
  thrust::device_vector<TestType> bounds(1, 10);
  thrust::device_vector<TestType> output(1);
  hash_to_indices<<<1, 1>>>(thrust::raw_pointer_cast(hashes.data()),
                            thrust::raw_pointer_cast(bounds.data()),
                            thrust::raw_pointer_cast(output.data()),
                            1);
  CUCO_CUDA_TRY(cudaGetLastError());
  REQUIRE(output[0] == expected);
}
