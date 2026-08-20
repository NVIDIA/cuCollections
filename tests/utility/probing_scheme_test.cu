/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/utility/cuda.hpp>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/probing_scheme.cuh>

#include <cuda/std/functional>
#include <cuda/std/limits>
#include <thrust/device_vector.h>

#include <cooperative_groups.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>

struct constant_hash {
  cuda::std::uint32_t value;

  __host__ __device__ constexpr constant_hash(cuda::std::uint32_t value = 0) noexcept : value{value}
  {
  }

  __host__ __device__ constexpr cuda::std::uint32_t operator()(cuda::std::int32_t) const noexcept
  {
    return value;
  }
};

template <typename T>
struct constexpr_extent {
  using value_type = T;

  __host__ __device__ constexpr operator value_type() const noexcept { return value; }

  friend __host__ __device__ constexpr value_type operator-(constexpr_extent<T> lhs,
                                                            value_type rhs) noexcept
  {
    return lhs.value - rhs;
  }

  friend __host__ __device__ constexpr value_type operator%(value_type lhs,
                                                            constexpr_extent<T> rhs) noexcept
  {
    return lhs % rhs.value;
  }

  value_type value;
};

template <int32_t BucketSize, class ProbingScheme, class Key, class Extent, class OutputIt>
__global__ void generate_scalar_probing_sequence(
  ProbingScheme probing_scheme, Key key, Extent upper_bound, size_t seq_length, OutputIt out_seq)
{
  auto constexpr cg_size = ProbingScheme::cg_size;
  static_assert(cg_size == 1, "Invalid CG size");

  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid == 0) {
    auto iter = probing_scheme.template make_iterator<BucketSize>(key, upper_bound);

    for (size_t i = 0; i < seq_length; ++i) {
      out_seq[i] = *iter;
      ++iter;
    }
  }
}

template <int32_t BucketSize, class ProbingScheme, class Key, class Extent, class OutputIt>
__global__ void generate_cg_probing_sequence(
  ProbingScheme probing_scheme, Key key, Extent upper_bound, size_t seq_length, OutputIt out_seq)
{
  auto constexpr cg_size = ProbingScheme::cg_size;

  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < cg_size) {
    auto const tile =
      cooperative_groups::tiled_partition<cg_size, cooperative_groups::thread_block>(
        cooperative_groups::this_thread_block());

    auto iter = probing_scheme.template make_iterator<BucketSize>(tile, key, upper_bound);

    for (size_t i = tile.thread_rank(); i < seq_length; ++i) {
      out_seq[i] = *iter;
      ++iter;
    }
  }
}

TEMPLATE_TEST_CASE_SIG(
  "utility probing_scheme tests",
  "",
  ((typename Key, cuco::test::probe_sequence Probe, int32_t BucketSize), Key, Probe, BucketSize),
  (int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, cuco::test::probe_sequence::linear_probing, 2))
{
  using probing_scheme_t = cuco::linear_probing<1, cuco::default_hash_function<int>>;
  auto const upper_bound = cuco::make_valid_extent<probing_scheme_t, cuco::storage<BucketSize>>(
    cuco::extent<std::size_t>{10});
  constexpr size_t seq_length{8};
  constexpr Key key{42};

  using probe = std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                                   cuco::linear_probing<1, cuco::default_hash_function<Key>>,
                                   cuco::double_hashing<1, cuco::default_hash_function<Key>>>;

  thrust::device_vector<size_t> scalar_seq(seq_length);
  generate_scalar_probing_sequence<BucketSize, probe>
    <<<1, 1>>>(probe{}, key, upper_bound, seq_length, scalar_seq.begin());
  thrust::device_vector<size_t> cg_seq(seq_length);
  generate_cg_probing_sequence<BucketSize, probe>
    <<<1, 1>>>(probe{}, key, upper_bound, seq_length, cg_seq.begin());

  REQUIRE(cuco::test::equal(
    scalar_seq.begin(), scalar_seq.end(), cg_seq.begin(), cuda::std::equal_to<std::size_t>{}));
}

template <typename Probe>
void check_scalar_sequence(Probe probe, cuda::std::int32_t requested_capacity)
{
  constexpr std::size_t sequence_length = 8;
  auto const upper_bound                = cuco::make_valid_extent<Probe, cuco::storage<1>>(
    cuco::extent<cuda::std::int32_t>{requested_capacity});
  auto const capacity = upper_bound.value();

  thrust::device_vector<cuda::std::int32_t> sequence(sequence_length);
  generate_scalar_probing_sequence<1>
    <<<1, 1>>>(probe, cuda::std::int32_t{7}, upper_bound, sequence_length, sequence.begin());

  REQUIRE(cuco::test::all_of(sequence.begin(), sequence.end(), [capacity] __device__(auto index) {
    return index >= 0 and index < capacity;
  }));
}

template <typename Probe>
void check_cg_sequence(Probe probe, cuda::std::int32_t requested_capacity)
{
  constexpr std::size_t sequence_length = 8;
  auto const upper_bound                = cuco::make_valid_extent<Probe, cuco::storage<1>>(
    cuco::extent<cuda::std::int32_t>{requested_capacity});
  auto const capacity = upper_bound.value();

  thrust::device_vector<cuda::std::int32_t> sequence(sequence_length);
  generate_cg_probing_sequence<1><<<1, Probe::cg_size>>>(
    probe, cuda::std::int32_t{7}, upper_bound, sequence_length, sequence.begin());

  REQUIRE(cuco::test::all_of(sequence.begin(), sequence.end(), [capacity] __device__(auto index) {
    return index >= 0 and index < capacity;
  }));
}

TEST_CASE("Probing schemes support the full unsigned hash range", "")
{
  constexpr auto high_bit = cuda::std::uint32_t{0x80000000};

  SECTION("Scalar linear probing")
  {
    check_scalar_sequence(cuco::linear_probing<1, constant_hash>{constant_hash{high_bit}}, 10);
  }

  SECTION("Cooperative linear probing")
  {
    check_cg_sequence(cuco::linear_probing<2, constant_hash>{constant_hash{high_bit}}, 10);
  }

  SECTION("Scalar double hashing primary hash")
  {
    check_scalar_sequence(
      cuco::double_hashing<1, constant_hash>{constant_hash{high_bit}, constant_hash{0}}, 10);
  }

  SECTION("Scalar double hashing secondary hash")
  {
    check_scalar_sequence(
      cuco::double_hashing<1, constant_hash>{constant_hash{0}, constant_hash{high_bit}}, 10);
  }

  SECTION("Cooperative double hashing primary hash")
  {
    check_cg_sequence(
      cuco::double_hashing<2, constant_hash>{constant_hash{high_bit}, constant_hash{0}}, 11);
  }

  SECTION("Cooperative double hashing secondary hash")
  {
    check_cg_sequence(
      cuco::double_hashing<2, constant_hash>{constant_hash{0}, constant_hash{high_bit}}, 11);
  }
}

TEST_CASE("Probing iterator wraps without overflowing its signed size type", "")
{
  constexpr auto max = cuda::std::numeric_limits<cuda::std::int32_t>::max();

  constexpr auto wrapped_index = [max] {
    cuco::detail::probing_iterator<constexpr_extent<cuda::std::int32_t>> iterator{
      max - 2, max - 3, constexpr_extent<cuda::std::int32_t>{max}};
    ++iterator;
    return *iterator;
  }();

  STATIC_REQUIRE(wrapped_index == max - 5);
}

TEST_CASE("Probing iterator wraps without overflowing its unsigned size type", "")
{
  constexpr auto max = cuda::std::numeric_limits<cuda::std::uint32_t>::max();

  constexpr auto wrapped_index = [max] {
    cuco::detail::probing_iterator<constexpr_extent<cuda::std::uint32_t>> iterator{
      max - 2, max - 3, constexpr_extent<cuda::std::uint32_t>{max}};
    ++iterator;
    return *iterator;
  }();

  STATIC_REQUIRE(wrapped_index == max - 5);
}
