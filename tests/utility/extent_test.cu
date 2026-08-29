/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/storage.cuh>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>

auto constexpr cg_size     = 2;
auto constexpr bucket_size = 4;

using storage_t = cuco::storage<bucket_size>;
template <typename H1, typename H2>
using probing_t = cuco::double_hashing<cg_size, H1, H2>;

TEMPLATE_TEST_CASE_SIG(
  "utility extent tests", "", ((typename SizeType), SizeType), (int32_t), (int64_t), (std::size_t))
{
  SizeType constexpr num            = 1234;
  SizeType constexpr gold_reference = 1256;  // 157 x 2 x 4

  SECTION("Static extent must be evaluated at compile time.")
  {
    auto const size = cuco::extent<SizeType, num>{};
    STATIC_REQUIRE(num == size);
  }

  SECTION("Dynamic extent is evaluated at run time.")
  {
    auto const size = cuco::extent(num);
    REQUIRE(size == num);
  }

  SECTION("Compute static valid extent at compile time.")
  {
    auto constexpr size = cuco::extent<SizeType, num>{};
    auto constexpr res  = cuco::make_valid_extent<probing_t, storage_t>(size);
    STATIC_REQUIRE(gold_reference == res.value());
  }

  SECTION("Compute dynamic valid extent at run time.")
  {
    auto const size = cuco::extent<SizeType>{num};
    auto const res  = cuco::make_valid_extent<probing_t, storage_t>(size);
    REQUIRE(gold_reference == res.value());
  }

  SECTION("Invalid desired load factor throws exception")
  {
    using probing_scheme_type = cuco::linear_probing<cg_size, cuco::default_hash_function<int>>;
    using storage_type        = cuco::storage<bucket_size>;

    auto const size = cuco::extent<SizeType>{num};

    // Test load factor <= 0
    REQUIRE_THROWS(cuco::make_valid_extent<probing_scheme_type, storage_type>(size, 0.0));
    REQUIRE_THROWS(cuco::make_valid_extent<probing_scheme_type, storage_type>(size, -0.5));

    // Test load factor > 1
    REQUIRE_THROWS(cuco::make_valid_extent<probing_scheme_type, storage_type>(size, 1.5));
  }
}

TEST_CASE("utility extent boundary tests", "")
{
  using hash_type = cuco::default_hash_function<std::int32_t>;
  using double_1  = cuco::double_hashing<1, hash_type>;
  using double_2  = cuco::double_hashing<2, hash_type>;
  using linear_1  = cuco::linear_probing<1, hash_type>;
  using linear_2  = cuco::linear_probing<2, hash_type>;
  using storage   = cuco::storage<1>;

  constexpr auto i32_max = std::numeric_limits<std::int32_t>::max();
  constexpr auto u32_max = std::numeric_limits<std::uint32_t>::max();
  constexpr auto u64_max = std::numeric_limits<std::uint64_t>::max();

  SECTION("Representable capacities are preserved")
  {
    auto const signed_double =
      cuco::make_valid_extent<double_1, storage>(cuco::extent<std::int32_t>{i32_max});
    auto const signed_linear =
      cuco::make_valid_extent<linear_1, storage>(cuco::extent<std::int32_t>{i32_max});
    auto const unsigned_linear_32 =
      cuco::make_valid_extent<linear_1, storage>(cuco::extent<std::uint32_t>{u32_max});
    auto const unsigned_linear_64 =
      cuco::make_valid_extent<linear_1, storage>(cuco::extent<std::uint64_t>{u64_max});

    REQUIRE(signed_double.value() == i32_max);
    REQUIRE(signed_linear.value() == i32_max);
    REQUIRE(unsigned_linear_32.value() == u32_max);
    REQUIRE(unsigned_linear_64.value() == u64_max);
  }

  SECTION("Unrepresentable rounding is rejected")
  {
    REQUIRE_THROWS(
      cuco::make_valid_extent<double_2, storage>(cuco::extent<std::int32_t>{i32_max - 1}));
    REQUIRE_THROWS(cuco::make_valid_extent<linear_2, storage>(cuco::extent<std::int32_t>{i32_max}));
    REQUIRE_THROWS(
      cuco::make_valid_extent<double_1, storage>(cuco::extent<std::uint32_t>{u32_max}));
    REQUIRE_THROWS(
      cuco::make_valid_extent<double_1, storage>(cuco::extent<std::uint64_t>{u64_max}));
  }

  SECTION("Zero and negative signed extents retain their existing behavior")
  {
    auto const double_zero =
      cuco::make_valid_extent<double_2, storage>(cuco::extent<std::int32_t>{0});
    auto const double_negative =
      cuco::make_valid_extent<double_2, storage>(cuco::extent<std::int32_t>{-10});
    auto const linear_zero =
      cuco::make_valid_extent<linear_2, storage>(cuco::extent<std::int32_t>{0});
    auto const linear_negative =
      cuco::make_valid_extent<linear_2, storage>(cuco::extent<std::int32_t>{-10});

    REQUIRE(double_zero.value() == 4);
    REQUIRE(double_negative.value() == 4);
    REQUIRE(linear_zero.value() == 4);
    REQUIRE(linear_negative.value() == 2);
  }

  SECTION("Load factor conversion is checked before narrowing")
  {
    auto const negative = cuco::make_valid_extent<double_1, storage>(
      cuco::extent<std::int32_t>{std::numeric_limits<std::int32_t>::min()}, 0.5);
    REQUIRE(negative.value() == 2);

    REQUIRE_THROWS(
      cuco::make_valid_extent<double_1, storage>(cuco::extent<std::int32_t>{i32_max}, 0.5));
    REQUIRE_THROWS(
      cuco::make_valid_extent<linear_1, storage>(cuco::extent<std::uint64_t>{u64_max}, 0.5));
  }
}
