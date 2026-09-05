/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/bloom_filter.cuh>
#include <cuco/utility/error.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

TEMPLATE_TEST_CASE_SIG("bloom_filter byte sizing",
                       "[bloom_filter][sizing][host]",
                       (class Policy),
                       (cuco::bloom_filter_policy<int32_t>),
                       (cuco::bloom_filter_policy<int32_t, cuco::xxhash_64<int32_t>, 8, 8>))
{
  using filter_type =
    cuco::bloom_filter<int32_t, cuco::extent<std::size_t>, cuda::thread_scope_device, Policy>;
  constexpr auto block_bytes = Policy::words_per_block * sizeof(typename Policy::word_type);
  constexpr auto max_bytes   = Policy::max_filter_blocks * block_bytes;

  STATIC_REQUIRE(filter_type::max_size() == max_bytes);
  STATIC_REQUIRE(filter_type::aligned_size(block_bytes) == block_bytes);
  REQUIRE(filter_type::aligned_size(2 * block_bytes - 1) == block_bytes);
  REQUIRE(filter_type::aligned_size(max_bytes) == max_bytes);
  REQUIRE(filter_type::aligned_size(std::numeric_limits<std::size_t>::max()) == max_bytes);
  REQUIRE_THROWS_AS(filter_type::aligned_size(block_bytes - 1), cuco::logic_error);

  // These must throw before device allocation, including on hosts without a GPU.
  for (auto bytes : {std::size_t{0}, block_bytes + 1, max_bytes + block_bytes}) {
    REQUIRE_THROWS_AS(filter_type{cuco::bloom_filter_size_bytes{bytes}}, cuco::logic_error);
  }
}

TEST_CASE("bloom_filter byte sizing extent limits", "[bloom_filter][sizing][host]")
{
  using filter_type = cuco::bloom_filter<int32_t, cuco::extent<uint8_t>>;
  constexpr auto block_bytes =
    filter_type::words_per_block * sizeof(typename filter_type::word_type);
  constexpr auto max_bytes = std::numeric_limits<uint8_t>::max() * block_bytes;

  STATIC_REQUIRE(filter_type::max_size() == max_bytes);
  REQUIRE_THROWS_AS(filter_type{cuco::bloom_filter_size_bytes{max_bytes + block_bytes}},
                    cuco::logic_error);

  using static_filter = cuco::bloom_filter<int32_t, cuco::extent<std::size_t, 2>>;
  REQUIRE_THROWS_AS(static_filter{cuco::bloom_filter_size_bytes{block_bytes}}, cuco::logic_error);
}

TEMPLATE_TEST_CASE_SIG("bloom_filter byte construction",
                       "[bloom_filter][sizing][gpu]",
                       (class Extent),
                       (cuco::extent<std::size_t>),
                       (cuco::extent<uint8_t, 2>))
{
  using filter_type    = cuco::bloom_filter<int32_t, Extent>;
  constexpr auto bytes = 2 * filter_type::words_per_block * sizeof(typename filter_type::word_type);

  auto by_bytes  = filter_type{cuco::bloom_filter_size_bytes{bytes}};
  auto by_blocks = filter_type{Extent{2}};
  REQUIRE(static_cast<std::size_t>(by_bytes.block_extent()) == 2);
  REQUIRE(by_bytes.block_extent() == by_blocks.block_extent());
}
