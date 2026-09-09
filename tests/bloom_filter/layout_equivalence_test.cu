/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Byte-equal bitsets across (AddH, AddV) layout permutations, identical contains results
// across (ContainsH, ContainsV) layout permutations, equivalence between dynamic vs static
// `cuco::extent`, and invariance under the ConditionalAdd / EarlyExitContains policy knobs
// (both are optimizations that must not change results) -- all for fixed
// (Hash, WordBytes, WordsPerBlock, PatternBits, keys).

#include <test_utils.hpp>

#include <cuco/bloom_filter.cuh>
#include <cuco/detail/error.hpp>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

TEST_CASE("bloom_filter: word byte selection", "")
{
  using hash_type      = cuco::xxhash_64<int32_t>;
  using default_policy = cuco::bloom_filter_policy<int32_t>;
  using wide_policy    = cuco::bloom_filter_policy<int32_t, hash_type, 8>;
  using legacy_default_policy =
    cuco::parametric_filter_policy<hash_type, std::uint32_t, 8, 8, 8, 1, 1, 8, false, false>;
  using legacy_wide_policy =
    cuco::parametric_filter_policy<hash_type, std::uint64_t, 4, 4, 4, 1, 1, 4, false, false>;

  STATIC_REQUIRE((std::is_same_v<typename default_policy::word_type, unsigned int>));
  STATIC_REQUIRE((std::is_same_v<typename wide_policy::word_type, unsigned long long int>));
  STATIC_REQUIRE((std::is_same_v<legacy_default_policy, default_policy>));
  STATIC_REQUIRE((std::is_same_v<legacy_wide_policy, wide_policy>));
  STATIC_REQUIRE(default_policy::word_bytes == 4);
  STATIC_REQUIRE(wide_policy::word_bytes == 8);
  STATIC_REQUIRE(default_policy::words_per_block == 8);
  STATIC_REQUIRE(wide_policy::words_per_block == 4);
}

TEMPLATE_TEST_CASE_SIG(
  "bloom_filter: bitset is invariant under (AddH, AddV) layout permutations",
  "",
  ((class AltPolicy), AltPolicy),
  (cuco::bloom_filter_policy<int32_t, cuco::xxhash_64<int32_t>, 4, 8, 8, 1, 8, 1, 8>),
  (cuco::bloom_filter_policy<int32_t, cuco::xxhash_64<int32_t>, 4, 8, 8, 2, 4, 1, 8>),
  (cuco::bloom_filter_policy<int32_t, cuco::xxhash_64<int32_t>, 4, 8, 8, 4, 2, 1, 8>),
  (cuco::bloom_filter_policy<int32_t, cuco::xxhash_64<int32_t>, 4, 8, 8, 2, 2, 1, 8>),
  (cuco::bloom_filter_policy<int32_t, cuco::xxhash_64<int32_t>, 4, 8, 8, 4, 1, 1, 8>))
{
  using Key            = int32_t;
  using default_policy = cuco::bloom_filter_policy<Key>;
  using filter_default_t =
    cuco::bloom_filter<Key, cuco::extent<std::size_t>, cuda::thread_scope_device, default_policy>;
  using filter_alt_t =
    cuco::bloom_filter<Key, cuco::extent<std::size_t>, cuda::thread_scope_device, AltPolicy>;

  constexpr int32_t num_blocks = 1'000;
  constexpr int32_t num_keys   = 400;

  auto filter_default = filter_default_t{num_blocks};
  auto filter_alt     = filter_alt_t{num_blocks};

  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end());

  filter_default.add(keys.begin(), keys.end());
  filter_alt.add(keys.begin(), keys.end());

  auto const total_words =
    static_cast<std::size_t>(filter_default.block_extent()) * filter_default_t::words_per_block;
  REQUIRE(thrust::equal(
    thrust::device, filter_default.data(), filter_default.data() + total_words, filter_alt.data()));
}

TEMPLATE_TEST_CASE_SIG(
  "bloom_filter: contains results are invariant under (ContainsH, ContainsV) permutations",
  "",
  ((class AltPolicy), AltPolicy),
  (cuco::bloom_filter_policy<int32_t, cuco::xxhash_64<int32_t>, 4, 8, 8, 8, 1, 8, 1>),
  (cuco::bloom_filter_policy<int32_t, cuco::xxhash_64<int32_t>, 4, 8, 8, 8, 1, 2, 4>),
  (cuco::bloom_filter_policy<int32_t, cuco::xxhash_64<int32_t>, 4, 8, 8, 8, 1, 4, 2>),
  (cuco::bloom_filter_policy<int32_t, cuco::xxhash_64<int32_t>, 4, 8, 8, 8, 1, 2, 2>),
  (cuco::bloom_filter_policy<int32_t, cuco::xxhash_64<int32_t>, 4, 8, 8, 8, 1, 1, 4>))
{
  using Key            = int32_t;
  using default_policy = cuco::bloom_filter_policy<Key>;
  using filter_default_t =
    cuco::bloom_filter<Key, cuco::extent<std::size_t>, cuda::thread_scope_device, default_policy>;
  using filter_alt_t =
    cuco::bloom_filter<Key, cuco::extent<std::size_t>, cuda::thread_scope_device, AltPolicy>;

  constexpr int32_t num_blocks = 1'000;
  constexpr int32_t num_keys   = 400;
  constexpr int32_t num_probe  = 800;  // mix of inserted and disjoint

  auto filter_default = filter_default_t{num_blocks};
  auto filter_alt     = filter_alt_t{num_blocks};

  thrust::device_vector<Key> insert_keys(num_keys);
  thrust::sequence(thrust::device, insert_keys.begin(), insert_keys.end());
  filter_default.add(insert_keys.begin(), insert_keys.end());
  filter_alt.add(insert_keys.begin(), insert_keys.end());

  thrust::device_vector<Key> probe_keys(num_probe);
  thrust::sequence(thrust::device, probe_keys.begin(), probe_keys.end());

  thrust::device_vector<bool> result_default(num_probe);
  thrust::device_vector<bool> result_alt(num_probe);
  filter_default.contains(probe_keys.begin(), probe_keys.end(), result_default.begin());
  filter_alt.contains(probe_keys.begin(), probe_keys.end(), result_alt.begin());

  REQUIRE(thrust::equal(
    thrust::device, result_default.begin(), result_default.end(), result_alt.begin()));
}

TEST_CASE("bloom_filter: bitset is invariant under dynamic vs static cuco::extent", "")
{
  using Key                        = int32_t;
  using Policy                     = cuco::bloom_filter_policy<Key>;
  constexpr std::size_t num_blocks = 1'000;
  constexpr int32_t num_keys       = 400;

  using dynamic_extent_t = cuco::extent<std::size_t>;
  using static_extent_t  = cuco::extent<std::size_t, num_blocks>;
  using filter_dynamic_t =
    cuco::bloom_filter<Key, dynamic_extent_t, cuda::thread_scope_device, Policy>;
  using filter_static_t =
    cuco::bloom_filter<Key, static_extent_t, cuda::thread_scope_device, Policy>;

  auto filter_dynamic = filter_dynamic_t{num_blocks};
  auto filter_static  = filter_static_t{static_extent_t{}};

  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end());

  filter_dynamic.add(keys.begin(), keys.end());
  filter_static.add(keys.begin(), keys.end());

  auto const total_words = num_blocks * filter_dynamic_t::words_per_block;
  REQUIRE(thrust::equal(thrust::device,
                        filter_dynamic.data(),
                        filter_dynamic.data() + total_words,
                        filter_static.data()));
}

TEST_CASE("bloom_filter: bitset is invariant under ConditionalAdd", "")
{
  using Key = int32_t;
  // Same layout, ConditionalAdd off vs on. The read-before-atomic skip must yield the same bits.
  using filter_off_t = cuco::bloom_filter<Key,
                                          cuco::extent<std::size_t>,
                                          cuda::thread_scope_device,
                                          cuco::bloom_filter_policy<Key>>;
  using filter_on_t  = cuco::bloom_filter<
     Key,
     cuco::extent<std::size_t>,
     cuda::thread_scope_device,
     cuco::bloom_filter_policy<Key, cuco::xxhash_64<Key>, 4, 8, 8, 8, 1, 1, 8, true>>;

  constexpr int32_t num_blocks = 1'000;
  constexpr int32_t num_keys   = 400;

  auto filter_off = filter_off_t{num_blocks};
  auto filter_on  = filter_on_t{num_blocks};

  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end());

  // Add twice so the second pass hits already-set words, exercising the ConditionalAdd skip.
  filter_off.add(keys.begin(), keys.end());
  filter_off.add(keys.begin(), keys.end());
  filter_on.add(keys.begin(), keys.end());
  filter_on.add(keys.begin(), keys.end());

  auto const total_words =
    static_cast<std::size_t>(filter_off.block_extent()) * filter_off_t::words_per_block;
  REQUIRE(thrust::equal(
    thrust::device, filter_off.data(), filter_off.data() + total_words, filter_on.data()));
}

TEST_CASE("bloom_filter: contains results are invariant under EarlyExitContains", "")
{
  using Key = int32_t;
  // ContainsHorizontalLayout > 1 so the compare_patterns early-exit branch is actually used.
  using filter_off_t =
    cuco::bloom_filter<Key,
                       cuco::extent<std::size_t>,
                       cuda::thread_scope_device,
                       cuco::bloom_filter_policy<Key, cuco::xxhash_64<Key>, 4, 8, 8, 8, 1, 8, 1>>;
  using filter_on_t = cuco::bloom_filter<
    Key,
    cuco::extent<std::size_t>,
    cuda::thread_scope_device,
    cuco::bloom_filter_policy<Key, cuco::xxhash_64<Key>, 4, 8, 8, 8, 1, 8, 1, false, true>>;

  constexpr int32_t num_blocks = 1'000;
  constexpr int32_t num_keys   = 400;
  constexpr int32_t num_probe = 800;  // mix of inserted and disjoint, so misses fire the early exit

  auto filter_off = filter_off_t{num_blocks};
  auto filter_on  = filter_on_t{num_blocks};

  thrust::device_vector<Key> insert_keys(num_keys);
  thrust::sequence(thrust::device, insert_keys.begin(), insert_keys.end());
  filter_off.add(insert_keys.begin(), insert_keys.end());
  filter_on.add(insert_keys.begin(), insert_keys.end());

  thrust::device_vector<Key> probe_keys(num_probe);
  thrust::sequence(thrust::device, probe_keys.begin(), probe_keys.end());

  thrust::device_vector<bool> result_off(num_probe);
  thrust::device_vector<bool> result_on(num_probe);
  filter_off.contains(probe_keys.begin(), probe_keys.end(), result_off.begin());
  filter_on.contains(probe_keys.begin(), probe_keys.end(), result_on.begin());

  REQUIRE(thrust::equal(thrust::device, result_off.begin(), result_off.end(), result_on.begin()));
}
