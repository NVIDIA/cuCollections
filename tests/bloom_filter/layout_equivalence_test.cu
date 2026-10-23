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

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

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

namespace {
template <uint32_t Words,
          uint32_t PatternBits,
          uint32_t AddH,
          uint32_t AddV,
          uint32_t ContainsH,
          uint32_t ContainsV,
          bool FastL2Add = false>
using mixed_policy = cuco::bloom_filter_policy<int32_t,
                                               cuco::xxhash_64<int32_t>,
                                               4,
                                               Words,
                                               PatternBits,
                                               AddH,
                                               AddV,
                                               ContainsH,
                                               ContainsV,
                                               false,
                                               false,
                                               false,
                                               FastL2Add>;
}  // namespace

// Reference: one thread per key for add and contains. Alternatives mix cooperative add and
// contains layouts, several bits per word, uneven pattern bits and 64-byte blocks.
using mixed_layout_cases = std::tuple<
  std::pair<mixed_policy<8, 8, 1, 8, 1, 8>, mixed_policy<8, 8, 8, 1, 2, 4>>,
  std::pair<mixed_policy<8, 8, 1, 8, 1, 8>, mixed_policy<8, 8, 2, 4, 4, 2>>,
  std::pair<mixed_policy<8, 8, 1, 8, 1, 8>, mixed_policy<8, 8, 4, 1, 8, 1>>,
  std::pair<mixed_policy<8, 16, 1, 8, 1, 8>, mixed_policy<8, 16, 4, 2, 2, 2>>,
  std::pair<mixed_policy<8, 16, 1, 8, 1, 8>, mixed_policy<8, 16, 8, 1, 4, 1>>,
  std::pair<mixed_policy<8, 12, 1, 8, 1, 8>, mixed_policy<8, 12, 8, 1, 2, 4>>,
  std::pair<mixed_policy<8, 12, 1, 8, 1, 8>, mixed_policy<8, 12, 2, 2, 4, 2>>,
  std::pair<mixed_policy<16, 16, 1, 16, 1, 16>, mixed_policy<16, 16, 16, 1, 4, 4>>,
  std::pair<mixed_policy<16, 16, 1, 16, 1, 16>, mixed_policy<16, 16, 8, 1, 2, 8>>,
  std::pair<mixed_policy<16, 32, 1, 16, 1, 16>, mixed_policy<16, 32, 16, 1, 2, 8>>,
  std::pair<mixed_policy<8, 8, 1, 8, 1, 8, true>, mixed_policy<8, 8, 4, 2, 4, 2, true>>,
  std::pair<mixed_policy<16, 16, 1, 16, 1, 16, true>, mixed_policy<16, 16, 16, 1, 4, 4, true>>>;

TEMPLATE_LIST_TEST_CASE("bloom_filter: bitset and contains are invariant under mixed layouts",
                        "",
                        mixed_layout_cases)
{
  using Key          = int32_t;
  using filter_ref_t = cuco::bloom_filter<Key,
                                          cuco::extent<std::size_t>,
                                          cuda::thread_scope_device,
                                          typename TestType::first_type>;
  using filter_alt_t = cuco::bloom_filter<Key,
                                          cuco::extent<std::size_t>,
                                          cuda::thread_scope_device,
                                          typename TestType::second_type>;

  constexpr int32_t num_blocks = 1'000;
  constexpr int32_t num_keys   = 2'000;
  constexpr int32_t num_probe  = 4'000;  // inserted keys first, then disjoint ones

  auto filter_ref = filter_ref_t{num_blocks};
  auto filter_alt = filter_alt_t{num_blocks};

  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end());
  filter_ref.add(keys.begin(), keys.end());
  filter_alt.add(keys.begin(), keys.end());

  auto const total_words =
    static_cast<std::size_t>(filter_ref.block_extent()) * filter_ref_t::words_per_block;
  REQUIRE(thrust::equal(
    thrust::device, filter_ref.data(), filter_ref.data() + total_words, filter_alt.data()));

  thrust::device_vector<Key> probe(num_probe);
  thrust::sequence(thrust::device, probe.begin(), probe.end());
  thrust::device_vector<bool> result_ref(num_probe), result_alt(num_probe);
  filter_ref.contains(probe.begin(), probe.end(), result_ref.begin());
  filter_alt.contains(probe.begin(), probe.end(), result_alt.begin());
  REQUIRE(thrust::equal(thrust::device, result_ref.begin(), result_ref.end(), result_alt.begin()));
  REQUIRE(thrust::all_of(
    thrust::device, result_alt.begin(), result_alt.begin() + num_keys, [] __device__(bool b) {
      return b;
    }));
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

namespace {
template <std::uint32_t AddH, std::uint32_t AddV, std::uint32_t ContainsH, std::uint32_t ContainsV>
using skip_policy = cuco::bloom_filter_policy<int32_t,
                                              cuco::xxhash_64<int32_t>,
                                              4,
                                              8,
                                              8,
                                              AddH,
                                              AddV,
                                              ContainsH,
                                              ContainsV,
                                              false,
                                              false,
                                              false,
                                              true>;
}  // namespace

TEST_CASE("bloom_filter: fast_l2_add leaves exactly one word of a key's block untouched", "")
{
  using Key      = int32_t;
  using filter_t = cuco::bloom_filter<Key,
                                      cuco::extent<std::size_t>,
                                      cuda::thread_scope_device,
                                      skip_policy<8, 1, 1, 8>>;
  STATIC_REQUIRE(skip_policy<8, 1, 1, 8>::fast_l2_add);
  STATIC_REQUIRE(not cuco::bloom_filter_policy<Key>::fast_l2_add);

  constexpr int32_t num_blocks = 1;  // every key lands in block 0
  for (Key k : {0, 1, 2, 3, 12345, -7}) {
    auto filter = filter_t{num_blocks};
    thrust::device_vector<Key> key(1, k);
    filter.add(key.begin(), key.end());
    auto const nonzero = thrust::count_if(thrust::device,
                                          filter.data(),
                                          filter.data() + filter_t::words_per_block,
                                          [] __device__(auto w) { return w != 0; });
    // one bit per word (pattern_bits == words_per_block), one word skipped
    REQUIRE(nonzero == filter_t::words_per_block - 1);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "bloom_filter: fast_l2_add bitset and contains are invariant under layout permutations",
  "",
  ((class AltPolicy), AltPolicy),
  (skip_policy<1, 8, 1, 8>),
  (skip_policy<2, 4, 1, 8>),
  (skip_policy<4, 2, 2, 4>),
  (skip_policy<4, 1, 4, 2>),
  (skip_policy<8, 1, 8, 1>))
{
  using Key          = int32_t;
  using filter_ref_t = cuco::bloom_filter<Key,
                                          cuco::extent<std::size_t>,
                                          cuda::thread_scope_device,
                                          skip_policy<8, 1, 1, 8>>;
  using filter_alt_t =
    cuco::bloom_filter<Key, cuco::extent<std::size_t>, cuda::thread_scope_device, AltPolicy>;

  constexpr int32_t num_blocks = 1'000;
  constexpr int32_t num_keys   = 400;
  constexpr int32_t num_probe  = 800;  // mix of inserted and disjoint

  auto filter_ref = filter_ref_t{num_blocks};
  auto filter_alt = filter_alt_t{num_blocks};

  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end());
  filter_ref.add(keys.begin(), keys.end());
  filter_alt.add(keys.begin(), keys.end());

  auto const total_words =
    static_cast<std::size_t>(filter_ref.block_extent()) * filter_ref_t::words_per_block;
  REQUIRE(thrust::equal(
    thrust::device, filter_ref.data(), filter_ref.data() + total_words, filter_alt.data()));

  thrust::device_vector<Key> probe(num_probe);
  thrust::sequence(thrust::device, probe.begin(), probe.end());
  thrust::device_vector<bool> result_ref(num_probe), result_alt(num_probe);
  filter_ref.contains(probe.begin(), probe.end(), result_ref.begin());
  filter_alt.contains(probe.begin(), probe.end(), result_alt.begin());
  REQUIRE(thrust::equal(thrust::device, result_ref.begin(), result_ref.end(), result_alt.begin()));
  // no false negatives among the inserted keys
  REQUIRE(thrust::all_of(
    thrust::device, result_alt.begin(), result_alt.begin() + num_keys, [] __device__(bool b) {
      return b;
    }));
}

TEMPLATE_TEST_CASE_SIG(
  "bloom_filter: fast_l2_add on multi-sector blocks skips one word of the first sector",
  "",
  ((class Policy, int SkippedLimit), Policy, SkippedLimit),
  ((cuco::bloom_filter_policy<int32_t,
                              cuco::xxhash_64<int32_t>,
                              8,
                              8,
                              8,
                              8,
                              1,
                              2,
                              4,
                              false,
                              false,
                              false,
                              true>),
   4),
  ((cuco::bloom_filter_policy<int32_t,
                              cuco::xxhash_64<int32_t>,
                              8,
                              8,
                              8,
                              4,
                              2,
                              1,
                              8,
                              false,
                              false,
                              false,
                              true>),
   4),
  ((cuco::bloom_filter_policy<int32_t,
                              cuco::xxhash_64<int32_t>,
                              8,
                              16,
                              16,
                              16,
                              1,
                              4,
                              4,
                              false,
                              false,
                              false,
                              true>),
   4),
  ((cuco::bloom_filter_policy<int32_t,
                              cuco::xxhash_64<int32_t>,
                              4,
                              16,
                              16,
                              16,
                              1,
                              2,
                              8,
                              false,
                              false,
                              false,
                              true>),
   8),
  ((cuco::bloom_filter_policy<int32_t,
                              cuco::xxhash_64<int32_t>,
                              4,
                              16,
                              16,
                              1,
                              16,
                              1,
                              16,
                              false,
                              false,
                              false,
                              true>),
   8),
  ((cuco::bloom_filter_policy<int32_t,
                              cuco::xxhash_64<int32_t>,
                              8,
                              8,
                              8,
                              1,
                              8,
                              4,
                              2,
                              false,
                              false,
                              false,
                              true>),
   4))
{
  using Key = int32_t;
  using filter_t =
    cuco::bloom_filter<Key, cuco::extent<std::size_t>, cuda::thread_scope_device, Policy>;
  constexpr auto words = filter_t::words_per_block;

  for (Key k : {0, 1, 2, 3, 12345, -7}) {
    auto filter = filter_t{1};  // every key lands in block 0
    thrust::device_vector<Key> key(1, k);
    filter.add(key.begin(), key.end());
    thrust::device_vector<typename filter_t::word_type> block(filter.data(), filter.data() + words);
    thrust::host_vector<typename filter_t::word_type> h = block;
    int zeros = 0, zero_index = -1;
    for (int i = 0; i < static_cast<int>(words); ++i) {
      if (h[i] == 0) { ++zeros, zero_index = i; }
    }
    // one bit per word (pattern_bits == words_per_block): exactly one untouched word, in sector 0
    REQUIRE(zeros == 1);
    REQUIRE(zero_index < SkippedLimit);

    thrust::device_vector<bool> found(1);
    filter.contains(key.begin(), key.end(), found.begin());
    REQUIRE(found[0]);
  }
}

template <uint32_t Words, uint32_t AddH, uint32_t AddV>
using wide_skip_policy = cuco::bloom_filter_policy<int32_t,
                                                   cuco::xxhash_64<int32_t>,
                                                   4,
                                                   Words,
                                                   Words,
                                                   AddH,
                                                   AddV,
                                                   1,
                                                   Words,
                                                   false,
                                                   false,
                                                   false,
                                                   true>;

TEMPLATE_TEST_CASE_SIG(
  "bloom_filter: fast_l2_add cooperative bulk add matches the per-key layout on many keys",
  "",
  ((class CoopPolicy, class RefPolicy), CoopPolicy, RefPolicy),
  (skip_policy<8, 1, 1, 8>, skip_policy<1, 8, 1, 8>),
  (wide_skip_policy<16, 16, 1>, wide_skip_policy<16, 1, 16>))
{
  using Key = int32_t;
  // the horizontal layouts run the cooperative (shuffle-first) add; 1xN is non-cooperative
  using filter_coop_t =
    cuco::bloom_filter<Key, cuco::extent<std::size_t>, cuda::thread_scope_device, CoopPolicy>;
  using filter_ref_t =
    cuco::bloom_filter<Key, cuco::extent<std::size_t>, cuda::thread_scope_device, RefPolicy>;

  constexpr std::size_t num_blocks = 1 << 16;
  constexpr int32_t num_keys       = 3'000'017;  // many blocks and a partial last block

  auto filter_coop = filter_coop_t{num_blocks};
  auto filter_ref  = filter_ref_t{num_blocks};
  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end());
  filter_coop.add(keys.begin(), keys.end());
  filter_ref.add(keys.begin(), keys.end());

  auto const total_words = num_blocks * filter_coop_t::words_per_block;
  REQUIRE(thrust::equal(
    thrust::device, filter_coop.data(), filter_coop.data() + total_words, filter_ref.data()));
}

TEST_CASE("bloom_filter: fast_l2_add has no effect on blocks smaller than one sector", "")
{
  using Key = int32_t;
  // 16-byte blocks: 4 x 4-byte words, cooperative add over 4 threads
  using off_policy =
    cuco::bloom_filter_policy<Key, cuco::xxhash_64<Key>, 4, 4, 4, 4, 1, 1, 4, false, false, false>;
  using on_policy = cuco::
    bloom_filter_policy<Key, cuco::xxhash_64<Key>, 4, 4, 4, 4, 1, 1, 4, false, false, false, true>;
  STATIC_REQUIRE(not on_policy::fast_l2_add);

  using filter_off_t =
    cuco::bloom_filter<Key, cuco::extent<std::size_t>, cuda::thread_scope_device, off_policy>;
  using filter_on_t =
    cuco::bloom_filter<Key, cuco::extent<std::size_t>, cuda::thread_scope_device, on_policy>;

  constexpr std::size_t num_blocks = 1'000;
  constexpr int32_t num_keys       = 400;

  auto filter_off = filter_off_t{num_blocks};
  auto filter_on  = filter_on_t{num_blocks};
  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end());
  filter_off.add(keys.begin(), keys.end());
  filter_on.add(keys.begin(), keys.end());

  auto const total_words = num_blocks * filter_off_t::words_per_block;
  REQUIRE(thrust::equal(
    thrust::device, filter_off.data(), filter_off.data() + total_words, filter_on.data()));
}
