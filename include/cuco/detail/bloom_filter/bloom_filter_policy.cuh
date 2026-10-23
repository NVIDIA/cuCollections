/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/utility/math.cuh>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cstdint>

namespace cuco::detail {

/**
 * @brief Sectorized Bloom filter policy with multiplicative-hashing fingerprint generation.
 *
 * Implements the Sectorized Bloom Filter (SBF) variant from "Optimizing Bloom Filters for Modern
 * GPU Architectures" (arXiv:2512.15595).
 *
 * Each key selects exactly one fixed-size block of `WordsPerBlock` words: the upper 32 bits of the
 * 64-bit hash pick the block via multiply-shift, and the lower 32 bits drive compile-time
 * salt-based multiplicative hashing that distributes `PatternBits` set bits across that block's
 * words (so a 64-bit hash function is required). Confining a key's probes to one block keeps a
 * lookup to a single contiguous region, minimizing memory transactions; throughput is best while
 * the whole filter fits the GPU cache domain, so sizing it relative to L2 is the main lever.
 *
 * `add` and `contains` take independent vectorization layouts (horizontal = cooperative-group size,
 * vertical = contiguous words per thread per step) because they favor opposite access patterns:
 * `add` spreads a block's words across cooperating threads so atomic writes proceed in parallel
 * (default: fully horizontal), while `contains` defaults to the paper's optimized layout of at
 * most one 32-byte sector per cooperating thread. `PatternBits` trades false-positive rate against
 * space.
 *
 * @tparam Hash 64-bit hash functor whose call operator returns `uint64_t`.
 * @tparam WordBytes Size in bytes of the underlying word type. Must be `4` or `8`.
 * @tparam WordsPerBlock Words per filter block. Must be a power of two and <= 32.
 * @tparam PatternBits Number of fingerprint bits (k in the paper).
 * @tparam AddHorizontalLayout CG size used for `add` (paper's Theta). Must be a power of two and
 * `AddHorizontalLayout * AddVerticalLayout <= WordsPerBlock`.
 * @tparam AddVerticalLayout Contiguous words processed per thread per `add` step (paper's Phi).
 * @tparam ContainsHorizontalLayout CG size used for `contains` (paper's Theta).
 * @tparam ContainsVerticalLayout Contiguous words processed per thread per `contains` step (paper's
 * Phi).
 * @tparam ConditionalAdd When `true`, `add` reads each word before the atomic OR and skips the
 * write when the required bits are already set. This trades a read for fewer atomic writes and is
 * beneficial when the filter is highly contended (e.g. close to full) or the input contains many
 * duplicate keys, where most writes would be redundant.
 * @tparam EarlyExitContains When `true`, `contains` short-circuits a thread's evaluation on the
 * first missing fingerprint slice. Beneficial when queried keys have a low match rate (most lookups
 * miss) and filter contention is low, so the common negative path exits early.
 * @tparam PersistingL2Access When `true`, filter word accesses in `add` and `contains` are
 * annotated with a persisting L2 access policy when the filter storage is in global memory. This
 * only emits cache-policy hints on the generated memory instructions; it does not reserve
 * persisting L2 capacity, and non-global storage such as shared memory is left unannotated. Pair
 * this with an application-managed L2 set-aside (for example
 * `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes)`) and reset the persisting cache after
 * the Bloom-filter phase. Use this only when the filter fits the reserved L2 region or the key
 * stream has sufficient locality: filters far larger than L2 can continually mark random lines as
 * persisting, thrashing the cache and affecting unrelated kernels.
 * @tparam FastL2Add When `true`, one hash-selected word of the first 32-byte sector of each key's
 * block is left untouched (its pattern word is cleared), so a key sets
 * `pattern_bits - ceil(pattern_bits / words_per_block)` bits in `words_per_block - 1` words. `add`
 * issues no atomic for the skipped word, so the atomic updates of a key cover its first sector only
 * partially, which recent GPUs process more efficiently than a fully covered sector; this speeds up
 * `add` while the filter is cache-resident. It changes the bit pattern (slightly higher
 * false-positive rate; not interchangeable with filters built without it). No effect for blocks
 * smaller than one 32-byte sector.
 */
template <class Hash,
          uint32_t WordBytes,
          uint32_t WordsPerBlock,
          uint32_t PatternBits,
          uint32_t AddHorizontalLayout,
          uint32_t AddVerticalLayout,
          uint32_t ContainsHorizontalLayout,
          uint32_t ContainsVerticalLayout,
          bool ConditionalAdd,
          bool EarlyExitContains,
          bool PersistingL2Access,
          bool FastL2Add>
class bloom_filter_policy {
  static_assert(WordBytes == 4 || WordBytes == 8, "WordBytes must be 4 or 8 for native atomicOr");

 public:
  using hasher           = Hash;  ///< 64-bit hash functor type
  using word_type        = cuda::std::conditional_t<WordBytes == 4,
                                                    unsigned int,
                                                    unsigned long long int>;  ///< Filter-block word type
  using hash_result_type = uint64_t;  ///< Hash function output type

 private:
  static constexpr uint32_t max_salts                          = 64;
  static constexpr cuda::std::array<uint32_t, max_salts> salts = {
    0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU, 0x705495c7U, 0x2df1424bU, 0x9efc4947U,
    0x5c6bfb31U, 0xb24bcdffU, 0xb6843d6dU, 0x6db04543U, 0x3a12efddU, 0xb0ddd463U, 0x8d22f6e7U,
    0xb82f1e53U, 0x7db9f86bU, 0xc7afe639U, 0xfb135cd7U, 0x693256e1U, 0x9466d871U, 0x23d3d02fU,
    0x6461d049U, 0x66a91621U, 0xbaa3006fU, 0x52fb8d99U, 0x3ea88b4fU, 0xf470cfdU,  0xb1db79a5U,
    0x9809fcd1U, 0xbced4445U, 0x2eb7c737U, 0x2cea6803U, 0x156f1955U, 0x8813c027U, 0xa26819f9U,
    0x4c3b57bdU, 0x7df94487U, 0xb975e769U, 0xb8f20cb5U, 0x5c9e2e77U, 0x5fb1735fU, 0x3a6f759bU,
    0x3c090923U, 0xfced424dU, 0xa187a6a9U, 0x6f070a41U, 0x2c85233bU, 0x7e62258bU, 0x2771ef17U,
    0x13bbf093U, 0x4ff059e5U, 0xe3ce3d0fU, 0xf1b4789fU, 0x9fbb6173U, 0x6a320cf5U, 0x1be2c481U,
    0x7ba8222bU, 0x6fd619b3U, 0x7b1bbf0dU, 0x8b8993adU, 0x448eca95U, 0x82ab09d9U, 0x2ce53909U,
    0x4f548685U};
  static constexpr uint32_t word_bits = cuda::std::numeric_limits<word_type>::digits;

 public:
  static constexpr uint32_t word_bytes      = WordBytes;      ///< Size in bytes of each filter word
  static constexpr uint32_t words_per_block = WordsPerBlock;  ///< Number of words per filter block
  static constexpr uint32_t pattern_bits    = PatternBits;    ///< Fingerprint bits per key

  static constexpr uint32_t add_horizontal_layout =
    AddHorizontalLayout;  ///< horizontal vectorization layout for add operation
  static constexpr uint32_t add_vertical_layout =
    AddVerticalLayout;  ///< vertical vectorization layout for add operation
  static constexpr uint32_t contains_horizontal_layout =
    ContainsHorizontalLayout;  ///< horizontal vectorization layout for contains operation
  static constexpr uint32_t contains_vertical_layout =
    ContainsVerticalLayout;  ///< vertical vectorization layout for contains operation

  static constexpr bool conditional_add =
    ConditionalAdd;  ///< read-before-atomic on add (skip redundant writes)
  static constexpr bool early_exit_contains =
    EarlyExitContains;  ///< short-circuit contains on first missing slice
  static constexpr bool persisting_l2_access =
    PersistingL2Access;  ///< apply persisting L2 cache-policy hints to filter accesses
  /// Leave one hash-selected word of each key's block untouched. No effect for blocks smaller
  /// than one 32-byte sector, whose updates are partial already.
  static constexpr bool fast_l2_add = FastL2Add and WordsPerBlock * WordBytes >= 32;

  static constexpr size_t max_filter_blocks =
    cuda::std::numeric_limits<uint32_t>::max();  ///< Upper bound on the number of filter blocks
  /// Lower bound on `pattern_bits`: at least one bit per word so every word contributes.
  static constexpr auto min_pattern_bits = words_per_block;
  /// Upper bound on `pattern_bits`: the total number of bits in a filter block, capped by the
  /// number of available salts.
  static constexpr auto max_pattern_bits = cuda::std::min(word_bits * words_per_block, max_salts);

 private:
  static constexpr uint32_t bit_index_width = cuda::std::bit_width(word_bits - 1);
  // TODO: for non-multiple `(pattern_bits, words_per_block)` configs (e.g. PatternBits=12,
  // WordsPerBlock=8), the salt walk in `set_bits` advances `PatternArrayIndex` every
  // `max_bits_per_word` salts, packing all bits into the first
  // `ceil(pattern_bits / words_per_block)` words and leaving the rest at zero. This wastes block
  // capacity and inflates FPR. Distribute floor bits to every word plus one extra bit to the
  // first `pattern_bits % words_per_block` words, and update the salt-to-word mapping in
  // `set_bits` accordingly.
  static constexpr uint32_t max_bits_per_word =
    cuco::detail::int_div_ceil(pattern_bits, words_per_block);

 public:
  /**
   * @brief Constructs a Bloom filter policy.
   *
   * @param hash Hash function used to generate fingerprints.
   */
  __host__ __device__ constexpr bloom_filter_policy(Hash hash = {}) : hash_{hash}
  {
    static_assert(pattern_bits >= min_pattern_bits,
                  "pattern_bits must be at least words_per_block");
    static_assert(pattern_bits <= max_pattern_bits,
                  "pattern_bits must be less than the total number of bits in a filter block");
    // Require exact tiling. With `words_per_block` a power of two, this is equivalent to requiring
    // both `add_horizontal_layout` and `add_vertical_layout` to be powers of two with product
    // <= `words_per_block`. The internal loop count uses integer division on the product; non-
    // dividing layouts would leave trailing words uninserted on add while contains still expects
    // non-zero patterns there, producing false negatives for every inserted key.
    static_assert(words_per_block % (add_horizontal_layout * add_vertical_layout) == 0,
                  "add_horizontal_layout * add_vertical_layout must evenly divide words_per_block");
    static_assert(
      words_per_block % (contains_horizontal_layout * contains_vertical_layout) == 0,
      "contains_horizontal_layout * contains_vertical_layout must evenly divide words_per_block");
  }

  /**
   * @brief Splits the 64-bit hash of a key into its upper and lower 32 bits.
   *
   * The upper half is used for block selection (via multiply-shift); the lower half drives the
   * per-word fingerprint pattern via salt-based multiplicative hashing.
   *
   * @tparam Key Key type.
   *
   * @param key Key to hash.
   *
   * @return `{upper 32 bits, lower 32 bits}` of the 64-bit hash.
   */
  template <class Key>
  __device__ constexpr cuda::std::pair<uint32_t, uint32_t> split_hash(Key const& key) const
  {
    // The split_hash() design requires a 64-bit hash split into upper 32 bits (block selection
    // via multiply-shift) and lower 32 bits (pattern generation via salt-based multiplicative
    // hashing). This is a permanent design requirement, not a temporary limitation.
    static_assert(cuda::std::is_same_v<decltype(hash_(key)), hash_result_type>,
                  "bloom_filter_policy requires a 64-bit hash function");
    auto const hash_value = hash_(key);
    return {static_cast<uint32_t>(hash_value >> 32), static_cast<uint32_t>(hash_value)};
  }

  /**
   * @brief Determines the filter block a key maps to via fast multiply-shift modulo.
   *
   * @tparam Extent Size type used to determine the number of blocks in the filter.
   *
   * @param upper_hash_value Upper 32 bits of the key's hash.
   * @param num_blocks Number of blocks in the filter.
   *
   * @return Block index in `[0, num_blocks)`.
   */
  template <class Extent>
  __device__ constexpr auto block_index(uint32_t upper_hash_value, Extent num_blocks) const
  {
    return static_cast<uint32_t>((static_cast<uint64_t>(upper_hash_value) *
                                  static_cast<typename Extent::value_type>(num_blocks)) >>
                                 32);
  }

  /**
   * @brief Generates the per-word fingerprint pattern for a key when the horizontal layout is 1.
   *
   * @tparam LoopIndex Outer-loop iteration index when `words_per_block / VerticalLayout > 1`.
   * @tparam VerticalLayout Number of contiguous words this call produces.
   *
   * @param lower_hash_value Lower 32 bits of the key's hash.
   *
   * @return Array of `VerticalLayout` words.
   */
  template <uint32_t LoopIndex, uint32_t VerticalLayout>
  __device__ constexpr auto array_pattern(uint32_t lower_hash_value) const
  {
    auto pattern = pattern_impl<LoopIndex, VerticalLayout>(lower_hash_value);
    if constexpr (fast_l2_add) {
      clear_skipped(pattern, lower_hash_value, LoopIndex * VerticalLayout);
    }
    return pattern;
  }

  /**
   * @brief Generates the per-word fingerprint pattern for a key when the horizontal layout is > 1.
   *
   * @tparam LoopIndex Outer-loop iteration index.
   * @tparam HorizontalLayout Cooperative-group size cooperating on a single key.
   * @tparam VerticalLayout Number of contiguous words this call produces.
   *
   * @param lower_hash_value Lower 32 bits of the key's hash.
   * @param thread_index Caller's rank within the cooperative group.
   *
   * @return Array of `VerticalLayout` words owned by the calling thread.
   */
  template <uint32_t LoopIndex, uint32_t HorizontalLayout, uint32_t VerticalLayout>
  __device__ constexpr auto array_pattern(uint32_t lower_hash_value, uint32_t thread_index) const
  {
    return array_pattern<LoopIndex>(
      lower_hash_value, make_thread_salts<HorizontalLayout, VerticalLayout>(thread_index));
  }

  /**
   * @brief Salts of one thread of a cooperative group (horizontal layout > 1) for every outer-loop
   * iteration.
   *
   * The salts depend only on the thread's rank. Selecting them once, ahead of a loop over keys,
   * leaves one multiply per fingerprint bit inside the loop.
   *
   * @tparam HorizontalLayout Cooperative-group size cooperating on a single key.
   * @tparam VerticalLayout Number of contiguous words per thread and outer-loop iteration.
   */
  template <uint32_t HorizontalLayout, uint32_t VerticalLayout>
  struct thread_salts {
    static constexpr uint32_t horizontal_layout = HorizontalLayout;  ///< Threads per key
    static constexpr uint32_t vertical_layout   = VerticalLayout;    ///< Words per thread
    /// Outer-loop iterations covering a block
    static constexpr uint32_t loop_count = words_per_block / (HorizontalLayout * VerticalLayout);
    /// Salts of one virtual thread
    static constexpr uint32_t salts_per_thread = max_bits_per_word * VerticalLayout;

    /// Salts per outer-loop iteration; unused entries (see `num_salts`) are zero
    cuda::std::array<cuda::std::array<uint32_t, salts_per_thread>, loop_count> salt;
    /// Valid salts per outer-loop iteration, fewer than `salts_per_thread` only for trailing
    /// virtual threads when `pattern_bits` is not a multiple of `words_per_block`
    cuda::std::array<uint32_t, loop_count> num_salts;
    uint32_t thread_index;  ///< Rank within the cooperative group
  };

  /**
   * @brief Selects the salts of one cooperative-group thread.
   *
   * @tparam HorizontalLayout Cooperative-group size cooperating on a single key.
   * @tparam VerticalLayout Number of contiguous words per thread and outer-loop iteration.
   *
   * @param thread_index Caller's rank within the cooperative group.
   *
   * @return The thread's salts.
   */
  template <uint32_t HorizontalLayout, uint32_t VerticalLayout>
  __device__ constexpr auto make_thread_salts(uint32_t thread_index) const
  {
    thread_salts<HorizontalLayout, VerticalLayout> ts{};
    ts.thread_index = thread_index;
    select_salts<0>(ts, thread_index);
    return ts;
  }

  /**
   * @brief Generates the per-word fingerprint pattern of one cooperative-group thread from its
   * salts.
   *
   * @tparam LoopIndex Outer-loop iteration index.
   * @tparam ThreadSalts `thread_salts` of the calling thread.
   *
   * @param lower_hash_value Lower 32 bits of the key's hash.
   * @param ts Salts of the calling thread.
   *
   * @return Array of `ThreadSalts::vertical_layout` words owned by the calling thread.
   */
  template <uint32_t LoopIndex, class ThreadSalts>
  __device__ constexpr auto array_pattern(uint32_t lower_hash_value, ThreadSalts const& ts) const
  {
    auto pattern = salted_pattern<LoopIndex>(lower_hash_value, ts);
    if constexpr (fast_l2_add) {
      clear_skipped(pattern,
                    lower_hash_value,
                    (LoopIndex * ThreadSalts::horizontal_layout + ts.thread_index) *
                      ThreadSalts::vertical_layout);
    }
    return pattern;
  }

  /**
   * @brief Checks the stored words against a key's pattern when the horizontal layout is 1. The
   * FastL2Add skipped word passes by its index, so every word keeps a plain bit test.
   *
   * @tparam LoopIndex Outer-loop iteration index when `words_per_block / VerticalLayout > 1`.
   * @tparam VerticalLayout Number of contiguous words checked.
   * @tparam StoredArrayT Array of `VerticalLayout` words.
   *
   * @param stored_pattern The stored words.
   * @param lower_hash_value Lower 32 bits of the key's hash.
   *
   * @return Whether every checked word contains its pattern bits.
   */
  template <uint32_t LoopIndex, uint32_t VerticalLayout, class StoredArrayT>
  __device__ constexpr bool matches_pattern(StoredArrayT const& stored_pattern,
                                            uint32_t lower_hash_value) const
  {
    auto const expected = pattern_impl<LoopIndex, VerticalLayout>(lower_hash_value);
    bool match          = true;
    for (uint32_t i = 0; i < VerticalLayout; ++i) {
      match &= (stored_pattern[i] & expected[i]) == expected[i] or
               (fast_l2_add and LoopIndex * VerticalLayout + i == skip_index(lower_hash_value));
    }
    return match;
  }

  /**
   * @brief Checks the stored words against a key's pattern when the horizontal layout is > 1. The
   * FastL2Add skipped word passes by its index, so every word keeps a plain bit test.
   *
   * @tparam LoopIndex Outer-loop iteration index.
   * @tparam ThreadSalts `thread_salts` of the calling thread.
   * @tparam StoredArrayT Array of `ThreadSalts::vertical_layout` words.
   *
   * @param stored_pattern The calling thread's stored words.
   * @param lower_hash_value Lower 32 bits of the key's hash.
   * @param ts Salts of the calling thread.
   *
   * @return Whether every checked word contains its pattern bits.
   */
  template <uint32_t LoopIndex, class ThreadSalts, class StoredArrayT>
  __device__ constexpr bool matches_pattern(StoredArrayT const& stored_pattern,
                                            uint32_t lower_hash_value,
                                            ThreadSalts const& ts) const
  {
    auto const expected = salted_pattern<LoopIndex>(lower_hash_value, ts);
    uint32_t const first_word =
      (LoopIndex * ThreadSalts::horizontal_layout + ts.thread_index) * ThreadSalts::vertical_layout;
    bool match = true;
    for (uint32_t i = 0; i < ThreadSalts::vertical_layout; ++i) {
      match &= (stored_pattern[i] & expected[i]) == expected[i] or
               (fast_l2_add and first_word + i == skip_index(lower_hash_value));
    }
    return match;
  }

  /**
   * @brief Generates the per-word fingerprint pattern without the FastL2Add skip applied, when the
   * horizontal layout is 1.
   *
   * @tparam LoopIndex Outer-loop iteration index when `words_per_block / VerticalLayout > 1`.
   * @tparam VerticalLayout Number of contiguous words this call produces.
   *
   * @param lower_hash_value Lower 32 bits of the key's hash.
   *
   * @return Array of `VerticalLayout` words.
   */
  template <uint32_t LoopIndex, uint32_t VerticalLayout>
  __device__ constexpr auto full_array_pattern(uint32_t lower_hash_value) const
  {
    return pattern_impl<LoopIndex, VerticalLayout>(lower_hash_value);
  }

  /**
   * @brief Index of the FastL2Add skipped word in the block's first sector: the top bits of the
   * hash.
   *
   * @param hash Lower 32 bits of the key's hash.
   *
   * @return Word index in `[0, words per sector)`.
   */
  __device__ static constexpr uint32_t skip_index(uint32_t hash)
  {
    constexpr uint32_t index_width = cuda::std::bit_width(words_per_sector - 1);
    return hash >> (32 - index_width);
  }

 private:
  hasher hash_;

  /// The skipped word is taken from the block's first 32-byte sector: that is the sector whose
  /// partial coverage makes the update cheaper.
  static constexpr uint32_t sector_bytes     = 32;
  static constexpr uint32_t words_per_sector = sector_bytes / WordBytes;

  /// Clears the pattern word of the hash-selected skipped word if it lies in this pattern array,
  /// whose first element is block word `first_word`.
  ///
  /// Branchless: the skip index is per key (data-dependent), so comparing it against each word
  /// index lets the compiler emit a switch over its values, which diverges across a warp. Instead
  /// a keep-mask with the skipped bit cleared is shifted per word and turned into an all-ones or
  /// all-zeros word mask.
  template <class PatternArrayT>
  __device__ constexpr void clear_skipped(PatternArrayT& pattern_array,
                                          uint32_t hash,
                                          uint32_t first_word) const
  {
    uint32_t const skip = skip_index(hash);
    if constexpr (cuda::std::tuple_size_v<PatternArrayT> == 1) {
      pattern_array[0] &= word_type{0} - static_cast<word_type>(first_word != skip);
    } else {
      uint32_t const keep = ~(uint32_t{1} << skip);  // words_per_block <= 32
      for (uint32_t i = 0; i < pattern_array.size(); ++i) {
        pattern_array[i] &= word_type{0} - static_cast<word_type>((keep >> (first_word + i)) & 1u);
      }
    }
  }

  /**
   * @brief pattern_impl - Computes the bit pattern for a vertical layout of words.
   * I use the terminology of a `virtual thread` to refer to an ordering of the vertical layouts,
   * namely
   *   virtual_thread_index = LoopIndex * HorizontalLayout + thread_index,
   * where LoopIndex is the index of the outermost loop in the range:
   *     [0, words_per_block / (HorizontalLayout * VerticalLayout)).
   * @param hash
   * @return cuda::std::array<word_type, VerticalLayout> - The bit pattern for the vertical layout
   * defined by the LoopIndex.
   */

  // Precondition: <add/contains>_horizontal_layout == 1
  template <uint32_t LoopIndex, uint32_t VerticalLayout>
  __device__ constexpr auto pattern_impl(uint32_t hash) const
  {
    using pattern_array_t = cuda::std::array<word_type, VerticalLayout>;

    // Sanity check
    constexpr uint32_t num_iterations = words_per_block / VerticalLayout;
    static_assert(LoopIndex < num_iterations,
                  "the loop index cannot exceed the number of loop iterations");

    pattern_array_t pattern_array{0};
    constexpr uint32_t salt_start_index = max_bits_per_word * VerticalLayout * LoopIndex;
    constexpr uint32_t salt_end_index =
      cuda::std::min(salt_start_index + max_bits_per_word * VerticalLayout, pattern_bits);
    constexpr uint32_t pattern_array_start_index = 0;
    set_bits<salt_start_index, salt_end_index, pattern_array_start_index>(hash, pattern_array);
    return pattern_array;
  }

  // Fills `ts` for outer-loop iterations LoopIndex.. with the salts of virtual thread
  // `LoopIndex * horizontal_layout + thread_index`. Runs once per thread, ahead of any loop over
  // keys.
  template <uint32_t LoopIndex, class ThreadSalts>
  __device__ constexpr void select_salts(ThreadSalts& ts, uint32_t thread_index) const
  {
    if constexpr (LoopIndex < ThreadSalts::loop_count) {
      using salt_array_t = cuda::std::array<uint32_t, ThreadSalts::salts_per_thread>;
      ts.salt[LoopIndex] =
        salts_of<LoopIndex, 0, ThreadSalts::horizontal_layout, ThreadSalts, salt_array_t>(
          thread_index);
      if constexpr (pattern_bits != max_bits_per_word * words_per_block) {
        ts.num_salts[LoopIndex] =
          num_salts_of<LoopIndex, 0, ThreadSalts::horizontal_layout, ThreadSalts>(thread_index);
      }
      select_salts<LoopIndex + 1>(ts, thread_index);
    }
  }

  // Salts of rank `thread_index` in [LowerRank, UpperRank): a compile-time tree of selects, so the
  // compiler can keep it branch-free.
  template <uint32_t LoopIndex,
            uint32_t LowerRank,
            uint32_t UpperRank,
            class ThreadSalts,
            class SaltArrayT>
  __device__ static constexpr SaltArrayT salts_of(uint32_t thread_index)
  {
    if constexpr (LowerRank + 1 == UpperRank) {
      constexpr uint32_t first =
        ThreadSalts::salts_per_thread * (LoopIndex * ThreadSalts::horizontal_layout + LowerRank);
      constexpr uint32_t last = cuda::std::min(first + ThreadSalts::salts_per_thread, pattern_bits);
      SaltArrayT s{};
      copy_salts<first, last>(s, cuda::std::make_index_sequence<ThreadSalts::salts_per_thread>{});
      return s;
    } else {
      // Interleaved halves selected by the lowest rank bit first: a select per pair, then per
      // pair of pairs, which the compiler keeps branch-free.
      constexpr uint32_t stride = 1;
      return salts_bit<LoopIndex,
                       LowerRank,
                       UpperRank - LowerRank,
                       stride,
                       ThreadSalts,
                       SaltArrayT>(thread_index);
    }
  }

  // Ranks LowerRank + k * Stride for k in [0, Count): select on bit log2(Stride) of the rank.
  template <uint32_t LoopIndex,
            uint32_t LowerRank,
            uint32_t Count,
            uint32_t Stride,
            class ThreadSalts,
            class SaltArrayT>
  __device__ static constexpr SaltArrayT salts_bit(uint32_t thread_index)
  {
    if constexpr (Count == 1) {
      return salts_of<LoopIndex, LowerRank, LowerRank + 1, ThreadSalts, SaltArrayT>(thread_index);
    } else {
      auto const even =
        salts_bit<LoopIndex, LowerRank, Count / 2, Stride * 2, ThreadSalts, SaltArrayT>(
          thread_index);
      auto const odd =
        salts_bit<LoopIndex, LowerRank + Stride, Count / 2, Stride * 2, ThreadSalts, SaltArrayT>(
          thread_index);
      return (thread_index & Stride) ? odd : even;
    }
  }

  // Valid salts of rank `thread_index` in [LowerRank, UpperRank).
  template <uint32_t LoopIndex, uint32_t LowerRank, uint32_t UpperRank, class ThreadSalts>
  __device__ static constexpr uint32_t num_salts_of(uint32_t thread_index)
  {
    if constexpr (LowerRank + 1 == UpperRank) {
      constexpr uint32_t first =
        ThreadSalts::salts_per_thread * (LoopIndex * ThreadSalts::horizontal_layout + LowerRank);
      constexpr uint32_t last = cuda::std::min(first + ThreadSalts::salts_per_thread, pattern_bits);
      return last > first ? last - first : 0u;
    } else {
      constexpr uint32_t mid = (LowerRank + UpperRank) / 2;
      return thread_index < mid
               ? num_salts_of<LoopIndex, LowerRank, mid, ThreadSalts>(thread_index)
               : num_salts_of<LoopIndex, mid, UpperRank, ThreadSalts>(thread_index);
    }
  }

  // Salt `Index` if it belongs to the range ending at `Last` (<= pattern_bits), else 0.
  template <uint32_t Index, uint32_t Last>
  __device__ static constexpr uint32_t salt_or_zero()
  {
    if constexpr (Index < Last) {
      return cuda::std::get<Index>(salts);
    } else {
      return 0u;
    }
  }

  // dst[s] = salts[First + s] for First + s < Last, else 0.
  template <uint32_t First, uint32_t Last, class SaltArrayT, size_t... S>
  __device__ static constexpr void copy_salts(SaltArrayT& dst, cuda::std::index_sequence<S...>)
  {
    ((dst[S] = salt_or_zero<First + static_cast<uint32_t>(S), Last>()), ...);
  }

  // Fingerprint words of one cooperative-group thread from its salts (no FastL2Add skip).
  template <uint32_t LoopIndex, class ThreadSalts>
  __device__ constexpr auto salted_pattern(uint32_t hash, ThreadSalts const& ts) const
  {
    static_assert(LoopIndex < ThreadSalts::loop_count,
                  "the loop index cannot exceed the number of loop iterations");
    // Every virtual thread has all its salts unless pattern_bits leaves trailing words short.
    constexpr bool all_salts = pattern_bits == max_bits_per_word * words_per_block;

    cuda::std::array<word_type, ThreadSalts::vertical_layout> pattern_array{0};
    for (uint32_t s = 0; s < ThreadSalts::salts_per_thread; ++s) {
      if (all_salts or s < ts.num_salts[LoopIndex]) {
        uint32_t const bit_index = (ts.salt[LoopIndex][s] * hash) >> (32 - bit_index_width);
        pattern_array[s / max_bits_per_word] |= word_type{1} << bit_index;
      }
    }
    return pattern_array;
  }

  // Set bits in the pattern array using salts starting from SaltIndex.
  template <uint32_t SaltIndex,
            uint32_t SaltEndIndex,
            uint32_t PatternArrayIndex,
            class PatternArrayT>
  __device__ constexpr void set_bits(uint32_t hash, PatternArrayT& pattern_array) const
  {
    if constexpr (SaltIndex < SaltEndIndex) {
      // Select top bit_index_width bits from salted hash to determine the bit index.
      uint32_t const bit_index =
        (cuda::std::get<SaltIndex>(salts) * hash) >> (32 - bit_index_width);

      // Set the bit in the pattern array.
      cuda::std::get<PatternArrayIndex>(pattern_array) |= word_type{1} << bit_index;

      // Recurse.
      constexpr uint32_t next_salt_index = SaltIndex + 1;
      constexpr uint32_t next_pattern_array_index =
        PatternArrayIndex + (next_salt_index % max_bits_per_word == 0 ? 1 : 0);
      set_bits<next_salt_index, SaltEndIndex, next_pattern_array_index>(hash, pattern_array);
    }
  }
};

}  // namespace cuco::detail
