/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#pragma once

#include <cuco/detail/utility/math.cuh>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cstdint>

namespace cuco::detail {

/**
 * @brief Sectorized Bloom filter policy with multiplicative-hashing fingerprint generation.
 *
 * Implements the Sectorized Bloom Filter (SBF) variant from "Optimizing Bloom Filters for Modern
 * GPU Architectures" (arXiv:2512.15595). Distributes `PatternBits` set bits across `WordsPerBlock`
 * words using compile-time salt-based multiplicative hashing. The hash result is split into upper
 * 32 bits (block selection via multiply-shift) and lower 32 bits (pattern generation), so a 64-bit
 * hash function is required by design.
 *
 * @tparam Hash 64-bit hash functor whose return type satisfies `is_same_v<hash_result_type,
 * uint64_t>`.
 * @tparam Word Underlying word type of a filter block. Must be an atomically updatable integral.
 * @tparam WordsPerBlock Words per filter block. Must be a power of two and <= 32.
 * @tparam PatternBits Number of fingerprint bits (k in the paper).
 * @tparam AddHorizontalLayout CG size used for `add` (paper's Theta). Must be a power of two and
 * `AddHorizontalLayout * AddVerticalLayout <= WordsPerBlock`.
 * @tparam AddVerticalLayout Contiguous words processed per thread per `add` step (paper's Phi).
 * @tparam ContainsHorizontalLayout CG size used for `contains` (paper's Theta).
 * @tparam ContainsVerticalLayout Contiguous words processed per thread per `contains` step (paper's
 * Phi).
 */
template <class Hash,
          class Word,
          uint32_t WordsPerBlock,
          uint32_t PatternBits,
          uint32_t AddHorizontalLayout,
          uint32_t AddVerticalLayout,
          uint32_t ContainsHorizontalLayout,
          uint32_t ContainsVerticalLayout>
class parametric_filter_policy {
 public:
  using hasher             = Hash;                            ///< 64-bit hash functor type
  using word_type          = Word;                            ///< Underlying filter-block word type
  using hash_argument_type = typename hasher::argument_type;  ///< Hash function input type
  using hash_result_type =
    decltype(std::declval<hasher>()(std::declval<hash_argument_type>()));  ///< Hash function
                                                                           ///< output type

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
   * @brief Constructs a parametric filter policy.
   *
   * @param hash Hash function used to generate fingerprints.
   */
  __host__ __device__ constexpr parametric_filter_policy(Hash hash = {}) : hash_{hash}
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
    // The split_hash() design requires a 64-bit hash split into upper 32 bits (block selection
    // via multiply-shift) and lower 32 bits (pattern generation via salt-based multiplicative
    // hashing). This is a permanent design requirement, not a temporary limitation.
    static_assert(cuda::std::is_same_v<hash_result_type, uint64_t>,
                  "parametric_filter_policy requires a 64-bit hash function");
  }

  /**
   * @brief Splits the 64-bit hash of a key into its upper and lower 32 bits.
   *
   * The upper half is used for block selection (via multiply-shift); the lower half drives the
   * per-word fingerprint pattern via salt-based multiplicative hashing.
   *
   * @param key Key to hash.
   *
   * @return `{upper 32 bits, lower 32 bits}` of the 64-bit hash.
   */
  __device__ constexpr cuda::std::pair<uint32_t, uint32_t> split_hash(hash_argument_type key) const
  {
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
    return pattern_impl<LoopIndex, VerticalLayout>(lower_hash_value);
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
    return pattern_impl<LoopIndex, HorizontalLayout, VerticalLayout>(lower_hash_value,
                                                                     thread_index);
  }

 private:
  hasher hash_;

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

  // Precondition: <add/contains>_horizontal_layout > 1
  template <uint32_t LoopIndex, uint32_t HorizontalLayout, uint32_t VerticalLayout>
  __device__ constexpr auto pattern_impl(uint32_t hash, uint32_t thread_index) const
  {
    using pattern_array_t = cuda::std::array<word_type, VerticalLayout>;

    // Sanity check
    constexpr uint32_t num_iterations = words_per_block / (HorizontalLayout * VerticalLayout);
    static_assert(LoopIndex < num_iterations,
                  "the loop index cannot exceed the number of loop iterations");

    // [lower_bound, upper_bound) defines the range of virtual thread indices for this loop
    // iteration.
    constexpr uint32_t lower_bound = LoopIndex * HorizontalLayout;
    constexpr uint32_t upper_bound = lower_bound + HorizontalLayout;

    // A virtual thread flips max_bits_per_virtual_thread bits in the pattern array, excepting
    // potentially some of the last virtual threads (if pattern_bits % words_per_block != 0).
    constexpr uint32_t max_bits_per_virtual_thread = max_bits_per_word * VerticalLayout;

    pattern_array_t pattern_array{0};
    if constexpr (num_iterations == 1) {
      thread_dispatch<max_bits_per_virtual_thread, lower_bound, upper_bound>(
        hash, thread_index, pattern_array);
    } else {
      uint32_t const virtual_thread_index = LoopIndex * HorizontalLayout + thread_index;
      thread_dispatch<max_bits_per_virtual_thread, lower_bound, upper_bound>(
        hash, virtual_thread_index, pattern_array);
    }
    return pattern_array;
  }

  // Dispatches a dynamic virtual thread index to a static virtual thread index by building a
  // compile-time decision tree over the range [LowerBound, UpperBound) for the virtual thread
  // index. This method is only used when <add/contains>_horizontal_layout > 1.
  template <uint32_t MaxBitsPerVirtualThread,
            uint32_t LowerBound,
            uint32_t UpperBound,
            class PatternArrayT>
  __device__ constexpr void thread_dispatch(uint32_t hash,
                                            uint32_t thread_index,
                                            PatternArrayT& pattern_array) const
  {
    // Sanity check
    static_assert(LowerBound < UpperBound);

    if constexpr (LowerBound + 1 == UpperBound) {
      // Base case: thread_index == LowerBound
      constexpr uint32_t salt_start_index = MaxBitsPerVirtualThread * LowerBound;
      constexpr uint32_t salt_end_index =
        cuda::std::min(salt_start_index + MaxBitsPerVirtualThread, pattern_bits);
      constexpr uint32_t pattern_array_start_index = 0;
      set_bits<salt_start_index, salt_end_index, pattern_array_start_index>(hash, pattern_array);
    } else {
      // Recursive case: thread_index > LowerBound
      constexpr uint32_t mid = (LowerBound + UpperBound) / 2;
      if (thread_index < mid) {
        thread_dispatch<MaxBitsPerVirtualThread, LowerBound, mid>(
          hash, thread_index, pattern_array);
      } else {
        thread_dispatch<MaxBitsPerVirtualThread, mid, UpperBound>(
          hash, thread_index, pattern_array);
      }
    }
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