/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
 * Implements the Sectorized Bloom Filter (SBF) and Cache-Sectorized Bloom Filter (CSBF) variants
 * from "Optimizing Bloom Filters for Modern GPU Architectures" (arXiv:2512.15595). Distributes
 * `PatternBits` set bits across `WordsPerBlock` words using compile-time salt-based multiplicative
 * hashing. The hash result is split into upper 32 bits (block selection via multiply-shift) and
 * lower 32 bits (pattern generation), so a 64-bit hash function is required by design.
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
 * @tparam GroupsPerBlock Cache-sectorization groups per block (paper's z). Defaults to
 * `WordsPerBlock` for non-CSBF mode; setting `GroupsPerBlock < WordsPerBlock` enables CSBF.
 */
template <class Hash,
          class Word,
          uint32_t WordsPerBlock,
          uint32_t PatternBits,
          uint32_t AddHorizontalLayout,
          uint32_t AddVerticalLayout,
          uint32_t ContainsHorizontalLayout,
          uint32_t ContainsVerticalLayout,
          uint32_t GroupsPerBlock = WordsPerBlock>
class parametric_filter_policy {
 public:
  using hasher             = Hash;
  using word_type          = Word;
  using hash_argument_type = typename hasher::argument_type;
  using hash_result_type   = decltype(std::declval<hasher>()(std::declval<hash_argument_type>()));

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
  static constexpr uint32_t words_per_block = WordsPerBlock;
  static constexpr uint32_t pattern_bits    = PatternBits;

  static constexpr uint32_t add_horizontal_layout =
    AddHorizontalLayout;  ///< horizontal vectorization layout for add operation
  static constexpr uint32_t add_vertical_layout =
    AddVerticalLayout;  ///< vertical vectorization layout for add operation
  static constexpr uint32_t contains_horizontal_layout =
    ContainsHorizontalLayout;  ///< horizontal vectorization layout for contains operation
  static constexpr uint32_t contains_vertical_layout =
    ContainsVerticalLayout;  ///< vertical vectorization layout for contains operation

  static constexpr size_t max_filter_blocks = cuda::std::numeric_limits<uint32_t>::max();
  // This ensures each word in the block has at least one bit set; otherwise we would never
  // use some of the words
  static constexpr auto min_pattern_bits = words_per_block;
  // The maximum number of bits to be set for a key is capped by the total number of bits in
  // the filter block, as well as the number of available salts
  static constexpr auto max_pattern_bits = cuda::std::min(word_bits * words_per_block, max_salts);

  //===----------Cache-Sectorized----------===//
  static constexpr uint32_t groups_per_block = GroupsPerBlock;
  static constexpr bool is_cache_sectorized  = groups_per_block != words_per_block ? true : false;
  static constexpr uint32_t words_per_group  = words_per_block / groups_per_block;
  static constexpr uint32_t max_bits_per_group =
    cuco::detail::int_div_ceil(pattern_bits, groups_per_block);
  static constexpr uint32_t add_groups_per_vertical_layout = add_vertical_layout / words_per_group;
  static constexpr uint32_t contains_groups_per_vertical_layout =
    contains_vertical_layout / words_per_group;
  static constexpr uint32_t group_index_salt  = 0x5bd1e995U;
  static constexpr uint32_t group_index_width = cuda::std::bit_width(words_per_group - 1);
  static constexpr uint32_t group_index_mask  = words_per_group - 1;

 private:
  static constexpr uint32_t bit_index_width = cuda::std::bit_width(word_bits - 1);
  static constexpr uint32_t max_bits_per_word =
    cuco::detail::int_div_ceil(pattern_bits, words_per_block);

 public:
  __host__ __device__ constexpr parametric_filter_policy(Hash hash = {}) : hash_{hash}
  {
    static_assert(pattern_bits >= min_pattern_bits,
                  "pattern_bits must be at least words_per_block");
    static_assert(pattern_bits <= max_pattern_bits,
                  "pattern_bits must be less than the total number of bits in a filter block");
    static_assert(add_horizontal_layout * add_vertical_layout <= words_per_block,
                  "the product of add_horizontal_layout and add_vertical_layout must be less than "
                  "or equal to words_per_block");
    static_assert(contains_horizontal_layout * contains_vertical_layout <= words_per_block,
                  "the product of contains_horizontal_layout and contains_vertical_layout must be "
                  "less than or equal to words_per_block");
    // The split_hash() design requires a 64-bit hash split into upper 32 bits (block selection
    // via multiply-shift) and lower 32 bits (pattern generation via salt-based multiplicative
    // hashing). This is a permanent design requirement, not a temporary limitation.
    static_assert(cuda::std::is_same_v<hash_result_type, uint64_t>,
                  "parametric_filter_policy requires a 64-bit hash function");

    //===----------Cache-Sectorized----------===//
    static_assert(
      is_cache_sectorized == false || (groups_per_block > 0 && groups_per_block < words_per_block &&
                                       (words_per_block % groups_per_block == 0)),
      "in cache-sectorized filter, the number of groups must be positive, be fewer "
      "than words_per_block, and evenly divide words_per_block");
    static_assert(is_cache_sectorized == false || (add_vertical_layout >= words_per_group),
                  "in cache-sectorized filter, the vertical layout for add must be at "
                  "least the number of words per group");
    static_assert(is_cache_sectorized == false || (contains_vertical_layout >= words_per_group),
                  "in cache-sectorized filter, the vertical layout for contains must be at "
                  "least the number of words per group");
    static_assert(is_cache_sectorized == false || groups_per_block * group_index_width <= 32,
                  "in cache-sectorized filter, the number of bits needed to index groups must fit "
                  "within 32 bits");
  }

  // Return {upper 32b, lower 32b} of 64b hash
  __device__ constexpr cuda::std::pair<uint32_t, uint32_t> split_hash(hash_argument_type key) const
  {
    auto const hash_value = hash_(key);
    return {static_cast<uint32_t>(hash_value >> 32), static_cast<uint32_t>(hash_value)};
  }

  template <class Extent>
  __device__ constexpr auto block_index(uint32_t upper_hash_value, Extent num_blocks) const
  {
    // return upper_hash_value % num_blocks;
    return static_cast<uint32_t>((static_cast<uint64_t>(upper_hash_value) *
                                  static_cast<typename Extent::value_type>(num_blocks)) >>
                                 32);
  }

  template <uint32_t LoopIndex, uint32_t VerticalLayout>
  __device__ constexpr auto array_pattern(uint32_t lower_hash_value) const
  {
    return pattern_impl<LoopIndex, VerticalLayout>(lower_hash_value);
  }

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
    // For cache-sectorized, we set bits for one word for each group per vertical layout.
    constexpr uint32_t groups_per_vertical_layout = VerticalLayout / words_per_group;
    using pattern_array_t =
      cuda::std::conditional_t<is_cache_sectorized,
                               cuda::std::array<word_type, groups_per_vertical_layout>,
                               cuda::std::array<word_type, VerticalLayout>>;

    // Sanity check
    constexpr uint32_t num_iterations = words_per_block / VerticalLayout;
    static_assert(LoopIndex < num_iterations,
                  "the loop index cannot exceed the number of loop iterations");

    pattern_array_t pattern_array{0};
    constexpr uint32_t salt_start_index =
      is_cache_sectorized ? max_bits_per_group * groups_per_vertical_layout * LoopIndex
                          : max_bits_per_word * VerticalLayout * LoopIndex;
    constexpr uint32_t salt_end_index =
      is_cache_sectorized
        ? cuda::std::min(salt_start_index + max_bits_per_group * groups_per_vertical_layout,
                         pattern_bits)
        : cuda::std::min(salt_start_index + max_bits_per_word * VerticalLayout, pattern_bits);
    constexpr uint32_t pattern_array_start_index = 0;
    set_bits<salt_start_index, salt_end_index, pattern_array_start_index>(hash, pattern_array);
    return pattern_array;
  }

  // Precondition: <add/contains>_horizontal_layout > 1
  template <uint32_t LoopIndex, uint32_t HorizontalLayout, uint32_t VerticalLayout>
  __device__ constexpr auto pattern_impl(uint32_t hash, uint32_t thread_index) const
  {
    // For cache-sectorized, we set bits for one word for each group per vertical layout.
    constexpr uint32_t groups_per_vertical_layout = VerticalLayout / words_per_group;
    using pattern_array_t =
      cuda::std::conditional_t<is_cache_sectorized,
                               cuda::std::array<word_type, groups_per_vertical_layout>,
                               cuda::std::array<word_type, VerticalLayout>>;

    // Sanity check
    constexpr uint32_t num_iterations = words_per_block / (HorizontalLayout * VerticalLayout);
    static_assert(LoopIndex < num_iterations,
                  "the loop index cannot exceed the number of loop iterations");

    // [lower_bound, upper_bound) defines the range of virtual thread indices for this loop
    // iteration.
    constexpr uint32_t lower_bound = LoopIndex * HorizontalLayout;
    constexpr uint32_t upper_bound = lower_bound + HorizontalLayout;

    // A virtual thread flips max_bits_per_virtual_thread bits in the pattern array, excepting
    // potentially some of the last virtual threads (if [sectorized] pattern_bits % words_per_block
    // != 0 or [cache-sectorized] pattern_bits & groups_per_block != 0).
    constexpr uint32_t max_bits_per_virtual_thread =
      is_cache_sectorized ? max_bits_per_group * groups_per_vertical_layout
                          : max_bits_per_word * VerticalLayout;

    pattern_array_t pattern_array{0};
    if constexpr (num_iterations == 1) {
      thread_dispatch<max_bits_per_virtual_thread, lower_bound, upper_bound>(
        hash, thread_index, pattern_array);
    } else {
      const uint32_t virtual_thread_index = LoopIndex * HorizontalLayout + thread_index;
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
      // if (threadIdx.x == 0) { printf("Salt Idx: %u\n", SaltIndex); }
      const uint32_t bit_index =
        (cuda::std::get<SaltIndex>(salts) * hash) >> (32 - bit_index_width);

      // Set the bit in the pattern array.
      cuda::std::get<PatternArrayIndex>(pattern_array) |= word_type{1} << bit_index;

      // Recurse.
      constexpr uint32_t next_salt_index = SaltIndex + 1;
      constexpr uint32_t next_pattern_array_index =
        is_cache_sectorized
          ? PatternArrayIndex + (next_salt_index % max_bits_per_group == 0 ? 1 : 0)
          : PatternArrayIndex + (next_salt_index % max_bits_per_word == 0 ? 1 : 0);
      set_bits<next_salt_index, SaltEndIndex, next_pattern_array_index>(hash, pattern_array);
    }
  }
};

}  // namespace cuco::detail