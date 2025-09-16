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

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <sys/types.h>

#include <cstdint>

namespace cuco::experimental::detail {

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
  using hasher             = Hash;
  using word_type          = Word;
  using hash_argument_type = typename hasher::argument_type;
  using hash_result_type   = decltype(std::declval<hasher>()(std::declval<hash_argument_type>()));

  static constexpr uint32_t words_per_block = WordsPerBlock;
  static constexpr uint32_t pattern_bits    = PatternBits;

  // TODO this could be expressed as two cuda::std::extents<uint32_t, HORIZONTAL, VERTICAL> instead
  static constexpr uint32_t add_horizontal_layout =
    AddHorizontalLayout;  ///< horizontal vectorization layout for add operation
  static constexpr uint32_t add_vertical_layout =
    AddVerticalLayout;  ///< vertical vectorization layout for add operation
  static constexpr uint32_t contains_horizontal_layout =
    ContainsHorizontalLayout;  ///< horizontal vectorization layout for contains operation
  static constexpr uint32_t contains_vertical_layout =
    ContainsVerticalLayout;  ///< vertical vectorization layout for contains operation

  static constexpr size_t max_filter_bytes  = cuda::std::numeric_limits<size_t>::max();
  static constexpr size_t max_filter_blocks = cuda::std::numeric_limits<size_t>::max();

 private:
  static constexpr std::uint32_t word_bits         = cuda::std::numeric_limits<word_type>::digits;
  static constexpr std::uint32_t bit_index_width   = cuda::std::bit_width(word_bits - 1);
  static constexpr std::uint32_t max_bits_per_word = cuda::ceil_div(pattern_bits, words_per_block);

  static constexpr cuda::std::array<uint32_t, 16> salts = {0x47b6137bU,
                                                           0x44974d91U,
                                                           0x8824ad5bU,
                                                           0xa2b7289dU,
                                                           0x705495c7U,
                                                           0x2df1424bU,
                                                           0x9efc4947U,
                                                           0x5c6bfb31U,
                                                           0x3a5d6b07U,
                                                           0x7f24a931U,
                                                           0x1b8c93edU,
                                                           0x61e24f7bU,
                                                           0xd35f8a5fU,
                                                           0xb9ac37b3U,
                                                           0xf26a19e1U,
                                                           0x8e3b7d9fU};

 public:
  __host__ __device__ constexpr parametric_filter_policy(Hash hash = {}) : hash_{hash}
  {  // This ensures each word in the block has at least one bit set; otherwise we would never
     // use some of the words
    constexpr uint32_t min_pattern_bits = words_per_block;

    // The maximum number of bits to be set for a key is capped by the total number of bits in
    // the filter block
    constexpr uint32_t max_pattern_bits = word_bits * words_per_block;

    constexpr uint32_t hash_bits = cuda::std::numeric_limits<hash_result_type>::digits;
    constexpr uint32_t max_pattern_bits_from_hash = hash_bits / bit_index_width;
    static_assert(pattern_bits <= max_pattern_bits_from_hash,
                  "hash_result_type too narrow to generate the requested number of pattern_bits");
    static_assert(pattern_bits >= min_pattern_bits,
                  "pattern_bits must be at least words_per_block");
    static_assert(pattern_bits <= max_pattern_bits,
                  "pattern_bits must be less than the total number of bits in a filter "
                  "block");
    /// KEVIN: Requiring 64b hash return type for now
    static_assert(cuda::std::is_same_v<hash_result_type, uint64_t>,
                  "currently only 64b hash_result_type is supported");
    /// KEVIN: We can increase the number of salts if needed
    static_assert(pattern_bits <= salts.size(),
                  "pattern_bits exceeds the number of available salts");
  }

  __device__ constexpr hash_result_type hash(hash_argument_type const& key) const
  {
    return hash_(key);
  }

  // Return {upper 32b, lower 32b} of 64b hash
  __device__ constexpr cuda::std::pair<uint32_t, uint32_t> split_hash(
    hash_argument_type const& key) const
  {
    uint64_t full_hash = hash_(key);
    return {static_cast<uint32_t>(full_hash >> 32), static_cast<uint32_t>(full_hash)};
  }

  template <class Extent>
  __device__ constexpr auto block_index(uint32_t upper_hash_value, Extent num_blocks) const
  {
    return upper_hash_value % num_blocks;
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
   * namely virtual_thread_index = LoopIndex * HorizontalLayout + thread_index, where LoopIndex is
   * the index of the outermost loop in the range:
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
    constexpr uint32_t salt_start_index          = max_bits_per_word * VerticalLayout * LoopIndex;
    constexpr uint32_t pattern_array_start_index = 0;
    set_bits<salt_start_index, pattern_array_start_index>(hash, pattern_array);
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
      const uint32_t virtual_thread_index = LoopIndex * HorizontalLayout + thread_index;
      thread_dispatch<max_bits_per_virtual_thread, lower_bound, upper_bound>(
        hash, virtual_thread_index, pattern_array);
    }
    return pattern_array;
  }

  // Dispatches a dynamic thread index to a static virtual thread index by building a compile-time
  // decision tree over the range [LowerBound, UpperBound) for the virtual thread index.
  // This method is only used when <add/contains>_horizontal_layout > 1.
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
      constexpr uint32_t salt_start_index          = MaxBitsPerVirtualThread * LowerBound;
      constexpr uint32_t pattern_array_start_index = 0;
      set_bits<salt_start_index, pattern_array_start_index>(hash, pattern_array);
    } else {
      // Recursive case: thread_index > LowerBound
      constexpr uint32_t mid = (LowerBound + UpperBound) / 2;
      if (thread_index < mid) {
        thread_dispatch<MaxBitsPerVirtualThread, LowerBound, mid>(hash, thread_index, pattern_array);
      } else {
        thread_dispatch<MaxBitsPerVirtualThread, mid, UpperBound>(hash, thread_index, pattern_array);
      }
    }
  }

  // Set bits in the pattern array using salts starting from SaltIndex.
  template <uint32_t SaltIndex, uint32_t PatternArrayIndex, class PatternArrayT>
  __device__ constexpr void set_bits(uint32_t hash, PatternArrayT& pattern_array) const
  {
    if constexpr (SaltIndex < pattern_bits) {
      // Select top bit_index_width bits from salted hash to determine the bit index.
      const uint32_t bit_index =
        (cuda::std::get<SaltIndex>(salts) * hash) >> (32 - bit_index_width);

      // Set the bit in the pattern array.
      cuda::std::get<PatternArrayIndex>(pattern_array) |= word_type{1} << bit_index;

      // Recurse.
      constexpr uint32_t next_salt_index = SaltIndex + 1;
      constexpr uint32_t next_pattern_array_index =
        PatternArrayIndex + (next_salt_index % max_bits_per_word == 0 ? 1 : 0);
      set_bits<next_salt_index, next_pattern_array_index>(hash, pattern_array);
    }
  }
};

}  // namespace cuco::experimental::detail