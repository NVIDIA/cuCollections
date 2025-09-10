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

#include <cuda/std/bit>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

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
  static constexpr std::uint32_t min_bits_per_word = words_per_block;
  static constexpr std::uint32_t remainder_bits    = pattern_bits % words_per_block;

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
  }

  __device__ constexpr hash_result_type hash(hash_argument_type const& key) const
  {
    return hash_(key);
  }

  template <class Extent>
  __device__ constexpr auto block_index(hash_result_type hash, Extent num_blocks) const
  {
    return hash % num_blocks;
  }

  __device__ constexpr word_type word_pattern(hash_result_type hash, std::uint32_t word_index) const
  {
    word_type constexpr bit_index_mask = (word_type{1} << bit_index_width) - 1;

    auto const bits_so_far =
      min_bits_per_word * word_index +
      (remainder_bits == 0 ? 0 : (word_index < remainder_bits ? word_index : remainder_bits));

    hash >>= bits_so_far * bit_index_width;

    word_type word = 0;
    int32_t const bits_per_word =
      min_bits_per_word + (remainder_bits == 0 ? 0 : (word_index < remainder_bits ? 1 : 0));

    for (int32_t bit = 0; bit < bits_per_word; ++bit) {
      word |= word_type{1} << (hash & bit_index_mask);
      hash >>= bit_index_width;
    }

    return word;
  }

 private:
  hasher hash_;
};

}  // namespace cuco::experimental::detail