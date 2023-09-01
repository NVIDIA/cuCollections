/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda/std/bit>

namespace cuco {
namespace experimental {
namespace detail {

/*
 * @brief Gather bits of a range of keys
 *
 * @tparam BitvectorRef Bitvector reference type
 * @tparam KeyIt Device-accessible iterator to input keys
 * @tparam ValueIt Device-accessible iterator to values
 * @tparam size_type Size type
 *
 * @param ref Bitvector ref
 * @param keys Begin iterator to keys
 * @param outputs Begin iterator to outputs
 * @param num_keys Number of input keys
 */
template <typename BitvectorRef, typename KeyIt, typename ValueIt, typename size_type>
__global__ void bitvector_get_kernel(BitvectorRef ref,
                                     KeyIt keys,
                                     ValueIt outputs,
                                     size_type num_keys)
{
  uint32_t const loop_stride = gridDim.x * blockDim.x;
  uint32_t key_id            = blockDim.x * blockIdx.x + threadIdx.x;

  while (key_id < num_keys) {
    outputs[key_id] = ref.get(keys[key_id]);
    key_id += loop_stride;
  }
}

/*
 * @brief Gather rank values for a range of keys
 *
 * @tparam BitvectorRef Bitvector reference type
 * @tparam KeyIt Device-accessible iterator to input keys
 * @tparam ValueIt Device-accessible iterator to values
 * @tparam size_type Size type
 *
 * @param ref Bitvector ref
 * @param keys Begin iterator to keys
 * @param outputs Begin iterator to outputs
 * @param num_keys Number of input keys
 */
template <typename BitvectorRef, typename KeyIt, typename ValueIt, typename size_type>
__global__ void bitvector_rank_kernel(BitvectorRef ref,
                                      KeyIt keys,
                                      ValueIt outputs,
                                      size_type num_keys)
{
  uint32_t const loop_stride = gridDim.x * blockDim.x;
  uint32_t key_id            = blockDim.x * blockIdx.x + threadIdx.x;

  while (key_id < num_keys) {
    outputs[key_id] = ref.rank(keys[key_id]);
    key_id += loop_stride;
  }
}

/*
 * @brief Gather select values for a range of keys
 *
 * @tparam BitvectorRef Bitvector reference type
 * @tparam KeyIt Device-accessible iterator to input keys
 * @tparam ValueIt Device-accessible iterator to values
 * @tparam size_type Size type
 *
 * @param ref Bitvector ref
 * @param keys Begin iterator to keys
 * @param outputs Begin iterator to outputs
 * @param num_keys Number of input keys
 */
template <typename BitvectorRef, typename KeyIt, typename ValueIt, typename size_type>
__global__ void bitvector_select_kernel(BitvectorRef ref,
                                        KeyIt keys,
                                        ValueIt outputs,
                                        size_type num_keys)
{
  uint32_t const loop_stride = gridDim.x * blockDim.x;
  uint32_t key_id            = blockDim.x * blockIdx.x + threadIdx.x;

  while (key_id < num_keys) {
    outputs[key_id] = ref.select(keys[key_id]);
    key_id += loop_stride;
  }
}

/*
 * @brief Computes number of set or not-set bits in each word
 *
 * @tparam slot_type Word type
 * @tparam size_type Size type
 *
 * @param words Input array of words
 * @param bit_counts Output array of per-word bit counts
 * @param num_words Number of words
 * @param flip_bits Boolean to request negation of words before counting bits
 */
template <typename slot_type, typename size_type>
__global__ void bit_counts_kernel(const slot_type* words,
                                  size_type* bit_counts,
                                  size_type num_words,
                                  bool flip_bits)
{
  size_type word_id = blockDim.x * blockIdx.x + threadIdx.x;
  size_type stride  = gridDim.x * blockDim.x;

  while (word_id < num_words) {
    auto word           = words[word_id];
    bit_counts[word_id] = cuda::std::popcount(flip_bits ? ~word : word);
    word_id += stride;
  }
}

/*
 * @brief Compute rank values at block size intervals.
 *
 * ranks[i] = Number of set bits in [0, i) range
 * This kernel transforms prefix sum array of per-word bit counts
 * into base-delta encoding style of `rank` struct.
 * Since prefix sum is available, there are no dependencies across blocks.

 * @tparam size_type Size type
 *
 * @param prefix_bit_counts Prefix sum array of per-word bit counts
 * @param ranks Output array of ranks
 * @param num_words Length of input array
 * @param num_blocks Length of ouput array
 * @param words_per_block Number of words in each block
 */
template <typename size_type>
__global__ void encode_ranks_from_prefix_bit_counts(const size_type* prefix_bit_counts,
                                                    rank* ranks,
                                                    size_type num_words,
                                                    size_type num_blocks,
                                                    size_type words_per_block)
{
  size_type rank_id = blockDim.x * blockIdx.x + threadIdx.x;
  size_type stride  = gridDim.x * blockDim.x;

  while (rank_id < num_blocks) {
    size_type word_id = rank_id * words_per_block;

    // Set base value of rank
    auto& rank = ranks[rank_id];
    rank.set_base(prefix_bit_counts[word_id]);

    if (rank_id < num_blocks - 1) {
      // For each subsequent word in this block, compute deltas from base
      for (size_type block_offset = 0; block_offset < words_per_block - 1; block_offset++) {
        auto delta = prefix_bit_counts[word_id + block_offset + 1] - prefix_bit_counts[word_id];
        rank.offsets_[block_offset] = delta;
      }
    }
    rank_id += stride;
  }
}

/*
 * @brief Compute select values at block size intervals.
 *
 * selects[i] = Position of (i+ 1)th set bit
 * This kernel check for blocks where prefix sum crosses a multiple of `bits_per_block`.
 * Such blocks are marked in the output boolean array
 *
 * @tparam size_type Size type
 *
 * @param prefix_bit_counts Prefix sum array of per-word bit counts
 * @param selects_markers Ouput array indicating whether a block has selects entry or not
 * @param num_blocks Length of ouput array
 * @param words_per_block Number of words in each block
 * @param bits_per_block Number of bits in each block
 */
template <typename size_type>
__global__ void mark_blocks_with_select_entries(const size_type* prefix_bit_counts,
                                                size_type* select_markers,
                                                size_type num_blocks,
                                                size_type words_per_block,
                                                size_type bits_per_block)
{
  size_type block_id = blockDim.x * blockIdx.x + threadIdx.x;
  size_type stride   = gridDim.x * blockDim.x;

  while (block_id < num_blocks) {
    if (block_id == 0) {  // Block 0 always has a selects entry
      select_markers[block_id] = 1;
      block_id += stride;
      continue;
    }

    select_markers[block_id] = 0;  // Always clear marker first
    size_type word_id        = block_id * words_per_block;
    size_type prev_count     = prefix_bit_counts[word_id];

    for (size_t block_offset = 1; block_offset <= words_per_block; block_offset++) {
      size_type count = prefix_bit_counts[word_id + block_offset];

      // Selects entry is added when cumulative bitcount crosses a multiple of bits_per_block
      if ((prev_count - 1) / bits_per_block != (count - 1) / bits_per_block) {
        select_markers[block_id] = 1;
        break;
      }
      prev_count = count;
    }

    block_id += stride;
  }
}

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
