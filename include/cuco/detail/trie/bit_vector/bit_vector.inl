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

#include <cuco/detail/trie/bit_vector/kernels.cuh>
#include <cuco/detail/tuning.cuh>
#include <cuco/detail/utils.hpp>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <cuda/std/bit>

namespace cuco {
namespace experimental {
namespace detail {

template <class Allocator>
bit_vector<Allocator>::bit_vector(Allocator const& allocator)
  : allocator_{allocator},
    n_bits_{0},
    words_{allocator},
    ranks_{allocator},
    ranks0_{allocator},
    selects_{allocator},
    selects0_{allocator}
{
}

template <class Allocator>
bit_vector<Allocator>::~bit_vector()
{
}

template <class Allocator>
void bit_vector<Allocator>::append(bool bit) noexcept
{
  if (n_bits_ % bits_per_block == 0) {
    words_.resize(words_.size() + words_per_block);  // Extend storage by one block
  }

  set(n_bits_++, bit);
}

template <class Allocator>
void bit_vector<Allocator>::set(size_type index, bool bit) noexcept
{
  size_type word_id = index / bits_per_word;
  size_type bit_id  = index % bits_per_word;
  if (bit) {
    words_[word_id] |= 1UL << bit_id;
  } else {
    words_[word_id] &= ~(1UL << bit_id);
  }
}

template <class Allocator>
void bit_vector<Allocator>::set_last(bool bit) noexcept
{
  set(n_bits_ - 1, bit);
}

template <class Allocator>
template <typename KeyIt, typename OutputIt>
void bit_vector<Allocator>::get(KeyIt keys_begin,
                                KeyIt keys_end,
                                OutputIt outputs_begin,
                                cuda_stream_ref stream) const noexcept

{
  auto const num_keys = cuco::detail::distance(keys_begin, keys_end);
  if (num_keys == 0) { return; }

  auto grid_size = default_grid_size(num_keys);

  bitvector_get_kernel<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
    ref(), keys_begin, outputs_begin, num_keys);
}

template <class Allocator>
template <typename KeyIt, typename OutputIt>
void bit_vector<Allocator>::ranks(KeyIt keys_begin,
                                  KeyIt keys_end,
                                  OutputIt outputs_begin,
                                  cuda_stream_ref stream) const noexcept

{
  auto const num_keys = cuco::detail::distance(keys_begin, keys_end);
  if (num_keys == 0) { return; }

  auto grid_size = default_grid_size(num_keys);

  bitvector_rank_kernel<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
    ref(), keys_begin, outputs_begin, num_keys);
}

template <class Allocator>
template <typename KeyIt, typename OutputIt>
void bit_vector<Allocator>::selects(KeyIt keys_begin,
                                    KeyIt keys_end,
                                    OutputIt outputs_begin,
                                    cuda_stream_ref stream) const noexcept

{
  auto const num_keys = cuco::detail::distance(keys_begin, keys_end);
  if (num_keys == 0) { return; }

  auto grid_size = default_grid_size(num_keys);

  bitvector_select_kernel<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
    ref(), keys_begin, outputs_begin, num_keys);
}

template <class Allocator>
void bit_vector<Allocator>::build_ranks_and_selects(
  thrust::device_vector<rank, rank_allocator_type>& ranks,
  thrust::device_vector<size_type, size_allocator_type>& selects,
  bool flip_bits) noexcept
{
  if (n_bits_ == 0) { return; }

  // Step 1. Compute prefix sum of per-word bit counts
  // Population counts for each word
  // Sized to have one extra entry for subsequent prefix sum
  size_type num_words = words_.size();
  thrust::device_vector<size_type> bit_counts(num_words + 1);
  auto grid_size = default_grid_size(num_words);
  bit_counts_kernel<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE>>>(
    thrust::raw_pointer_cast(words_.data()),
    thrust::raw_pointer_cast(bit_counts.data()),
    num_words,
    flip_bits);

  thrust::exclusive_scan(thrust::device, bit_counts.begin(), bit_counts.end(), bit_counts.begin());

  // Step 2. Compute ranks
  size_type num_blocks = (num_words - 1) / words_per_block + 2;
  ranks.resize(num_blocks);

  grid_size = default_grid_size(num_blocks);
  encode_ranks_from_prefix_bit_counts<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE>>>(
    thrust::raw_pointer_cast(bit_counts.data()),
    thrust::raw_pointer_cast(ranks.data()),
    num_words,
    num_blocks,
    words_per_block);

  // Step 3. Compute selects
  thrust::device_vector<size_type> select_markers(num_blocks);
  mark_blocks_with_select_entries<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE>>>(
    thrust::raw_pointer_cast(bit_counts.data()),
    thrust::raw_pointer_cast(select_markers.data()),
    num_blocks,
    words_per_block,
    bits_per_block);

  size_type num_selects =
    thrust::reduce(thrust::device, select_markers.begin(), select_markers.end());
  selects.resize(num_selects);

  // Generate indices of non-zeros in select_markers
  thrust::copy_if(thrust::device,
                  thrust::make_counting_iterator(0lu),
                  thrust::make_counting_iterator(num_blocks),
                  select_markers.begin(),
                  selects.begin(),
                  thrust::identity());
}

template <class Allocator>
void bit_vector<Allocator>::build() noexcept
{
  build_ranks_and_selects(ranks_, selects_, false);   // 1-bits
  build_ranks_and_selects(ranks0_, selects0_, true);  // 0-bits
}

template <class Allocator>
bit_vector<Allocator>::ref_type bit_vector<Allocator>::ref() const noexcept
{
  return ref_type{storage_ref_type{thrust::raw_pointer_cast(words_.data()),
                                   thrust::raw_pointer_cast(ranks_.data()),
                                   thrust::raw_pointer_cast(selects_.data()),
                                   thrust::raw_pointer_cast(ranks0_.data()),
                                   thrust::raw_pointer_cast(selects0_.data())}};
}

template <class Allocator>
constexpr bit_vector<Allocator>::size_type bit_vector<Allocator>::size() const noexcept
{
  return n_bits_;
}

template <class Allocator>
constexpr bit_vector<Allocator>::size_type bit_vector<Allocator>::default_grid_size(
  size_type num_elements) const noexcept
{
  return (num_elements - 1) / (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE) + 1;
}

// Device reference implementations
template <class Allocator>
__host__ __device__ constexpr bit_vector<Allocator>::reference::reference(
  storage_ref_type storage) noexcept
  : storage_{storage}
{
}

template <class Allocator>
__device__ bool bit_vector<Allocator>::reference::get(size_type key) const noexcept
{
  return (storage_.words_ref_[key / bits_per_word] >> (key % bits_per_word)) & 1UL;
}

template <class Allocator>
__device__ typename bit_vector<Allocator>::slot_type bit_vector<Allocator>::reference::get_word(
  size_type word_id) const noexcept
{
  return storage_.words_ref_[word_id];
}

template <class Allocator>
__device__ typename bit_vector<Allocator>::size_type
bit_vector<Allocator>::reference::find_next_set(size_type key) const noexcept
{
  size_type word_id = key / bits_per_word;
  size_type bit_id  = key % bits_per_word;
  slot_type word    = storage_.words_ref_[word_id];
  word &= ~(0lu) << bit_id;
  while (word == 0) {
    word = storage_.words_ref_[++word_id];
  }
  return word_id * bits_per_word + __ffsll(word) - 1;  // cuda intrinsic
}

template <class Allocator>
__device__ typename bit_vector<Allocator>::size_type bit_vector<Allocator>::reference::rank(
  size_type key) const noexcept
{
  size_type word_id = key / bits_per_word;
  size_type bit_id  = key % bits_per_word;
  size_type rank_id = word_id / words_per_block;
  size_type rel_id  = word_id % words_per_block;

  auto rank   = storage_.ranks_ref_[rank_id];
  size_type n = rank.abs();

  if (rel_id != 0) { n += rank.rels_[rel_id - 1]; }

  n += cuda::std::popcount(storage_.words_ref_[word_id] & ((1UL << bit_id) - 1));

  return n;
}

template <class Allocator>
__device__ typename bit_vector<Allocator>::size_type bit_vector<Allocator>::reference::select(
  size_type count) const noexcept
{
  auto rank_id = get_initial_rank_estimate(count, storage_.selects_ref_, storage_.ranks_ref_);
  auto rank    = storage_.ranks_ref_[rank_id];

  size_type word_id = rank_id * words_per_block;
  word_id += subtract_rank_from_count(count, rank);

  return word_id * bits_per_word + select_bit_in_word(count, storage_.words_ref_[word_id]);
}

template <class Allocator>
__device__ typename bit_vector<Allocator>::size_type bit_vector<Allocator>::reference::select0(
  size_type count) const noexcept
{
  auto rank_id = get_initial_rank_estimate(count, storage_.selects0_ref_, storage_.ranks0_ref_);
  auto rank    = storage_.ranks0_ref_[rank_id];

  size_type word_id = rank_id * words_per_block;
  word_id += subtract_rank_from_count(count, rank);

  return word_id * bits_per_word + select_bit_in_word(count, ~(storage_.words_ref_[word_id]));
}

template <class Allocator>
template <typename SelectsRef, typename RanksRef>
__device__ typename bit_vector<Allocator>::size_type
bit_vector<Allocator>::reference::get_initial_rank_estimate(size_type count,
                                                            SelectsRef const& selects,
                                                            RanksRef const& ranks) const noexcept
{
  size_type block_id = count / (bits_per_word * words_per_block);
  size_type begin    = selects[block_id];
  size_type end      = selects[block_id + 1] + 1UL;

  if (begin + 10 >= end) {  // Linear search
    while (count >= ranks[begin + 1].abs()) {
      ++begin;
    }
  } else {  // Binary search
    while (begin + 1 < end) {
      size_type middle = (begin + end) / 2;
      if (count < ranks[middle].abs()) {
        end = middle;
      } else {
        begin = middle;
      }
    }
  }
  return begin;
}

template <class Allocator>
template <typename Rank>
__device__ typename bit_vector<Allocator>::size_type
bit_vector<Allocator>::reference::subtract_rank_from_count(size_type& count,
                                                           Rank rank) const noexcept
{
  count -= rank.abs();

  bool a0       = count >= rank.rels_[0];
  bool a1       = count >= rank.rels_[1];
  bool a2       = count >= rank.rels_[2];
  size_type inc = a0 + a1 + a2;

  count -= (inc > 0) * rank.rels_[inc - (inc > 0)];

  return inc;
}

template <class Allocator>
__device__ typename bit_vector<Allocator>::size_type
bit_vector<Allocator>::reference::select_bit_in_word(size_type N, slot_type word) const noexcept
{
  for (size_type pos = 0; pos < N; pos++) {
    word &= word - 1;
  }
  return __ffsll(word & -word) - 1;  // cuda intrinsic
}
}  // namespace detail
}  // namespace experimental
}  // namespace cuco
