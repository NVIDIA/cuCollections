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

namespace cuco {
namespace experimental {

template <typename T>
T* move_vector_to_device(std::vector<T>& host_vector, thrust::device_vector<T>& device_vector) {
  device_vector = host_vector;
  host_vector.clear();
  return thrust::raw_pointer_cast(device_vector.data());
}

bit_vector::bit_vector() : words(), ranks(), selects(), n_bits(0) {}

void bit_vector::add(Key bit) {
  if (n_bits % 256 == 0) {
    words.resize((n_bits + 256) / 64);
  }
  set(n_bits, bit);
  ++n_bits;
}

void bit_vector::build() {
  uint64_t n_blocks = words.size() / 4;
  uint64_t n_ones = 0, n_zeroes = 0;
  ranks.resize(n_blocks + 1);
  ranks0.resize(n_blocks + 1);
  for (uint64_t block_id = 0; block_id < n_blocks; ++block_id) {
    ranks[block_id].set_abs(n_ones);
    ranks0[block_id].set_abs(n_zeroes);
    for (uint64_t j = 0; j < 4; ++j) {
      if (j != 0) {
        uint64_t rel1 = n_ones - ranks[block_id].abs();
        ranks[block_id].rels[j - 1] = rel1;

        uint64_t rel0 = n_zeroes - ranks0[block_id].abs();
        ranks0[block_id].rels[j - 1] = rel0;
      }

      uint64_t word_id = (block_id * 4) + j;
      {
        uint64_t word = words[word_id];
        uint64_t n_pops = Popcnt(word);
        uint64_t new_n_ones = n_ones + n_pops;
        if (((n_ones + 255) / 256) != ((new_n_ones + 255) / 256)) {
          uint64_t count = n_ones;
          while (word != 0) {
            uint64_t pos = Ctz(word);
            if (count % 256 == 0) {
              selects.push_back(((word_id * 64) + pos) / 256);
              break;
            }
            word ^= 1UL << pos;
            ++count;
          }
        }
        n_ones = new_n_ones;
      }
      {
        uint64_t word = ~words[word_id];
        uint64_t n_pops = Popcnt(word);
        uint64_t new_n_zeroes = n_zeroes + n_pops;
        if (((n_zeroes + 255) / 256) != ((new_n_zeroes + 255) / 256)) {
          uint64_t count = n_zeroes;
          while (word != 0) {
            uint64_t pos = Ctz(word);
            if (count % 256 == 0) {
              selects0.push_back(((word_id * 64) + pos) / 256);
              break;
            }
            word ^= 1UL << pos;
            ++count;
          }
        }
        n_zeroes = new_n_zeroes;
      }
    }
  }
  ranks.back().set_abs(n_ones);
  ranks0.back().set_abs(n_zeroes);
  selects.push_back(words.size() * 64 / 256);
  selects0.push_back(words.size() * 64 / 256);

  move_to_device();
}

__device__ uint64_t bit_vector::get(Key i) const { return (d_words_ptr[i / 64] >> (i % 64)) & 1UL; }

void bit_vector::set(Key i, bool bit) {
  if (bit) {
    words[i / 64] |= (1UL << (i % 64));
  } else {
    words[i / 64] &= ~(1UL << (i % 64));
  }
}

void bit_vector::set_last(bool bit) {
  set(n_bits - 1, bit);
}

__device__ uint64_t bit_vector::rank(Key i) const {
  uint64_t word_id = i / 64;
  uint64_t bit_id = i % 64;
  uint64_t rank_id = word_id / 4;
  uint64_t rel_id = word_id % 4;
  uint64_t n = d_ranks_ptr[rank_id].abs();
  if (rel_id != 0) {
    n += d_ranks_ptr[rank_id].rels[rel_id - 1];
  }
  n += __popcll(d_words_ptr[word_id] & ((1UL << bit_id) - 1));
  return n;
}

__device__ uint64_t bit_vector::select(Key i) const {
  const uint64_t block_id = i / 256;
  uint64_t begin = d_selects_ptr[block_id];
  uint64_t end = d_selects_ptr[block_id + 1] + 1UL;
  if (begin + 10 >= end) {
    while (i >= d_ranks_ptr[begin + 1].abs()) {
      ++begin;
    }
  } else {
    while (begin + 1 < end) {
      const uint64_t middle = (begin + end) / 2;
      if (i < d_ranks_ptr[middle].abs()) {
        end = middle;
      } else {
        begin = middle;
      }
    }
  }
  const uint64_t rank_id = begin;
  const auto& rank = d_ranks_ptr[rank_id];
  i -= rank.abs();

  uint64_t word_id = rank_id * 4;
  bool a0 = i >= rank.rels[0];
  bool a1 = i >= rank.rels[1];
  bool a2 = i >= rank.rels[2];

  uint32_t inc = a0 + a1 + a2;
  word_id += inc;
  i -= (inc > 0) * rank.rels[inc - (inc > 0)];

  return (word_id * 64) + ith_set_pos(i, d_words_ptr[word_id]);
}

__device__ uint64_t bit_vector::select0(Key i) const {
  const uint64_t block_id = i / 256;
  uint64_t begin = d_selects0_ptr[block_id];
  uint64_t end = d_selects0_ptr[block_id + 1] + 1UL;
  if (begin + 10 >= end) {
    while (i >= d_ranks0_ptr[begin + 1].abs()) {
      ++begin;
    }
  } else {
    while (begin + 1 < end) {
      const uint64_t middle = (begin + end) / 2;
      if (i < d_ranks0_ptr[middle].abs()) {
        end = middle;
      } else {
        begin = middle;
      }
    }
  }
  const uint64_t rank_id = begin;
  const auto& rank = d_ranks0_ptr[rank_id];
  i -= rank.abs();

  uint64_t word_id = rank_id * 4;
  bool a0 = i >= rank.rels[0];
  bool a1 = i >= rank.rels[1];
  bool a2 = i >= rank.rels[2];

  uint32_t inc = a0 + a1 + a2;
  word_id += inc;
  i -= (inc > 0) * rank.rels[inc - (inc > 0)];

  return (word_id * 64) + ith_set_pos(i, ~d_words_ptr[word_id]);
}

__device__ uint64_t bit_vector::find_next_set(Key i) const {
  uint64_t word_id = i / 64;
  uint64_t bit_id = i % 64;
  uint64_t word = d_words_ptr[word_id];
  word &= ~(0lu) << bit_id;
  while (word == 0) {
    word = d_words_ptr[++word_id];
  }
  return (word_id * 64) + __builtin_ffsll(word) - 1;
}

size_t bit_vector::size() const {
  return n_bits;
}

size_t bit_vector::memory_footprint() const {
  return sizeof(uint64_t) * words.size() + sizeof(Rank) * (ranks.size() + ranks0.size()) +
         sizeof(uint32_t) * (selects.size() + selects0.size());
}

void bit_vector::move_to_device() {
  d_words_ptr = move_vector_to_device(words, d_words);
  d_ranks_ptr = move_vector_to_device(ranks, d_ranks);
  d_ranks0_ptr = move_vector_to_device(ranks, d_ranks);

  num_selects = selects.size();
  d_selects_ptr = move_vector_to_device(selects, d_selects);
  num_selects0 = selects0.size();
  d_selects0_ptr = move_vector_to_device(selects0, d_selects0);
}

}  // namespace experimental
}  // namespace cuco

