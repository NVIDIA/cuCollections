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

namespace cuco {
namespace experimental {

template <typename T>
T* move_vector_to_device(std::vector<T>& host_vector, thrust::device_vector<T>& device_vector) {
  device_vector = host_vector;
  host_vector.clear();
  return thrust::raw_pointer_cast(device_vector.data());
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Allocator,
          class Storage>
bit_vector<Key, Extent, Scope, Allocator, Storage>::bit_vector(Extent capacity)
    : words(), ranks(), selects(), n_bits(0), storage_{make_valid_extent<cg_size, window_size>(capacity), allocator_} {
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Allocator,
          class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::add(bool bit) {
  if (n_bits % 256 == 0) {
    words.resize((n_bits + 256) / 64);
  }
  set(n_bits, bit);
  ++n_bits;
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Allocator,
          class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::build() {
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
        uint64_t n_pops = __builtin_popcountll(word);
        uint64_t new_n_ones = n_ones + n_pops;
        if (((n_ones + 255) / 256) != ((new_n_ones + 255) / 256)) {
          uint64_t count = n_ones;
          while (word != 0) {
            uint64_t pos = __builtin_ctzll(word);
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
        uint64_t n_pops = __builtin_popcountll(word);
        uint64_t new_n_zeroes = n_zeroes + n_pops;
        if (((n_zeroes + 255) / 256) != ((new_n_zeroes + 255) / 256)) {
          uint64_t count = n_zeroes;
          while (word != 0) {
            uint64_t pos = __builtin_ctzll(word);
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

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Allocator,
          class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::set(Key i, bool bit) {
  if (bit) {
    words[i / 64] |= (1UL << (i % 64));
  } else {
    words[i / 64] &= ~(1UL << (i % 64));
  }
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Allocator,
          class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::set_last(bool bit) {
  set(n_bits - 1, bit);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Allocator,
          class Storage>
size_t bit_vector<Key, Extent, Scope, Allocator, Storage>::memory_footprint() const {
  return sizeof(uint64_t) * words.size() + sizeof(Rank) * (ranks.size() + ranks0.size()) +
         sizeof(uint32_t) * (selects.size() + selects0.size());
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Allocator,
          class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::move_to_device() {
  d_words_ptr = move_vector_to_device(words, d_words);
  d_ranks_ptr = move_vector_to_device(ranks, d_ranks);
  d_ranks0_ptr = move_vector_to_device(ranks, d_ranks);

  num_selects = selects.size();
  d_selects_ptr = move_vector_to_device(selects, d_selects);
  num_selects0 = selects0.size();
  d_selects0_ptr = move_vector_to_device(selects0, d_selects0);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Allocator,
          class Storage>
template <typename... Operators>
auto bit_vector<Key, Extent, Scope, Allocator, Storage>::ref(
  Operators...) const noexcept
{
  static_assert(sizeof...(Operators), "No operators specified");
  return ref_type<Operators...>{d_words_ptr, d_ranks_ptr, d_selects_ptr, num_selects};
}

}  // namespace experimental
}  // namespace cuco

