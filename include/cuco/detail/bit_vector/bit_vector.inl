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

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
bit_vector<Key, Extent, Scope, Allocator, Storage>::bit_vector(Extent capacity)
  : words(),
    ranks(),
    selects(),
    n_bits(0),
    aow_words{make_valid_extent<cg_size, window_size>(capacity), allocator_},
    aow_ranks{make_valid_extent<cg_size, window_size>(capacity), allocator_},
    aow_selects{make_valid_extent<cg_size, window_size>(capacity), allocator_},
    aow_ranks0{make_valid_extent<cg_size, window_size>(capacity), allocator_},
    aow_selects0{make_valid_extent<cg_size, window_size>(capacity), allocator_}
{
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::add(bool bit)
{
  if (n_bits % 256 == 0) { words.resize((n_bits + 256) / 64); }
  set(n_bits, bit);
  ++n_bits;
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::build()
{
  uint64_t n_blocks = words.size() / 4;
  ranks.resize(n_blocks + 1);
  ranks0.resize(n_blocks + 1);

  uint64_t n_ones = 0, n_zeroes = 0;
  for (uint64_t block_id = 0; block_id < n_blocks; ++block_id) {
    ranks[block_id].set_abs(n_ones);
    ranks0[block_id].set_abs(n_zeroes);

    for (uint64_t block_offset = 0; block_offset < 4; ++block_offset) {
      if (block_offset != 0) {
        ranks[block_id].rels[block_offset - 1]  = n_ones - ranks[block_id].abs();
        ranks0[block_id].rels[block_offset - 1] = n_zeroes - ranks0[block_id].abs();
      }

      auto update_selects =
        [](uint64_t word_id, uint64_t word, uint64_t& gcount, std::vector<uint64_t>& selects) {
          uint64_t n_pops     = __builtin_popcountll(word);
          uint64_t new_gcount = gcount + n_pops;
          if (((gcount + 255) / 256) != ((new_gcount + 255) / 256)) {
            uint64_t count = gcount;
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
          gcount = new_gcount;
        };

      uint64_t word_id = (block_id * 4) + block_offset;
      update_selects(word_id, words[word_id], n_ones, selects);
      update_selects(word_id, ~words[word_id], n_zeroes, selects0);
    }
  }

  ranks.back().set_abs(n_ones);
  ranks0.back().set_abs(n_zeroes);
  selects.push_back(words.size() * 64 / 256);
  selects0.push_back(words.size() * 64 / 256);

  move_to_device();
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::set(Key i, bool bit)
{
  if (bit) {
    words[i / 64] |= (1UL << (i % 64));
  } else {
    words[i / 64] &= ~(1UL << (i % 64));
  }
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::set_last(bool bit)
{
  set(n_bits - 1, bit);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
size_t bit_vector<Key, Extent, Scope, Allocator, Storage>::memory_footprint() const
{
  return sizeof(uint64_t) * words.size() + sizeof(Rank) * (ranks.size() + ranks0.size()) +
         sizeof(uint64_t) * (selects.size() + selects0.size());
}

template <typename WindowT, class T>
__global__ void copy_to_window(WindowT* windows, cuco::detail::index_type n, T* values)
{
  cuco::detail::index_type const loop_stride = gridDim.x * blockDim.x;
  cuco::detail::index_type idx               = blockDim.x * blockIdx.x + threadIdx.x;

  while (idx < n) {
    auto& window_slots = *(windows + idx);
    window_slots[0]    = values[idx];
    idx += loop_stride;
  }
}

template <class Storage, class T>
void initialize_aow(Storage& storage, T* ptr, uint64_t num_elements)
{
  auto constexpr stride = 4;
  auto const grid_size  = (num_elements + stride * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
                         (stride * detail::CUCO_DEFAULT_BLOCK_SIZE);

  copy_to_window<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE>>>(storage.data(), num_elements, ptr);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
template <class T>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::copy_host_array_to_aow(
  storage_type& aow, std::vector<T>& host_array)
{
  thrust::device_vector<T> device_array = host_array;
  auto device_ptr                       = (uint64_t*)thrust::raw_pointer_cast(device_array.data());

  uint64_t num_elements = host_array.size();
  host_array.clear();

  initialize_aow(aow, device_ptr, num_elements);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::move_to_device()
{
  copy_host_array_to_aow(aow_words, words);
  copy_host_array_to_aow(aow_ranks, ranks);
  copy_host_array_to_aow(aow_selects, selects);
  copy_host_array_to_aow(aow_ranks0, ranks0);
  copy_host_array_to_aow(aow_selects0, selects0);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
template <typename... Operators>
auto bit_vector<Key, Extent, Scope, Allocator, Storage>::ref(Operators...) const noexcept
{
  static_assert(sizeof...(Operators), "No operators specified");
  return ref_type<Operators...>{
    aow_words.ref(), aow_ranks.ref(), aow_selects.ref(), aow_ranks0.ref(), aow_selects0.ref()};
}

}  // namespace experimental
}  // namespace cuco
