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
bit_vector<Key, Extent, Scope, Allocator, Storage>::bit_vector()
  : words_(),
    ranks_(),
    selects_(),
    n_bits_(0),
    aow_words_(nullptr),
    aow_ranks_(nullptr),
    aow_selects_(nullptr),
    aow_ranks0_(nullptr),
    aow_selects0_(nullptr)
{
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
bit_vector<Key, Extent, Scope, Allocator, Storage>::~bit_vector()
{
  delete aow_words_;
  delete aow_ranks_;
  delete aow_selects_;
  delete aow_ranks0_;
  delete aow_selects0_;
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::add(bool bit)
{
  if (n_bits_ % 256 == 0) { words_.resize((n_bits_ + 256) / 64); }
  set(n_bits_, bit);
  ++n_bits_;
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::build()
{
  uint64_t n_blocks = words_.size() / 4;
  ranks_.resize(n_blocks + 1);
  ranks0_.resize(n_blocks + 1);

  uint64_t n_ones = 0, n_zeroes = 0;
  for (uint64_t block_id = 0; block_id < n_blocks; ++block_id) {
    ranks_[block_id].set_abs(n_ones);
    ranks0_[block_id].set_abs(n_zeroes);

    for (uint64_t block_offset = 0; block_offset < 4; ++block_offset) {
      if (block_offset != 0) {
        ranks_[block_id].rels_[block_offset - 1]  = n_ones - ranks_[block_id].abs();
        ranks0_[block_id].rels_[block_offset - 1] = n_zeroes - ranks0_[block_id].abs();
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
      update_selects(word_id, words_[word_id], n_ones, selects_);
      update_selects(word_id, ~words_[word_id], n_zeroes, selects0_);
    }
  }

  ranks_.back().set_abs(n_ones);
  ranks0_.back().set_abs(n_zeroes);
  selects_.push_back(words_.size() * 64 / 256);
  selects0_.push_back(words_.size() * 64 / 256);

  move_to_device();
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::set(Key key, bool bit)
{
  if (bit) {
    words_[key / 64] |= (1UL << (key % 64));
  } else {
    words_[key / 64] &= ~(1UL << (key % 64));
  }
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::set_last(bool bit)
{
  set(n_bits_ - 1, bit);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
size_t bit_vector<Key, Extent, Scope, Allocator, Storage>::memory_footprint() const
{
  return sizeof(uint64_t) * words_.size() + sizeof(rank) * (ranks_.size() + ranks0_.size()) +
         sizeof(uint64_t) * (selects_.size() + selects0_.size());
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
void initialize_aow(Storage* storage, T* ptr, uint64_t num_elements)
{
  auto constexpr stride = 4;
  auto const grid_size  = (num_elements + stride * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
                         (stride * detail::CUCO_DEFAULT_BLOCK_SIZE);

  copy_to_window<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE>>>(
    storage->data(), num_elements, ptr);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
template <class T>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::copy_host_array_to_aow(
  storage_type** aow, std::vector<T>& host_array)
{
  thrust::device_vector<T> device_array = host_array;
  auto device_ptr                       = (uint64_t*)thrust::raw_pointer_cast(device_array.data());

  uint64_t num_elements = host_array.size();
  host_array.clear();

  *aow = new storage_type(make_valid_extent<cg_size, window_size>(extent<size_t>{num_elements + 1}),
                          allocator_);
  initialize_aow(*aow, device_ptr, num_elements);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
void bit_vector<Key, Extent, Scope, Allocator, Storage>::move_to_device()
{
  copy_host_array_to_aow(&aow_words_, words_);
  copy_host_array_to_aow(&aow_ranks_, ranks_);
  copy_host_array_to_aow(&aow_selects_, selects_);
  copy_host_array_to_aow(&aow_ranks0_, ranks0_);
  copy_host_array_to_aow(&aow_selects0_, selects0_);
}

template <class Key, class Extent, cuda::thread_scope Scope, class Allocator, class Storage>
template <typename... Operators>
auto bit_vector<Key, Extent, Scope, Allocator, Storage>::ref(Operators...) const noexcept
{
  static_assert(sizeof...(Operators), "No operators specified");
  return ref_type<Operators...>{aow_words_->ref(),
                                aow_ranks_->ref(),
                                aow_selects_->ref(),
                                aow_ranks0_->ref(),
                                aow_selects0_->ref()};
}

}  // namespace experimental
}  // namespace cuco
