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

template <class Allocator>
bit_vector<Allocator>::bit_vector(Allocator const& allocator)
  : words_(),
    ranks_(),
    selects_(),
    n_bits_(0),
    allocator_(allocator),
    aow_words_(nullptr),
    aow_ranks_(nullptr),
    aow_selects_(nullptr),
    aow_ranks0_(nullptr),
    aow_selects0_(nullptr)
{
}

template <class Allocator>
bit_vector<Allocator>::~bit_vector()
{
  delete aow_words_;
  delete aow_ranks_;
  delete aow_selects_;
  delete aow_ranks0_;
  delete aow_selects0_;
}

template <class Allocator>
void bit_vector<Allocator>::append(bool bit) noexcept
{
  if (n_bits_ % 256 == 0) { words_.resize((n_bits_ + 256) / 64); }  // Extend by four 64-bit words
  set(n_bits_, bit);
  ++n_bits_;
}

inline void update_selects(uint64_t word_id,
                           uint64_t word,
                           uint64_t& gcount,
                           std::vector<uint64_t>& selects) noexcept
{
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
}

inline void build_ranks_and_selects(const std::vector<uint64_t>& words,
                                    std::vector<rank>& ranks,
                                    std::vector<uint64_t>& selects,
                                    bool flip_bits) noexcept
{
  uint64_t n_blocks = words.size() / 4;  // Each block has four 64-bit words
  ranks.resize(n_blocks + 1);

  uint64_t count = 0;
  for (uint64_t block_id = 0; block_id < n_blocks; ++block_id) {
    ranks[block_id].set_abs(count);

    for (uint64_t block_offset = 0; block_offset < 4; ++block_offset) {
      if (block_offset != 0) {  // Compute the deltas
        ranks[block_id].rels_[block_offset - 1] = count - ranks[block_id].abs();
      }

      uint64_t word_id = (block_id * 4) + block_offset;
      auto word        = flip_bits ? ~words[word_id] : words[word_id];
      update_selects(word_id, word, count, selects);  // Will update count
    }
  }

  ranks.back().set_abs(count);
  selects.push_back(words.size() * 64 / 256);
}

template <class Allocator>
void bit_vector<Allocator>::build() noexcept
{
  build_ranks_and_selects(words_, ranks_, selects_, false);   // 1-bits
  build_ranks_and_selects(words_, ranks0_, selects0_, true);  // 0-bits

  move_to_device();
}

template <class Allocator>
void bit_vector<Allocator>::set(size_type index, bool bit) noexcept
{
  if (bit) {
    words_[index / 64] |= (1UL << (index % 64));
  } else {
    words_[index / 64] &= ~(1UL << (index % 64));
  }
}

template <class Allocator>
void bit_vector<Allocator>::set_last(bool bit) noexcept
{
  set(n_bits_ - 1, bit);
}

// Copies device array to window structure
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
void initialize_aow(Storage* storage, thrust::device_vector<T>& device_array, uint64_t num_elements)
{
  auto constexpr stride = 4;
  auto const grid_size  = (num_elements + stride * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
                         (stride * detail::CUCO_DEFAULT_BLOCK_SIZE);

  auto device_ptr = reinterpret_cast<uint64_t*>(thrust::raw_pointer_cast(device_array.data()));
  copy_to_window<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE>>>(
    storage->data(), num_elements, device_ptr);
}

template <class Allocator>
template <class T>
void bit_vector<Allocator>::copy_host_array_to_aow(storage_type** aow,
                                                   std::vector<T>& host_array) noexcept
{
  uint64_t num_elements = host_array.size();
  *aow                  = new storage_type(extent<size_type>{num_elements + 1}, allocator_);

  if (num_elements > 0) {
    // Move host array to device memory
    thrust::device_vector<T> device_array = host_array;
    host_array.clear();

    // Copy device array to window structure
    initialize_aow(*aow, device_array, num_elements);
  }
}

template <class Allocator>
void bit_vector<Allocator>::move_to_device() noexcept
{
  copy_host_array_to_aow(&aow_words_, words_);
  copy_host_array_to_aow(&aow_ranks_, ranks_);
  copy_host_array_to_aow(&aow_selects_, selects_);
  copy_host_array_to_aow(&aow_ranks0_, ranks0_);
  copy_host_array_to_aow(&aow_selects0_, selects0_);
}

template <class Allocator>
template <typename... Operators>
auto bit_vector<Allocator>::ref(Operators...) const noexcept
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
