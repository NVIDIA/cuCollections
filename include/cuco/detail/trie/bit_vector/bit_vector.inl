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
  : words_{}, ranks_{}, ranks0_{}, selects_{}, selects0_{}, n_bits_{0}, allocator_{allocator}
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
    size_type new_n_bits  = n_bits_ + bits_per_block;  // Extend storage by one block
    size_type new_n_words = new_n_bits / words_per_block;

    words_.resize(new_n_words);
  }
  set(n_bits_, bit);
  ++n_bits_;
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

  auto const grid_size =
    (num_keys - 1) / (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE) + 1;

  auto ref_ = this->ref(cuco::experimental::bv_read);

  bitvector_get_kernel<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
    ref_, keys_begin, outputs_begin, num_keys);
}

template <class Allocator>
template <typename KeyIt, typename ValueIt>
void bit_vector<Allocator>::set(KeyIt keys_begin,
                                KeyIt keys_end,
                                ValueIt vals_begin,
                                cuda_stream_ref stream) const noexcept
{
  auto const num_keys = cuco::detail::distance(keys_begin, keys_end);
  if (num_keys == 0) { return; }

  auto const grid_size =
    (num_keys - 1) / (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE) + 1;

  auto ref_ = this->ref(cuco::experimental::bv_set);

  bitvector_set_kernel<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
    ref_, keys_begin, vals_begin, num_keys);
}

template <typename BitvectorRef, typename KeyIt, typename OutputIt, typename size_type>
__global__ void bitvector_get_kernel(BitvectorRef ref,
                                     KeyIt keys,
                                     OutputIt outputs,
                                     size_type num_keys)
{
  uint32_t const loop_stride = gridDim.x * blockDim.x;
  uint32_t key_id            = blockDim.x * blockIdx.x + threadIdx.x;

  while (key_id < num_keys) {
    outputs[key_id] = ref.get(keys[key_id]);
    key_id += loop_stride;
  }
}

template <typename BitvectorRef, typename KeyIt, typename ValueIt, typename size_type>
__global__ void bitvector_set_kernel(BitvectorRef ref,
                                     KeyIt keys,
                                     ValueIt values,
                                     size_type num_keys)
{
  uint32_t const loop_stride = gridDim.x * blockDim.x;
  uint32_t key_id            = blockDim.x * blockIdx.x + threadIdx.x;

  while (key_id < num_keys) {
    ref.set(keys[key_id], values[key_id]);
    key_id += loop_stride;
  }
}

template <class Allocator>
void bit_vector<Allocator>::build() noexcept
{
  build_ranks_and_selects(words_, ranks_, selects_, false);   // 1-bits
  build_ranks_and_selects(words_, ranks0_, selects0_, true);  // 0-bits
  move_to_device();
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
void bit_vector<Allocator>::build_ranks_and_selects(const std::vector<slot_type>& words,
                                                    std::vector<rank>& ranks,
                                                    std::vector<size_type>& selects,
                                                    bool flip_bits) noexcept
{
  size_type n_blocks = words.size() / words_per_block;
  ranks.resize(n_blocks + 1);

  size_type count = 0;
  for (size_type block_id = 0; block_id < n_blocks; ++block_id) {
    ranks[block_id].set_abs(count);

    for (size_type block_offset = 0; block_offset < words_per_block; ++block_offset) {
      if (block_offset != 0) {  // Compute deltas
        ranks[block_id].rels_[block_offset - 1] = count - ranks[block_id].abs();
      }

      size_type word_id = (block_id * words_per_block) + block_offset;
      slot_type word    = flip_bits ? ~words[word_id] : words[word_id];

      size_type prev_count = count;
      count += cuda::std::popcount(word);

      if ((prev_count - 1) / bits_per_block != (count - 1) / bits_per_block) {
        add_selects_entry(word_id, word, prev_count, selects);
      }
    }
  }

  ranks.back().set_abs(count);
  selects.push_back(n_blocks);
}

template <class Allocator>
void bit_vector<Allocator>::add_selects_entry(size_type word_id,
                                              slot_type word,
                                              size_type count,
                                              std::vector<size_type>& selects) noexcept
{
  while (word != 0) {
    size_type pos = cuda::std::countr_zero(word);

    if (count % bits_per_block == 0) {
      selects.push_back((word_id * bits_per_word + pos) / bits_per_block);
      break;
    }

    word ^= 1UL << pos;
    ++count;
  }
}

template <class Allocator>
template <class T>
void bit_vector<Allocator>::copy_host_array_to_aow(std::unique_ptr<storage_type>* aow,
                                                   std::vector<T>& host_array) noexcept
{
  uint64_t num_elements = host_array.size();
  *aow = std::make_unique<storage_type>(extent<size_type>{num_elements + 1}, allocator_);

  if (num_elements > 0) {
    // Move host array to device memory
    thrust::device_vector<T> device_array = host_array;
    host_array.clear();

    // Copy device array to window structure
    initialize_aow(*aow, device_array, num_elements);
  }
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
void initialize_aow(std::unique_ptr<Storage>& storage,
                    thrust::device_vector<T>& device_array,
                    uint64_t num_elements)
{
  auto constexpr stride = 4;
  auto const grid_size  = (num_elements + stride * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
                         (stride * detail::CUCO_DEFAULT_BLOCK_SIZE);

  auto device_ptr = reinterpret_cast<uint64_t*>(thrust::raw_pointer_cast(device_array.data()));
  copy_to_window<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE>>>(
    storage->data(), num_elements, device_ptr);
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
