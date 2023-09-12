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

template <typename LabelType, class Allocator>
constexpr trie<LabelType, Allocator>::trie(Allocator const& allocator)
  : allocator_{allocator},
    num_keys_{0},
    num_nodes_{1},
    last_key_{},
    num_levels_{2},
    levels_{2},
    d_levels_ptr_{nullptr},
    device_ptr_{nullptr}
{
  levels_[0].louds_.push_back(0);
  levels_[0].louds_.push_back(1);
  levels_[1].louds_.push_back(1);
  levels_[0].outs_.push_back(0);
  levels_[0].labels_.push_back(root_label_);
}

template <typename LabelType, class Allocator>
trie<LabelType, Allocator>::~trie() noexcept(false)
{
  if (d_levels_ptr_) { CUCO_CUDA_TRY(cudaFree(d_levels_ptr_)); }
  if (device_ptr_) { CUCO_CUDA_TRY(cudaFree(device_ptr_)); }
}

template <typename LabelType, class Allocator>
void trie<LabelType, Allocator>::insert(const std::vector<LabelType>& key) noexcept
{
  if (key == last_key_) { return; }           // Ignore duplicate keys
  assert(num_keys_ == 0 || key > last_key_);  // Keys are expected to be inserted in sorted order

  if (key.empty()) {
    levels_[0].outs_.set(0, 1);
    ++levels_[1].offset_;
    ++num_keys_;
    return;
  }

  if (key.size() + 1 >= levels_.size()) { levels_.resize(key.size() + 2); }

  // Find first position where label is different from last_key
  // Trie is not updated till that position is reached, simply skip to next position
  size_type pos = 0;
  for (; pos < key.size(); ++pos) {
    auto& level = levels_[pos + 1];
    auto label  = key[pos];

    if ((pos == last_key_.size()) || (label != level.labels_.back())) {
      level.louds_.set_last(0);
      level.louds_.push_back(1);
      level.outs_.push_back(0);
      level.labels_.push_back(label);
      ++num_nodes_;
      break;
    }
  }

  // Process remaining labels after divergence point from last_key
  // Each such label will create a new edge and node pair
  for (++pos; pos < key.size(); ++pos) {
    auto& level = levels_[pos + 1];
    level.louds_.push_back(0);
    level.louds_.push_back(1);
    level.outs_.push_back(0);
    level.labels_.push_back(key[pos]);
    ++num_nodes_;
  }

  levels_[key.size() + 1].louds_.push_back(1);  // Mark end of current key
  ++levels_[key.size() + 1].offset_;
  levels_[key.size()].outs_.set_last(1);  // Set terminal bit indicating valid path

  ++num_keys_;
  last_key_ = key;
}

template <typename LabelType, class Allocator>
void trie<LabelType, Allocator>::build() noexcept(false)
{
  // Perform build level-by-level for all levels, followed by a deep-copy from host to device
  size_type offset = 0;

  thrust::device_vector<size_type> test_keys(1, 0);
  thrust::device_vector<bool> test_results(1);

  for (auto& level : levels_) {
    // Run host-bulk test on bitvectors to initiate internal build()
    level.louds_.test(test_keys.begin(), test_keys.end(), test_results.begin());
    louds_refs_.push_back(level.louds_.ref());

    level.outs_.test(test_keys.begin(), test_keys.end(), test_results.begin());
    outs_refs_.push_back(level.outs_.ref());

    level.labels_ptr_ = thrust::raw_pointer_cast(level.labels_.data());

    offset += level.offset_;
    level.offset_ = offset;
  }

  louds_refs_ptr_ = thrust::raw_pointer_cast(louds_refs_.data());
  outs_refs_ptr_  = thrust::raw_pointer_cast(outs_refs_.data());

  num_levels_ = levels_.size();

  // Move levels to device
  CUCO_CUDA_TRY(cudaMalloc(&d_levels_ptr_, sizeof(level) * num_levels_));
  CUCO_CUDA_TRY(
    cudaMemcpy(d_levels_ptr_, &levels_[0], sizeof(level) * num_levels_, cudaMemcpyHostToDevice));

  // Finally create a device copy of full trie structure
  CUCO_CUDA_TRY(cudaMalloc(&device_ptr_, sizeof(trie<LabelType, Allocator>)));
  CUCO_CUDA_TRY(
    cudaMemcpy(device_ptr_, this, sizeof(trie<LabelType, Allocator>), cudaMemcpyHostToDevice));
}

template <typename LabelType, class Allocator>
template <typename KeyIt, typename OffsetIt, typename OutputIt>
void trie<LabelType, Allocator>::lookup(KeyIt keys_begin,
                                        OffsetIt offsets_begin,
                                        OffsetIt offsets_end,
                                        OutputIt outputs_begin,
                                        cuda_stream_ref stream) const noexcept
{
  auto num_keys = cuco::detail::distance(offsets_begin, offsets_end) - 1;
  if (num_keys == 0) { return; }

  auto const grid_size = cuco::detail::grid_size(num_keys);
  auto ref_            = this->ref(cuco::experimental::trie_lookup);

  trie_lookup_kernel<<<grid_size, cuco::detail::default_block_size(), 0, stream>>>(
    ref_, keys_begin, offsets_begin, outputs_begin, num_keys);
}

template <typename TrieRef, typename KeyIt, typename OffsetIt, typename OutputIt>
__global__ void trie_lookup_kernel(
  TrieRef ref, KeyIt keys, OffsetIt offsets, OutputIt outputs, size_t num_keys)
{
  auto key_id            = cuco::detail::global_thread_id();
  auto const loop_stride = cuco::detail::grid_stride();

  while (key_id < num_keys) {
    auto key_start_pos = keys + offsets[key_id];
    auto key_length    = offsets[key_id + 1] - offsets[key_id];

    outputs[key_id] = ref.lookup(key_start_pos, key_length);
    key_id += loop_stride;
  }
}

template <typename LabelType, class Allocator>
template <typename... Operators>
auto trie<LabelType, Allocator>::ref(Operators...) const noexcept
{
  static_assert(sizeof...(Operators), "No operators specified");
  return ref_type<Operators...>{device_ptr_};
}

template <typename LabelType, class Allocator>
trie<LabelType, Allocator>::level::level()
  : louds_{}, outs_{}, labels_{}, labels_ptr_{nullptr}, offset_{0}
{
}

}  // namespace experimental
}  // namespace cuco
