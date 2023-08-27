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
trie<T>::trie()
  : levels_{2},
    d_levels_ptr_{nullptr},
    num_levels_{2},
    n_keys_{0},
    n_nodes_{1},
    last_key_{},
    device_ptr_{nullptr}
{
  levels_[0].louds.append(0);
  levels_[0].louds.append(1);
  levels_[1].louds.append(1);
  levels_[0].outs.append(0);
  levels_[0].labels.push_back(root_label_);
}

template <typename T>
trie<T>::~trie() noexcept(false)
{
  if (d_levels_ptr_) { CUCO_CUDA_TRY(cudaFree(d_levels_ptr_)); }
  if (device_ptr_) { CUCO_CUDA_TRY(cudaFree(device_ptr_)); }
}

template <typename T>
void trie<T>::insert(const std::vector<T>& key)
{
  if (key == last_key_) { return; }         // Ignore duplicate keys
  assert(n_keys_ == 0 || key > last_key_);  // Keys are expected to be inserted in sorted order

  if (key.empty()) {
    levels_[0].outs.set(0, 1);
    ++levels_[1].offset;
    ++n_keys_;
    return;
  }

  if (key.size() + 1 >= levels_.size()) { levels_.resize(key.size() + 2); }

  // Find first position where label is different from last_key
  // Trie is not updated till that position is reached, simply skip to next position
  size_type pos = 0;
  for (; pos < key.size(); ++pos) {
    auto& level = levels_[pos + 1];
    T label     = key[pos];

    if ((pos == last_key_.size()) || (label != level.labels.back())) {
      level.louds.set_last(0);
      level.louds.append(1);
      level.outs.append(0);
      level.labels.push_back(label);
      ++n_nodes_;
      break;
    }
  }

  // Process remaining labels after divergence point from last_key
  // Each such label will create a new edge and node pair in trie
  for (++pos; pos < key.size(); ++pos) {
    auto& level = levels_[pos + 1];
    level.louds.append(0);
    level.louds.append(1);
    level.outs.append(0);
    level.labels.push_back(key[pos]);
    ++n_nodes_;
  }

  levels_[key.size() + 1].louds.append(1);  // Mark end of current key
  ++levels_[key.size() + 1].offset;
  levels_[key.size()].outs.set_last(1);  // Set terminal bit indicating valid path

  ++n_keys_;
  last_key_ = key;
}

// Helper to move vector from host to device
// Host vector is clear to avoid duplication. Device pointer is returned
template <typename T>
T* move_vector_to_device(std::vector<T>& host_vector, thrust::device_vector<T>& device_vector)
{
  device_vector = host_vector;
  host_vector.clear();
  return thrust::raw_pointer_cast(device_vector.data());
}

template <typename T>
void trie<T>::build()
{
  // Perform build level-by-level for all levels, followed by a deep-copy from host to device

  // Host-side per-level bit-vector refs
  std::vector<bv_read_ref> louds_refs, outs_refs;
  size_type offset = 0;

  for (auto& level : levels_) {
    level.louds.build();
    louds_refs.push_back(level.louds.ref(bv_read));

    level.outs.build();
    outs_refs.push_back(level.outs.ref(bv_read));

    // Move labels to device
    level.d_labels_ptr = move_vector_to_device(level.labels, level.d_labels);

    offset += level.offset;
    level.offset = offset;
  }

  // Move bitvector refs to device
  d_louds_refs_ptr_ = move_vector_to_device(louds_refs, d_louds_refs_);
  d_outs_refs_ptr_  = move_vector_to_device(outs_refs, d_outs_refs_);

  num_levels_ = levels_.size();

  // Move levels to device
  CUCO_CUDA_TRY(cudaMalloc(&d_levels_ptr_, sizeof(level) * num_levels_));
  CUCO_CUDA_TRY(
    cudaMemcpy(d_levels_ptr_, &levels_[0], sizeof(level) * num_levels_, cudaMemcpyHostToDevice));

  // Finally create a device copy of full trie structure
  CUCO_CUDA_TRY(cudaMalloc(&device_ptr_, sizeof(trie<T>)));
  CUCO_CUDA_TRY(cudaMemcpy(device_ptr_, this, sizeof(trie<T>), cudaMemcpyHostToDevice));
}

template <typename T>
template <typename KeyIt, typename OffsetIt, typename OutputIt>
void trie<T>::lookup(KeyIt keys_begin,
                     OffsetIt offsets_begin,
                     OffsetIt offsets_end,
                     OutputIt outputs_begin,
                     cuda_stream_ref stream) const
{
  auto num_keys = cuco::detail::distance(offsets_begin, offsets_end) - 1;
  if (num_keys == 0) { return; }

  auto grid_size =
    (num_keys - 1) / (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE) + 1;

  auto ref_ = this->ref(cuco::experimental::trie_lookup);

  trie_lookup_kernel<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
    ref_, keys_begin, offsets_begin, outputs_begin, num_keys);
}

template <typename TrieRef, typename KeyIt, typename OffsetIt, typename OutputIt>
__global__ void trie_lookup_kernel(
  TrieRef ref, KeyIt keys, OffsetIt offsets, OutputIt outputs, uint64_t num_keys)
{
  size_t loop_stride = gridDim.x * blockDim.x;
  size_t key_id      = blockDim.x * blockIdx.x + threadIdx.x;

  while (key_id < num_keys) {
    auto key_start_pos = keys + offsets[key_id];
    size_t key_length  = offsets[key_id + 1] - offsets[key_id];

    outputs[key_id] = ref.lookup_key(key_start_pos, key_length);
    key_id += loop_stride;
  }
}

template <typename T>
template <typename... Operators>
auto trie<T>::ref(Operators...) const noexcept
{
  static_assert(sizeof...(Operators), "No operators specified");
  return ref_type<Operators...>{device_ptr_};
}

template <typename T>
trie<T>::level::level() : louds{}, outs{}, labels{}, d_labels_ptr{nullptr}, offset{0}
{
}

}  // namespace experimental
}  // namespace cuco
