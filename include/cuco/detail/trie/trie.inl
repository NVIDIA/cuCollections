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
    : levels_(2),
      d_levels_ptr_(nullptr),
      num_levels_(2),
      n_keys_(0),
      n_nodes_(1),
      footprint_(0),
      last_key_(),
      device_impl_(nullptr) {
  levels_[0].louds.add(0);
  levels_[0].louds.add(1);
  levels_[1].louds.add(1);
  levels_[0].outs.add(0);
  levels_[0].labels.push_back(sizeof(T) == 1 ? ' ' : (T)-1);
}

template <typename T>
trie<T>::~trie() noexcept(false) {
  if (d_levels_ptr_) {
    CUCO_CUDA_TRY(cudaFree(d_levels_ptr_));
  }
  if (device_impl_) {
    CUCO_CUDA_TRY(cudaFree(device_impl_));
  }
}

template <typename T>
void trie<T>::add(const std::vector<T>& key) {
  if (key == last_key_) {
    return;
  }
  assert(n_keys_ == 0 || key > last_key_);
  if (key.empty()) {
    levels_[0].outs.set(0, 1);
    ++levels_[1].offset;
    ++n_keys_;
    return;
  }
  if (key.size() + 1 >= levels_.size()) {
    levels_.resize(key.size() + 2);
  }
  uint64_t i = 0;
  for (; i < key.size(); ++i) {
    auto& level = levels_[i + 1];
    T byte = key[i];
    if ((i == last_key_.size()) || (byte != level.labels.back())) {
      level.louds.set_last(0);
      level.louds.add(1);
      level.outs.add(0);
      level.labels.push_back(key[i]);
      ++n_nodes_;
      break;
    }
  }
  for (++i; i < key.size(); ++i) {
    auto& level = levels_[i + 1];
    level.louds.add(0);
    level.louds.add(1);
    level.outs.add(0);
    level.labels.push_back(key[i]);
    ++n_nodes_;
  }
  levels_[key.size() + 1].louds.add(1);
  ++levels_[key.size() + 1].offset;
  levels_[key.size()].outs.set_last(1);
  ++n_keys_;
  last_key_ = key;
}

template <typename T>
void trie<T>::build() {
  uint64_t offset = 0;
  for (uint64_t i = 0; i < levels_.size(); ++i) {
    auto& level = levels_[i];
    level.louds.build();
    level.outs.build();
    offset += level.offset;
    level.offset = offset;
    footprint_ += level.memory_footprint();
    level.d_labels_ptr = move_vector_to_device(level.labels, level.d_labels);
  }

  num_levels_ = levels_.size();
  CUCO_CUDA_TRY(cudaMalloc(&d_levels_ptr_, sizeof(Level) * num_levels_));
  CUCO_CUDA_TRY(cudaMemcpy(d_levels_ptr_, &levels_[0], sizeof(Level) * num_levels_,
                   cudaMemcpyHostToDevice));

  CUCO_CUDA_TRY(cudaMalloc(&device_impl_, sizeof(trie<T>)));
  CUCO_CUDA_TRY(cudaMemcpy(device_impl_, this, sizeof(trie<T>), cudaMemcpyHostToDevice));
}

template <typename T>
__global__ __launch_bounds__(256, 1) void trie_lookup_kernel(const trie<T>* t, const T* keys,
                                                             const uint64_t* offsets, uint64_t* ids,
                                                             uint64_t num_queries,
                                                             uint64_t start_offset) {
  auto const key_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (key_id >= num_queries) {
    return;
  }

  const int length = offsets[key_id + 1] - offsets[key_id];
  const T* query = keys + (offsets[key_id] - start_offset);

  uint32_t node_id = 0;
  for (uint32_t cur_depth = 1; cur_depth <= length; cur_depth++) {
    if (!binary_search_labels_array(t, query[cur_depth - 1], node_id, cur_depth)) {
      ids[key_id] = -1lu;
      return;
    }
  }

  const auto& level = t->d_levels_ptr_[length];
  if (!level.outs.ref(cuco::experimental::get).get(node_id)) {
    ids[key_id] = -1lu;
    return;
  }
  ids[key_id] = level.offset + level.outs.ref(cuco::experimental::rank).rank(node_id);
}

template <typename T>
void trie<T>::lookup(const T* queries, const uint64_t* offsets, uint64_t* ids,
                            uint64_t num_queries, uint64_t start_offset,
                            cudaStream_t stream) const {
  int block_size = 256;
  int num_blocks = (num_queries - 1) / block_size + 1;

  trie_lookup_kernel<<<num_blocks, block_size, 0, stream>>>(device_impl_, queries, offsets, ids,
                                                            num_queries, start_offset);
}

template <typename T>
__device__ uint32_t init_node_pos(const trie<T>* t, uint32_t& node_id, uint32_t cur_depth) {
  uint32_t node_pos = 0;
  if (node_id != 0) {
    node_pos = t->d_levels_ptr_[cur_depth].louds.ref(cuco::experimental::select).select(node_id - 1) + 1;
    node_id = node_pos - node_id;
  }
  return node_pos;
}

template <typename T>
__device__ bool binary_search_labels_array(const trie<T>* t, T target, uint32_t& node_id, uint32_t level_id) {
  const auto& level = t->d_levels_ptr_[level_id];

  uint32_t node_pos = init_node_pos(t, node_id, level_id);
  uint32_t begin = node_id;
  uint32_t pos_end = level.louds.ref(cuco::experimental::find_next_set).find_next_set(node_pos);
  uint32_t end = node_id + (pos_end - node_pos);

  while (begin < end) {
    node_id = (begin + end) / 2;
    auto label = level.d_labels_ptr[node_id];
    if (target < label) {
      end = node_id;
    } else if (target > label) {
      begin = node_id + 1;
    } else {
      break;
    }
  }
  return begin < end;
}

}  // namespace experimental
}  // namespace cuco
