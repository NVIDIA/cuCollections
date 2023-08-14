/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

namespace cuco {

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
dynamic_map<Key, Value, Scope, Allocator>::dynamic_map(std::size_t initial_capacity,
                                                       empty_key<Key> empty_key_sentinel,
                                                       empty_value<Value> empty_value_sentinel,
                                                       Allocator const& alloc,
                                                       cudaStream_t stream)
  : empty_key_sentinel_(empty_key_sentinel.value),
    empty_value_sentinel_(empty_value_sentinel.value),
    erased_key_sentinel_(empty_key_sentinel.value),
    size_(0),
    capacity_(initial_capacity),
    min_insert_size_(1E4),
    max_load_factor_(0.60),
    alloc_{alloc}
{
  submaps_.push_back(std::make_unique<static_map<Key, Value, Scope, Allocator>>(
    initial_capacity,
    empty_key<Key>{empty_key_sentinel},
    empty_value<Value>{empty_value_sentinel},
    alloc,
    stream));
  submap_views_.push_back(submaps_[0]->get_device_view());
  submap_mutable_views_.push_back(submaps_[0]->get_device_mutable_view());
  submap_num_successes_.push_back(submaps_[0]->num_successes_);
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
dynamic_map<Key, Value, Scope, Allocator>::dynamic_map(std::size_t initial_capacity,
                                                       empty_key<Key> empty_key_sentinel,
                                                       empty_value<Value> empty_value_sentinel,
                                                       erased_key<Key> erased_key_sentinel,
                                                       Allocator const& alloc,
                                                       cudaStream_t stream)
  : empty_key_sentinel_(empty_key_sentinel.value),
    empty_value_sentinel_(empty_value_sentinel.value),
    erased_key_sentinel_(erased_key_sentinel.value),
    size_(0),
    capacity_(initial_capacity),
    min_insert_size_(1E4),
    max_load_factor_(0.60),
    alloc_{alloc}
{
  CUCO_EXPECTS(empty_key_sentinel_ != erased_key_sentinel_,
               "The empty key sentinel and erased key sentinel cannot be the same value.",
               std::runtime_error);

  submaps_.push_back(std::make_unique<static_map<Key, Value, Scope, Allocator>>(
    initial_capacity,
    empty_key<Key>{empty_key_sentinel_},
    empty_value<Value>{empty_value_sentinel_},
    erased_key<Key>{erased_key_sentinel_},
    alloc,
    stream));
  submap_views_.push_back(submaps_[0]->get_device_view());
  submap_mutable_views_.push_back(submaps_[0]->get_device_mutable_view());
  submap_num_successes_.push_back(submaps_[0]->num_successes_);
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
void dynamic_map<Key, Value, Scope, Allocator>::reserve(std::size_t n, cudaStream_t stream)
{
  int64_t num_elements_remaining = n;
  uint32_t submap_idx            = 0;
  while (num_elements_remaining > 0) {
    std::size_t submap_capacity;

    // if the submap already exists
    if (submap_idx < submaps_.size()) {
      submap_capacity = submaps_[submap_idx]->get_capacity();
    }
    // if the submap does not exist yet, create it
    else {
      submap_capacity = capacity_;
      if (erased_key_sentinel_ != empty_key_sentinel_) {
        submaps_.push_back(std::make_unique<static_map<Key, Value, Scope, Allocator>>(
          submap_capacity,
          empty_key<Key>{empty_key_sentinel_},
          empty_value<Value>{empty_value_sentinel_},
          erased_key<Key>{erased_key_sentinel_},
          alloc_,
          stream));
      } else {
        submaps_.push_back(std::make_unique<static_map<Key, Value, Scope, Allocator>>(
          submap_capacity,
          empty_key<Key>{empty_key_sentinel_},
          empty_value<Value>{empty_value_sentinel_},
          alloc_,
          stream));
      }
      submap_num_successes_.push_back(submaps_[submap_idx]->num_successes_);
      submap_views_.push_back(submaps_[submap_idx]->get_device_view());
      submap_mutable_views_.push_back(submaps_[submap_idx]->get_device_mutable_view());
      capacity_ *= 2;
    }

    num_elements_remaining -= max_load_factor_ * submap_capacity - min_insert_size_;
    submap_idx++;
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope, Allocator>::insert(
  InputIt first, InputIt last, Hash hash, KeyEqual key_equal, cudaStream_t stream)
{
  // TODO: memset an atomic variable is unsafe
  static_assert(sizeof(std::size_t) == sizeof(atomic_ctr_type),
                "sizeof(atomic_ctr_type) must be equal to sizeof(std:size_t).");

  auto constexpr block_size = 128;
  auto constexpr stride     = 1;
  auto constexpr tile_size  = 4;

  std::size_t num_to_insert = std::distance(first, last);

  reserve(size_ + num_to_insert, stream);

  uint32_t submap_idx = 0;
  while (num_to_insert > 0) {
    std::size_t capacity_remaining =
      max_load_factor_ * submaps_[submap_idx]->get_capacity() - submaps_[submap_idx]->get_size();
    // If we are tying to insert some of the remaining keys into this submap, we can insert
    // only if we meet the minimum insert size.
    if (capacity_remaining >= min_insert_size_) {
      CUCO_CUDA_TRY(
        cudaMemsetAsync(submap_num_successes_[submap_idx], 0, sizeof(atomic_ctr_type), stream));

      auto const n         = std::min(capacity_remaining, num_to_insert);
      auto const grid_size = (tile_size * n + stride * block_size - 1) / (stride * block_size);

      detail::insert<block_size, tile_size, cuco::pair<key_type, mapped_type>>
        <<<grid_size, block_size, 0, stream>>>(first,
                                               first + n,
                                               submap_views_.data().get(),
                                               submap_mutable_views_.data().get(),
                                               submap_num_successes_.data().get(),
                                               submap_idx,
                                               submaps_.size(),
                                               hash,
                                               key_equal);

      std::size_t h_num_successes;
      CUCO_CUDA_TRY(cudaMemcpyAsync(&h_num_successes,
                                    submap_num_successes_[submap_idx],
                                    sizeof(atomic_ctr_type),
                                    cudaMemcpyDeviceToHost,
                                    stream));
      submaps_[submap_idx]->size_ += h_num_successes;
      size_ += h_num_successes;
      first += n;
      num_to_insert -= n;
    }
    submap_idx++;
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope, Allocator>::erase(
  InputIt first, InputIt last, Hash hash, KeyEqual key_equal, cudaStream_t stream)
{
  // TODO: memset an atomic variable is unsafe
  static_assert(sizeof(std::size_t) == sizeof(atomic_ctr_type),
                "sizeof(atomic_ctr_type) must be equal to sizeof(std:size_t).");

  auto constexpr block_size = 128;
  auto constexpr stride     = 1;
  auto constexpr tile_size  = 4;

  auto const num_keys  = std::distance(first, last);
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

  // zero out submap success counters
  for (uint32_t i = 0; i < submaps_.size(); ++i) {
    CUCO_CUDA_TRY(cudaMemsetAsync(submap_num_successes_[i], 0, sizeof(atomic_ctr_type), stream));
  }

  auto const temp_storage_size = submaps_.size() * sizeof(unsigned long long);

  detail::erase<block_size, tile_size>
    <<<grid_size, block_size, temp_storage_size, stream>>>(first,
                                                           first + num_keys,
                                                           submap_mutable_views_.data().get(),
                                                           submap_num_successes_.data().get(),
                                                           submaps_.size(),
                                                           hash,
                                                           key_equal);

  for (uint32_t i = 0; i < submaps_.size(); ++i) {
    std::size_t h_submap_num_successes;
    CUCO_CUDA_TRY(cudaMemcpyAsync(&h_submap_num_successes,
                                  submap_num_successes_[i],
                                  sizeof(atomic_ctr_type),
                                  cudaMemcpyDeviceToHost,
                                  stream));
    submaps_[i]->size_ -= h_submap_num_successes;
    size_ -= h_submap_num_successes;
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope, Allocator>::find(InputIt first,
                                                     InputIt last,
                                                     OutputIt output_begin,
                                                     Hash hash,
                                                     KeyEqual key_equal,
                                                     cudaStream_t stream)
{
  auto constexpr block_size = 128;
  auto constexpr stride     = 1;
  auto constexpr tile_size  = 4;

  auto const num_keys  = std::distance(first, last);
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

  detail::find<block_size, tile_size, Value><<<grid_size, block_size, 0, stream>>>(
    first, last, output_begin, submap_views_.data().get(), submaps_.size(), hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope, Allocator>::contains(InputIt first,
                                                         InputIt last,
                                                         OutputIt output_begin,
                                                         Hash hash,
                                                         KeyEqual key_equal,
                                                         cudaStream_t stream)
{
  auto constexpr block_size = 128;
  auto constexpr stride     = 1;
  auto constexpr tile_size  = 4;

  auto const num_keys  = std::distance(first, last);
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

  detail::contains<block_size, tile_size><<<grid_size, block_size, 0, stream>>>(
    first, last, output_begin, submap_views_.data().get(), submaps_.size(), hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace cuco
