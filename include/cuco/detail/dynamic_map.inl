/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
                                                       sentinel::empty_key<Key> empty_key_sentinel,
                                                       sentinel::empty_value<Value> empty_value_sentinel,
                                                       Allocator const& alloc)
  : empty_key_sentinel_(empty_key_sentinel.value),
    empty_value_sentinel_(empty_value_sentinel.value),
    erased_key_sentinel_(empty_key_sentinel.value),
    size_(0),
    capacity_(initial_capacity),
    min_insert_size_(1E4),
    max_load_factor_(0.60),
    alloc_{alloc},
    counter_allocator_{alloc}
{
  submaps_.push_back(std::make_unique<static_map<Key, Value, Scope, Allocator>>(
    initial_capacity,
    sentinel::empty_key<Key>{empty_key_sentinel},
    sentinel::empty_value<Value>{empty_value_sentinel},
    alloc));
  submap_views_.push_back(submaps_[0]->get_device_view());
  submap_mutable_views_.push_back(submaps_[0]->get_device_mutable_view());
  submap_num_successes_.push_back(submaps_[0]->get_num_successes());
  
  num_successes_ = std::allocator_traits<counter_allocator_type>::allocate(counter_allocator_, 1);
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
dynamic_map<Key, Value, Scope, Allocator>::dynamic_map(std::size_t initial_capacity,
                                                       sentinel::empty_key<Key> empty_key_sentinel,
                                                       sentinel::empty_value<Value> empty_value_sentinel,
                                                       sentinel::erased_key<Key> erased_key_sentinel,
                                                       Allocator const& alloc)
  : empty_key_sentinel_(empty_key_sentinel.value),
    empty_value_sentinel_(empty_value_sentinel.value),
    erased_key_sentinel_(erased_key_sentinel.value),
    size_(0),
    capacity_(initial_capacity),
    min_insert_size_(1E4),
    max_load_factor_(0.60),
    alloc_{alloc},
    counter_allocator_{alloc}
{
  submaps_.push_back(std::make_unique<static_map<Key, Value, Scope, Allocator>>(
    initial_capacity,
    sentinel::empty_key<Key>{empty_key_sentinel_},
    sentinel::empty_value<Value>{empty_value_sentinel_},
    sentinel::erased_key<Key>{erased_key_sentinel_},
    alloc));
  submap_views_.push_back(submaps_[0]->get_device_view());
  submap_mutable_views_.push_back(submaps_[0]->get_device_mutable_view());
  submap_num_successes_.push_back(submaps_[0]->get_num_successes());

  num_successes_ = std::allocator_traits<counter_allocator_type>::allocate(counter_allocator_, 1);
}


template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
dynamic_map<Key, Value, Scope, Allocator>::~dynamic_map()
{
  std::allocator_traits<counter_allocator_type>::deallocate(counter_allocator_, num_successes_, 1);
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
void dynamic_map<Key, Value, Scope, Allocator>::reserve(std::size_t n)
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
      if(erased_key_sentinel_ != empty_key_sentinel_) {
        submaps_.push_back(std::make_unique<static_map<Key, Value, Scope, Allocator>>(
          submap_capacity,
          sentinel::empty_key<Key>{empty_key_sentinel_},
          sentinel::empty_value<Value>{empty_value_sentinel_},
          sentinel::erased_key<Key>{erased_key_sentinel_},
          alloc_));
      } else {
        submaps_.push_back(std::make_unique<static_map<Key, Value, Scope, Allocator>>(
          submap_capacity,
          sentinel::empty_key<Key>{empty_key_sentinel_},
          sentinel::empty_value<Value>{empty_value_sentinel_},
          alloc_));
      }
      submap_views_.push_back(submaps_[submap_idx]->get_device_view());
      submap_mutable_views_.push_back(submaps_[submap_idx]->get_device_mutable_view());
      submap_num_successes_.push_back(submaps_[submap_idx]->get_num_successes());

      capacity_ *= 2;
    }

    num_elements_remaining -= max_load_factor_ * submap_capacity - min_insert_size_;
    submap_idx++;
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope, Allocator>::insert(InputIt first,
                                                       InputIt last,
                                                       Hash hash,
                                                       KeyEqual key_equal)
{
  std::size_t num_to_insert = std::distance(first, last);

  reserve(size_ + num_to_insert);

  uint32_t submap_idx = 0;
  while (num_to_insert > 0) {
    std::size_t capacity_remaining =
      max_load_factor_ * submaps_[submap_idx]->get_capacity() - submaps_[submap_idx]->get_size();
    // If we are tying to insert some of the remaining keys into this submap, we can insert
    // only if we meet the minimum insert size.

    if (capacity_remaining >= min_insert_size_) {
      // TODO: memset an atomic variable is unsafe
      static_assert(sizeof(std::size_t) == sizeof(atomic_ctr_type));
      CUCO_CUDA_TRY(cudaMemset(num_successes_, 0, sizeof(atomic_ctr_type)));
      
      auto n                = std::min(capacity_remaining, num_to_insert);
      auto const block_size = 128;
      auto const stride     = 1;
      auto const tile_size  = 4;
      auto const grid_size  = (tile_size * n + stride * block_size - 1) / (stride * block_size);

      detail::insert<block_size, tile_size, cuco::pair_type<key_type, mapped_type>>
        <<<grid_size, block_size>>>(first,
                                    first + n,
                                    submap_views_.data().get(),
                                    submap_mutable_views_.data().get(),
                                    num_successes_,
                                    submap_idx,
                                    submaps_.size(),
                                    hash,
                                    key_equal);

      std::size_t h_num_successes;
      CUCO_CUDA_TRY(cudaMemcpy(
        &h_num_successes, num_successes_, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost));

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
void dynamic_map<Key, Value, Scope, Allocator>::erase(InputIt first,
                                                       InputIt last,
                                                       Hash hash,
                                                       KeyEqual key_equal)
{
  std::size_t num_keys = std::distance(first, last);

  auto const block_size = 128;
  auto const stride     = 1;
  auto const tile_size  = 4;
  auto const grid_size  = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

  // TODO: memset an atomic variable is unsafe
  static_assert(sizeof(std::size_t) == sizeof(atomic_ctr_type));
  CUCO_CUDA_TRY(cudaMemset(num_successes_, 0, sizeof(atomic_ctr_type)));
  
  // zero out submap success counters
  static_assert(sizeof(std::size_t) == sizeof(atomic_ctr_type));
  for(int i = 0; i < submaps_.size(); ++i) {
    CUCO_CUDA_TRY(cudaMemset(submap_num_successes_[i], 0, sizeof(atomic_ctr_type)));
  }
  
  // TODO: hacky, improve this
  // provide device-accessible vector for each submap num_successes variable
  thrust::device_vector<atomic_ctr_type*> d_submap_num_successes(submap_num_successes_);

  // TODO: hack (how to get size on host?)
  // use dynamic shared memory to hold block reduce space for each submap's erases
  constexpr size_t temp_storage_size_one_block = 48;
  auto const temp_storage_size = submaps_.size() * temp_storage_size_one_block;
      
  detail::erase<block_size, tile_size, cuco::pair_type<key_type, mapped_type>>
    <<<grid_size, block_size, temp_storage_size>>>(
      first,
      first + num_keys,
      submap_views_.data().get(),
      submap_mutable_views_.data().get(),
      num_successes_,
      d_submap_num_successes.data().get(),
      submaps_.size(),
      hash,
      key_equal);

  // update total dynamic map size
  std::size_t h_num_successes;
  CUCO_CUDA_TRY(cudaMemcpy(
    &h_num_successes, num_successes_, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost));
  size_ -= h_num_successes;
  
  // TODO: if only one submap, skip this step
  // update each submap's size
  for(int i = 0; i < submaps_.size(); ++i) {
    std::size_t h_submap_num_successes;
    CUCO_CUDA_TRY(cudaMemcpy(
      &h_submap_num_successes, submap_num_successes_[i], sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost));
    submaps_[i]->size_ -= h_submap_num_successes;
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope, Allocator>::find(
  InputIt first, InputIt last, OutputIt output_begin, Hash hash, KeyEqual key_equal)
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const tile_size  = 4;
  auto const grid_size  = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

  detail::find<block_size, tile_size, Value><<<grid_size, block_size>>>(
    first, last, output_begin, submap_views_.data().get(), submaps_.size(), hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope, Allocator>::contains(
  InputIt first, InputIt last, OutputIt output_begin, Hash hash, KeyEqual key_equal)
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const tile_size  = 4;
  auto const grid_size  = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

  detail::contains<block_size, tile_size><<<grid_size, block_size>>>(
    first, last, output_begin, submap_views_.data().get(), submaps_.size(), hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace cuco
