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



/**---------------------------------------------------------------------------*
  * @brief Enumeration of the possible results of attempting to insert into
  *a hash bucket
  *---------------------------------------------------------------------------**/
enum class insert_result {
  CONTINUE,  ///< Insert did not succeed, continue trying to insert
  SUCCESS,   ///< New pair inserted successfully
  DUPLICATE  ///< Insert did not succeed, key is already present
};



template <typename Key, typename Value, cuda::thread_scope Scope>
static_map<Key, Value, Scope>::static_map(std::size_t capacity, Key empty_key_sentinel, Value empty_value_sentinel) : 
  capacity_{capacity},
  empty_key_sentinel_{empty_key_sentinel},
  empty_value_sentinel_{empty_value_sentinel} {
    CUCO_CUDA_TRY(cudaMalloc(&slots_, capacity * sizeof(pair_atomic_type)));
    
    auto constexpr block_size = 256;
    auto constexpr stride = 4;
    auto const grid_size = (capacity + stride * block_size - 1) / (stride * block_size);
    detail::initialize
    <atomic_key_type, atomic_mapped_type>
    <<<grid_size, block_size>>>(slots_, empty_key_sentinel,
                                          empty_value_sentinel, capacity);

    CUCO_CUDA_TRY(cudaMallocManaged(&num_successes_, sizeof(atomic_ctr_type)));
}



template <typename Key, typename Value, cuda::thread_scope Scope>
static_map<Key, Value, Scope>::~static_map() {
  CUCO_CUDA_TRY(cudaFree(slots_));
  CUCO_CUDA_TRY(cudaFree(num_successes_));
}



template <typename Key, typename Value, cuda::thread_scope Scope>
void static_map<Key, Value, Scope>::resize() {
  CUCO_CUDA_TRY(cudaFree(slots_));
  
  capacity_ *= 2;
  size_ = 0;
  CUCO_CUDA_TRY(cudaMalloc(&slots_, capacity_ * sizeof(pair_atomic_type)));
  
  auto constexpr block_size = 256;
  auto constexpr stride = 4;
  auto const grid_size = (capacity_ + stride * block_size - 1) / (stride * block_size);
  detail::initialize
  <atomic_key_type, atomic_mapped_type>
  <<<grid_size, block_size>>>(slots_, empty_key_sentinel_,
                                        empty_value_sentinel_, capacity_);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename Hash, typename KeyEqual>
void static_map<Key, Value, Scope>::rehash(Hash hash, KeyEqual key_equal) {

  pair_atomic_type* old_slots;
  pair_atomic_type* new_slots;
  auto old_capacity = capacity_;
  auto new_capacity = 2 * capacity_;
  CUCO_CUDA_TRY(cudaMalloc(&new_slots, new_capacity * sizeof(pair_atomic_type)));
  
  auto constexpr block_size = 256;
  auto constexpr stride = 4;
  auto grid_size = (new_capacity + stride * block_size - 1) / (stride * block_size);
  detail::initialize
  <atomic_key_type, atomic_mapped_type>
  <<<grid_size, block_size>>>(new_slots, empty_key_sentinel_,
                                          empty_value_sentinel_, new_capacity);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  old_slots = slots_;
  slots_ = new_slots;

  capacity_ = new_capacity;

  grid_size = (old_capacity + stride * block_size - 1) / (stride * block_size);
  auto view = get_device_view();
  auto mutable_view = get_device_mutable_view();
  detail::rehash<Key, Value>
  <<<grid_size, block_size>>>
  (old_slots, old_capacity, mutable_view,
   empty_key_sentinel_,
   hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  CUCO_CUDA_TRY(cudaFree(old_slots));
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename InputIt, typename Hash, typename KeyEqual>
void static_map<Key, Value, Scope>::insert(InputIt first, InputIt last, 
                                           Hash hash, KeyEqual key_equal) {

  auto num_keys = std::distance(first, last);
  auto const block_size = 128;
  auto const stride = 1;
  auto const tile_size = 1;
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) /
                          (stride * block_size);
  auto view = get_device_mutable_view();

  *num_successes_ = 0;
  CUCO_CUDA_TRY(cudaMemPrefetchAsync(num_successes_, sizeof(atomic_ctr_type), 0));

  detail::insert<block_size>
  <<<grid_size, block_size>>>(first, first + num_keys, num_successes_, view, 
                              hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  size_ += num_successes_->load(cuda::std::memory_order_relaxed);
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename InputIt, typename Hash, typename KeyEqual>
void static_map<Key, Value, Scope>::insertSumReduce(InputIt first, InputIt last, 
                                           Hash hash, KeyEqual key_equal) {

  auto num_keys = std::distance(first, last);
  auto const block_size = 128;
  auto const stride = 1;
  auto const tile_size = 4;
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) /
                          (stride * block_size);
  auto view = get_device_view();
  auto mutable_view = get_device_mutable_view();

  *num_successes_ = 0;
  CUCO_CUDA_TRY(cudaMemPrefetchAsync(num_successes_, sizeof(atomic_ctr_type), 0));

  detail::insertSumReduce<block_size, tile_size, Key, Value>
  <<<grid_size, block_size>>>(first, first + num_keys, view, mutable_view,
                              num_successes_, hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  size_ += num_successes_->load(cuda::std::memory_order_relaxed);
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void static_map<Key, Value, Scope>::find(
                                    InputIt first, InputIt last, OutputIt output_begin, 
                                    Hash hash, KeyEqual key_equal) noexcept {
  auto num_keys = std::distance(first, last);
  auto const block_size = 128;
  auto const stride = 1;
  auto const tile_size = 1;
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) /
                         (stride * block_size);
  auto view = get_device_view();

  detail::find<block_size, Value>
  <<<grid_size, block_size>>>
  (first, last, output_begin, view, hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());    
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void static_map<Key, Value, Scope>::contains(
  InputIt first, InputIt last, OutputIt output_begin, Hash hash, KeyEqual key_equal) noexcept {
  
  auto num_keys = std::distance(first, last);
  auto const block_size = 128;
  auto const stride = 1;
  auto const tile_size = 4;
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) /
                          (stride * block_size);
  auto view = get_device_view();
  
  detail::contains<block_size, tile_size>
  <<<grid_size, block_size>>>
  (first, last, output_begin, view, hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename Hash, typename KeyEqual>
__device__ thrust::pair<typename static_map<Key, Value, Scope>::device_mutable_view::iterator, bool> 
static_map<Key, Value, Scope>::device_mutable_view::insert(
  value_type const& insert_pair, Hash hash, KeyEqual key_equal) noexcept {

  auto current_slot{initial_slot(insert_pair.first, hash)};

  while (true) {
    using cuda::std::memory_order_relaxed;
    auto expected_key = empty_key_sentinel_;
    auto& slot_key = current_slot->first;
    auto& slot_value = current_slot->second;

    bool key_success = slot_key.compare_exchange_strong(expected_key,
                                                        insert_pair.first,
                                                        memory_order_relaxed);

    if(key_success) {
      slot_value.store(insert_pair.second, memory_order_relaxed);
      return thrust::make_pair(current_slot, true);
    }
    // our key was already present in the slot, so our key is a duplicate
    else if(key_equal(insert_pair.first, expected_key)) {
      while(slot_value.load(memory_order_relaxed) == empty_value_sentinel_) {}
      return thrust::make_pair(current_slot, false);
    }
    
    // if we couldn't insert the key, but it wasn't a duplicate, then there must
    // have been some other key there, so we keep looking for a slot
    current_slot = next_slot(current_slot);
  }
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename CG, typename Hash, typename KeyEqual>
__device__ thrust::pair<typename static_map<Key, Value, Scope>::device_mutable_view::iterator, bool> 
static_map<Key, Value, Scope>::device_mutable_view::insert(
  CG g, value_type const& insert_pair, Hash hash, KeyEqual key_equal) noexcept {
  
  using cuda::std::memory_order_relaxed;
  auto current_slot = initial_slot(g, insert_pair.first, hash);
  
  while(true) {
    auto& slot_key = current_slot->first;
    auto& slot_value = current_slot->second;
    key_type const existing_key = slot_key.load(cuda::std::memory_order_seq_cst);
    uint32_t existing = g.ballot(key_equal(existing_key, insert_pair.first));
    
    // the key we are trying to insert is already in the map, so we return
    // with failure to insert
    if(existing) {
      uint32_t src_lane = __ffs(existing) - 1;
      auto src_current_slot = reinterpret_cast<iterator>(
        g.shfl(reinterpret_cast<intptr_t>(current_slot), src_lane));
      auto& src_slot_value = src_current_slot->second; 
      while(src_slot_value.load(memory_order_relaxed) == empty_value_sentinel_) {}
      return thrust::make_pair(src_current_slot, false);
    }
    
    uint32_t empty = g.ballot(existing_key == empty_key_sentinel_);

    // we found an empty slot, but not the key we are inserting, so this must
    // be an empty slot into which we can insert the key
    if(empty) {
      // the first lane in the group with an empty slot will attempt the insert
      insert_result status{insert_result::CONTINUE};
      uint32_t src_lane = __ffs(empty) - 1;

      if(g.thread_rank() == src_lane) {
        auto expected_key = empty_key_sentinel_;
        bool key_success = slot_key.compare_exchange_strong(expected_key,
                                                            insert_pair.first,
                                                            memory_order_relaxed);
              
        if(key_success) {
          slot_value.store(insert_pair.second, memory_order_relaxed);
          status = insert_result::SUCCESS;
        }
        // our key was already present in the slot, so our key is a duplicate
        else if(key_equal(insert_pair.first, expected_key)) {
          status = insert_result::DUPLICATE;
        }
        // another key was inserted in the slot we wanted to try
        // so we need to try the next empty slot in the window
      }

      uint32_t res_status = g.shfl(static_cast<uint32_t>(status), src_lane);
      status = static_cast<insert_result>(res_status);

      // successful insert
      if(status == insert_result::SUCCESS) {
        auto src_current_slot = reinterpret_cast<iterator>(
          g.shfl(reinterpret_cast<intptr_t>(current_slot), src_lane));
        return thrust::make_pair(src_current_slot, true);
      }
      // duplicate present during insert
      if(status == insert_result::DUPLICATE) {
        auto src_current_slot = reinterpret_cast<iterator>(
          g.shfl(reinterpret_cast<intptr_t>(current_slot), src_lane));
        auto& src_slot_value = src_current_slot->second; 
        while(src_slot_value.load(memory_order_relaxed) == empty_value_sentinel_) {}
        return thrust::make_pair(src_current_slot, false);
      }
      // if we've gotten this far, a different key took our spot 
      // before we could insert. We need to retry the insert on the
      // same window
    }
    // if there are no empty slots in the current window,
    // we move onto the next window
    else {
      current_slot = next_slot(g, current_slot);
    }
  }
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename Hash>
__device__ typename static_map<Key, Value, Scope>::device_mutable_view::iterator
static_map<Key, Value, Scope>::device_mutable_view::initial_slot(
  Key const& k, Hash hash) const noexcept {

  return &slots_[hash(k) % capacity_];
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename CG, typename Hash>
__device__ typename static_map<Key, Value, Scope>::device_mutable_view::iterator
static_map<Key, Value, Scope>::device_mutable_view::initial_slot(
  CG g, Key const& k, Hash hash) const noexcept {

  return &slots_[(hash(k) + g.thread_rank()) % capacity_];
}



template <typename Key, typename Value, cuda::thread_scope Scope>
__device__ typename static_map<Key, Value, Scope>::device_mutable_view::iterator
static_map<Key, Value, Scope>::device_mutable_view::next_slot(
  iterator s) const noexcept {

  return (++s < end()) ? s : slots_;
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename CG>
__device__ typename static_map<Key, Value, Scope>::device_mutable_view::iterator
static_map<Key, Value, Scope>::device_mutable_view::next_slot(
  CG g, iterator s) const noexcept {

  uint32_t index = s - slots_;
  return &slots_[(index + g.size()) % capacity_];
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename Hash, typename KeyEqual>
__device__ typename static_map<Key, Value, Scope>::device_view::iterator
static_map<Key, Value, Scope>::device_view::find(
  Key const& k, Hash hash, KeyEqual key_equal) noexcept {

  using cuda::std::memory_order_relaxed;
  auto current_slot = initial_slot(k, hash);

  while (true) {
    auto const existing_key =
        current_slot->first.load(cuda::std::memory_order_relaxed);
    // Key exists, return iterator to location
    if (key_equal(existing_key, k)) {
      while(current_slot->second.load(memory_order_relaxed) == empty_value_sentinel_) {}
      return current_slot;
    }

    // Key doesn't exist, return end()
    if (existing_key == empty_key_sentinel_) {
      return end();
    }

    current_slot = next_slot(current_slot);
  }
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename CG, typename Hash, typename KeyEqual>
__device__ typename static_map<Key, Value, Scope>::device_view::iterator
static_map<Key, Value, Scope>::device_view::find(
  CG g, Key const& k, Hash hash, KeyEqual key_equal) noexcept {
  
  using cuda::std::memory_order_relaxed;
  auto current_slot = initial_slot(g, k, hash);

  while(true) {
    auto const existing_key = 
        current_slot->first.load(cuda::std::memory_order_relaxed);
    uint32_t existing = g.ballot(key_equal(existing_key, k));
    
    // the key we were searching for was found by one of the threads,
    // so we return an iterator to the entry
    if(existing) {
      uint32_t src_lane = __ffs(existing) - 1;
      auto src_current_slot = reinterpret_cast<iterator>(
        g.shfl(reinterpret_cast<intptr_t>(current_slot), src_lane));
      auto& src_slot_value = src_current_slot->second;
      while(src_slot_value.load(memory_order_relaxed) == empty_value_sentinel_) {}
      return src_current_slot;
    }
    
    // we found an empty slot, meaning that the key we're searching 
    // for isn't in this submap, so we should move onto the next one
    uint32_t empty = g.ballot(existing_key == empty_key_sentinel_);
    if(empty) {
      return end();
    }

    // otherwise, all slots in the current window are full with other keys,
    // so we move onto the next window in the current submap
    
    current_slot = next_slot(g, current_slot);
  }
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename Hash, typename KeyEqual>
__device__ bool static_map<Key, Value, Scope>::device_view::contains(
  Key const& k, Hash hash, KeyEqual key_equal) noexcept {

  auto current_slot = initial_slot(k, hash);

  while (true) {
    auto const existing_key =
        current_slot->first.load(cuda::std::memory_order_relaxed);
    
    if (key_equal(existing_key, k)) {
      return true;
    }

    if (existing_key == empty_key_sentinel_) {
      return false;
    }

    current_slot = next_slot(current_slot);
  }
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename CG, typename Hash, typename KeyEqual>
__device__ bool static_map<Key, Value, Scope>::device_view::contains(
  CG g, Key const& k, Hash hash, KeyEqual key_equal) noexcept {

  auto current_slot = initial_slot(g, k, hash);

  while(true) {
    key_type const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);
    uint32_t existing = g.ballot(key_equal(existing_key, k));
    
    // the key we were searching for was found by one of the threads,
    // so we return an iterator to the entry
    if(existing) {
      return true;
    }
    
    // we found an empty slot, meaning that the key we're searching 
    // for isn't in this submap, so we should move onto the next one
    uint32_t empty = g.ballot(existing_key == empty_key_sentinel_);
    if(empty) {
      return false;
    }

    // otherwise, all slots in the current window are full with other keys,
    // so we move onto the next window in the current submap
    current_slot = next_slot(current_slot);
  }
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename Hash>
__device__ typename static_map<Key, Value, Scope>::device_view::iterator
static_map<Key, Value, Scope>::device_view::initial_slot(
  Key const& k, Hash hash) const noexcept {

  return &slots_[hash(k) % capacity_];
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename CG, typename Hash>
__device__ typename static_map<Key, Value, Scope>::device_view::iterator
static_map<Key, Value, Scope>::device_view::initial_slot(
  CG g, Key const& k, Hash hash) const noexcept {

  return &slots_[(hash(k) + g.thread_rank()) % capacity_];
}



template <typename Key, typename Value, cuda::thread_scope Scope>
__device__ typename static_map<Key, Value, Scope>::device_view::iterator
static_map<Key, Value, Scope>::device_view::next_slot(
  iterator s) const noexcept {

  return (++s < end()) ? s : slots_;
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename CG>
__device__ typename static_map<Key, Value, Scope>::device_view::iterator
static_map<Key, Value, Scope>::device_view::next_slot(
  CG g, iterator s) const noexcept {

  uint32_t index = s - slots_;
  return &slots_[(index + g.size()) % capacity_];
}



}