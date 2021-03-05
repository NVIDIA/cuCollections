/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
static_multimap<Key, Value, Scope, Allocator>::static_multimap(std::size_t capacity,
                                                               Key empty_key_sentinel,
                                                               Value empty_value_sentinel,
                                                               Allocator const& alloc)
  : capacity_{capacity},
    empty_key_sentinel_{empty_key_sentinel},
    empty_value_sentinel_{empty_value_sentinel},
    slot_allocator_{alloc}
{
  slots_ = std::allocator_traits<slot_allocator_type>::allocate(slot_allocator_, capacity);

  auto constexpr block_size = 256;
  auto constexpr stride     = 4;
  auto const grid_size      = (capacity + stride * block_size - 1) / (stride * block_size);
  detail::initialize<atomic_key_type, atomic_mapped_type>
    <<<grid_size, block_size>>>(slots_, empty_key_sentinel, empty_value_sentinel, capacity);
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
static_multimap<Key, Value, Scope, Allocator>::~static_multimap()
{
  std::allocator_traits<slot_allocator_type>::deallocate(slot_allocator_, slots_, capacity_);
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename Hash, typename KeyEqual>
void static_multimap<Key, Value, Scope, Allocator>::insert(InputIt first,
                                                           InputIt last,
                                                           Hash hash,
                                                           KeyEqual key_equal)
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const tile_size  = 4;
  auto const grid_size  = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_mutable_view();

  detail::insert<block_size, tile_size>
    <<<grid_size, block_size>>>(first, first + num_keys, view, hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void static_multimap<Key, Value, Scope, Allocator>::find(
  InputIt first, InputIt last, OutputIt output_begin, Hash hash, KeyEqual key_equal) noexcept
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const tile_size  = 4;
  auto const grid_size  = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_view();

  detail::find<block_size, tile_size, Value>
    <<<grid_size, block_size>>>(first, last, output_begin, view, hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void static_multimap<Key, Value, Scope, Allocator>::contains(
  InputIt first, InputIt last, OutputIt output_begin, Hash hash, KeyEqual key_equal) noexcept
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const tile_size  = 4;
  auto const grid_size  = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_view();

  detail::contains<block_size, tile_size>
    <<<grid_size, block_size>>>(first, last, output_begin, view, hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void static_multimap<Key, Value, Scope, Allocator>::find_all(InputIt first,
                                                             InputIt last,
                                                             OutputIt output_begin,
                                                             OutputIt output_end,
                                                             Hash hash,
                                                             KeyEqual key_equal) noexcept
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const tile_size  = 4;
  auto const grid_size  = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_view();

  detail::find_all<block_size, tile_size, Value>
    <<<grid_size, block_size>>>(first, last, output_begin, output_end, view, hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename Hash, typename KeyEqual>
std::size_t static_multimap<Key, Value, Scope, Allocator>::count(InputIt first,
                                                                 InputIt last,
                                                                 Hash hash,
                                                                 KeyEqual key_equal)
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const tile_size  = 4;
  auto const grid_size  = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_view();

  atomic_ctr_type* num_items;
  CUCO_CUDA_TRY(cudaMallocManaged(&num_items, sizeof(atomic_ctr_type)));
  *num_items = 0;
  int device_id;
  CUCO_CUDA_TRY(cudaGetDevice(&device_id));
  CUCO_CUDA_TRY(cudaMemPrefetchAsync(num_items, sizeof(atomic_ctr_type), device_id));

  detail::count<block_size, tile_size, Value>
    <<<grid_size, block_size>>>(first, last, num_items, view, hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  return *num_items;
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ void static_multimap<Key, Value, Scope, Allocator>::device_mutable_view::insert(
  value_type const& insert_pair, Hash hash, KeyEqual key_equal) noexcept
{
  auto current_slot{initial_slot(insert_pair.first, hash)};

  while (true) {
    using cuda::std::memory_order_relaxed;
    auto expected_key   = this->get_empty_key_sentinel();
    auto expected_value = this->get_empty_value_sentinel();
    auto& slot_key      = current_slot->first;
    auto& slot_value    = current_slot->second;

    bool key_success =
      slot_key.compare_exchange_strong(expected_key, insert_pair.first, memory_order_relaxed);
    bool value_success =
      slot_value.compare_exchange_strong(expected_value, insert_pair.second, memory_order_relaxed);

    if (key_success) {
      while (not value_success) {
        value_success =
          slot_value.compare_exchange_strong(expected_value = this->get_empty_value_sentinel(),
                                             insert_pair.second,
                                             memory_order_relaxed);
      }
      return;
    } else if (value_success) {
      slot_value.store(this->get_empty_value_sentinel(), memory_order_relaxed);
    }

    // If we couldn't insert the key, then there must have been some other key there. So we keep
    // looking for a slot
    current_slot = next_slot(current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ void static_multimap<Key, Value, Scope, Allocator>::device_mutable_view::insert(
  CG g, value_type const& insert_pair, Hash hash, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, insert_pair.first, hash);

  while (true) {
    key_type const existing_key = current_slot->first.load(cuda::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as the
    // sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty         = (existing_key == this->get_empty_key_sentinel());
    auto const window_contains_empty = g.ballot(slot_is_empty);

    if (window_contains_empty) {
      // the first lane in the group with an empty slot will attempt the insert
      insert_result status{insert_result::CONTINUE};
      uint32_t src_lane = __ffs(window_contains_empty) - 1;

      if (g.thread_rank() == src_lane) {
        using cuda::std::memory_order_relaxed;
        auto expected_key   = this->get_empty_key_sentinel();
        auto expected_value = this->get_empty_value_sentinel();
        auto& slot_key      = current_slot->first;
        auto& slot_value    = current_slot->second;

        bool key_success =
          slot_key.compare_exchange_strong(expected_key, insert_pair.first, memory_order_relaxed);
        bool value_success = slot_value.compare_exchange_strong(
          expected_value, insert_pair.second, memory_order_relaxed);

        if (key_success) {
          while (not value_success) {
            value_success =
              slot_value.compare_exchange_strong(expected_value = this->get_empty_value_sentinel(),
                                                 insert_pair.second,
                                                 memory_order_relaxed);
          }
          status = insert_result::SUCCESS;
        } else if (value_success) {
          slot_value.store(this->get_empty_value_sentinel(), memory_order_relaxed);
        }

        // another key was inserted in the slot we wanted to try
        // so we need to try the next empty slot in the window
      }

      uint32_t res_status = g.shfl(static_cast<uint32_t>(status), src_lane);
      status              = static_cast<insert_result>(res_status);

      // successful insert
      if (status == insert_result::SUCCESS) { return; }
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

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ typename static_multimap<Key, Value, Scope, Allocator>::device_view::iterator
static_multimap<Key, Value, Scope, Allocator>::device_view::find(Key const& k,
                                                                 Hash hash,
                                                                 KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);
    // Key doesn't exist, return end()
    if (existing_key == this->get_empty_key_sentinel()) { return this->end(); }

    // Key exists, return iterator to location
    if (key_equal(existing_key, k)) { return current_slot; }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ typename static_multimap<Key, Value, Scope, Allocator>::device_view::const_iterator
static_multimap<Key, Value, Scope, Allocator>::device_view::find(Key const& k,
                                                                 Hash hash,
                                                                 KeyEqual key_equal) const noexcept
{
  auto current_slot = initial_slot(k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);
    // Key doesn't exist, return end()
    if (existing_key == this->get_empty_key_sentinel()) { return this->end(); }

    // Key exists, return iterator to location
    if (key_equal(existing_key, k)) { return current_slot; }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ typename static_multimap<Key, Value, Scope, Allocator>::device_view::iterator
static_multimap<Key, Value, Scope, Allocator>::device_view::find(CG g,
                                                                 Key const& k,
                                                                 Hash hash,
                                                                 KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as the
    // sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty = (existing_key == this->get_empty_key_sentinel());

    // the key we were searching for was found by one of the threads,
    // so we return an iterator to the entry
    auto const exists = g.ballot(not slot_is_empty and key_equal(existing_key, k));
    if (exists) {
      uint32_t src_lane = __ffs(exists) - 1;
      // TODO: This shouldn't cast an iterator to an int to shuffle. Instead, get the index of the
      // current_slot and shuffle that instead.
      intptr_t res_slot = g.shfl(reinterpret_cast<intptr_t>(current_slot), src_lane);
      return reinterpret_cast<iterator>(res_slot);
    }

    // we found an empty slot, meaning that the key we're searching for isn't present
    if (g.ballot(slot_is_empty)) { return this->end(); }

    // otherwise, all slots in the current window are full with other keys, so we move onto the
    // next window
    current_slot = next_slot(g, current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ typename static_multimap<Key, Value, Scope, Allocator>::device_view::const_iterator
static_multimap<Key, Value, Scope, Allocator>::device_view::find(CG g,
                                                                 Key const& k,
                                                                 Hash hash,
                                                                 KeyEqual key_equal) const noexcept
{
  auto current_slot = initial_slot(g, k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as the
    // sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty = (existing_key == this->get_empty_key_sentinel());

    // the key we were searching for was found by one of the threads, so we return an iterator to
    // the entry
    auto const exists = g.ballot(not slot_is_empty and key_equal(existing_key, k));
    if (exists) {
      uint32_t src_lane = __ffs(exists) - 1;
      // TODO: This shouldn't cast an iterator to an int to shuffle. Instead, get the index of the
      // current_slot and shuffle that instead.
      intptr_t res_slot = g.shfl(reinterpret_cast<intptr_t>(current_slot), src_lane);
      return reinterpret_cast<const_iterator>(res_slot);
    }

    // we found an empty slot, meaning that the key we're searching
    // for isn't in this submap, so we should move onto the next one
    if (g.ballot(slot_is_empty)) { return this->end(); }

    // otherwise, all slots in the current window are full with other keys,
    // so we move onto the next window in the current submap

    current_slot = next_slot(g, current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ typename static_multimap<Key, Value, Scope, Allocator>::device_view::fancy_iterator
static_multimap<Key, Value, Scope, Allocator>::device_view::find_all(Key const& k,
                                                                     Hash hash,
                                                                     KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);
    // Key doesn't exist, return end()
    if (existing_key == this->get_empty_key_sentinel()) {
      return fancy_iterator{this->end(), k, *this};
    }

    // Key exists, return iterator to location
    if (key_equal(existing_key, k)) { return fancy_iterator{current_slot, k, *this}; }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ typename static_multimap<Key, Value, Scope, Allocator>::device_view::const_fancy_iterator
static_multimap<Key, Value, Scope, Allocator>::device_view::find_all(
  Key const& k, Hash hash, KeyEqual key_equal) const noexcept
{
  auto current_slot = initial_slot(k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);
    // Key doesn't exist, return end()
    if (existing_key == this->get_empty_key_sentinel()) {
      return const_fancy_iterator{this->end(), k, *this};
    }

    // Key exists, return iterator to location
    if (key_equal(existing_key, k)) { return const_fancy_iterator{current_slot, k, *this}; }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ typename static_multimap<Key, Value, Scope, Allocator>::device_view::fancy_iterator
static_multimap<Key, Value, Scope, Allocator>::device_view::find_all(CG g,
                                                                     Key const& k,
                                                                     Hash hash,
                                                                     KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as the
    // sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty = (existing_key == this->get_empty_key_sentinel());

    // the key we were searching for was found by one of the threads,
    // so we return an iterator to the entry
    auto const exists = g.ballot(not slot_is_empty and key_equal(existing_key, k));
    if (exists) {
      uint32_t src_lane = __ffs(exists) - 1;
      // TODO: This shouldn't cast an iterator to an int to shuffle. Instead, get the index of the
      // current_slot and shuffle that instead.
      intptr_t res_slot = g.shfl(reinterpret_cast<intptr_t>(current_slot), src_lane);
      return fancy_iterator{reinterpret_cast<iterator>(res_slot), k, *this};
    }

    // we found an empty slot, meaning that the key we're searching for isn't present
    if (g.ballot(slot_is_empty)) { return fancy_iterator{this->end(), k, *this}; }

    // otherwise, all slots in the current window are full with other keys, so we move onto the
    // next window
    current_slot = next_slot(g, current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ typename static_multimap<Key, Value, Scope, Allocator>::device_view::const_fancy_iterator
static_multimap<Key, Value, Scope, Allocator>::device_view::find_all(
  CG g, Key const& k, Hash hash, KeyEqual key_equal) const noexcept
{
  auto current_slot = initial_slot(g, k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as the
    // sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty = (existing_key == this->get_empty_key_sentinel());

    // the key we were searching for was found by one of the threads, so we return an iterator to
    // the entry
    auto const exists = g.ballot(not slot_is_empty and key_equal(existing_key, k));
    if (exists) {
      uint32_t src_lane = __ffs(exists) - 1;
      // TODO: This shouldn't cast an iterator to an int to shuffle. Instead, get the index of the
      // current_slot and shuffle that instead.
      intptr_t res_slot = g.shfl(reinterpret_cast<intptr_t>(current_slot), src_lane);
      return const_fancy_iterator{reinterpret_cast<iterator>(res_slot), k, *this};
    }

    // we found an empty slot, meaning that the key we're searching
    // for isn't in this submap, so we should move onto the next one
    if (g.ballot(slot_is_empty)) { return const_fancy_iterator{this->end(), k, *this}; }

    // otherwise, all slots in the current window are full with other keys,
    // so we move onto the next window in the current submap

    current_slot = next_slot(g, current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ bool static_multimap<Key, Value, Scope, Allocator>::device_view::contains(
  Key const& k, Hash hash, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    if (existing_key == empty_key_sentinel_) { return false; }

    if (key_equal(existing_key, k)) { return true; }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ bool static_multimap<Key, Value, Scope, Allocator>::device_view::contains(
  CG g, Key const& k, Hash hash, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k, hash);

  while (true) {
    key_type const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as the
    // sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty = (existing_key == this->get_empty_key_sentinel());

    // the key we were searching for was found by one of the threads, so we return an iterator to
    // the entry
    if (g.ballot(not slot_is_empty and key_equal(existing_key, k))) { return true; }

    // we found an empty slot, meaning that the key we're searching for isn't present
    if (g.ballot(slot_is_empty)) { return false; }

    // otherwise, all slots in the current window are full with other keys, so we move onto the next
    // window
    current_slot = next_slot(g, current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ size_t static_multimap<Key, Value, Scope, Allocator>::device_view::count(
  Key const& k, Hash hash, KeyEqual key_equal) noexcept
{
  auto found   = this->find_all(k, hash, key_equal);
  size_t count = 0;
  while (found != this->end()) {
    ++found;
    ++count;
  }
  return count;
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ size_t static_multimap<Key, Value, Scope, Allocator>::device_view::count(
  Key const& k, Hash hash, KeyEqual key_equal) const noexcept
{
  auto found   = this->find_all(k, hash, key_equal);
  size_t count = 0;
  while (found != this->end()) {
    ++found;
    ++count;
  }
  return count;
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ size_t static_multimap<Key, Value, Scope, Allocator>::device_view::count(
  CG g, Key const& k, Hash hash, KeyEqual key_equal) noexcept
{
  auto found   = this->find_all(g, k, hash, key_equal);
  size_t count = 0;
  while (found != this->end()) {
    ++found;
    ++count;
  }
  return count;
}

template <typename Key, typename Value, cuda::thread_scope Scope, typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ size_t static_multimap<Key, Value, Scope, Allocator>::device_view::count(
  CG g, Key const& k, Hash hash, KeyEqual key_equal) const noexcept
{
  auto found   = this->find_all(g, k, hash, key_equal);
  size_t count = 0;
  while (found != this->end()) {
    ++found;
    ++count;
  }
  return count;
}

}  // namespace cuco
