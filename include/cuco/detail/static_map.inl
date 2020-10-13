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
static_map<Key, Value, Scope>::static_map(std::size_t capacity,
                                          Key empty_key_sentinel,
                                          Value empty_value_sentinel)
  : capacity_{capacity},
    empty_key_sentinel_{empty_key_sentinel},
    empty_value_sentinel_{empty_value_sentinel}
{
  CUCO_CUDA_TRY(cudaMalloc(&slots_, capacity * sizeof(pair_atomic_type)));

  auto constexpr block_size = 256;
  auto constexpr stride     = 4;
  auto const grid_size      = (capacity + stride * block_size - 1) / (stride * block_size);
  detail::initialize<atomic_key_type, atomic_mapped_type>
    <<<grid_size, block_size>>>(slots_, empty_key_sentinel, empty_value_sentinel, capacity);

  CUCO_CUDA_TRY(cudaMallocManaged(&num_successes_, sizeof(atomic_ctr_type)));
}

template <typename Key, typename Value, cuda::thread_scope Scope>
static_map<Key, Value, Scope>::~static_map()
{
  CUCO_CUDA_TRY(cudaFree(slots_));
  CUCO_CUDA_TRY(cudaFree(num_successes_));
}

template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename InputIt, typename Hash, typename KeyEqual>
void static_map<Key, Value, Scope>::insert(InputIt first,
                                           InputIt last,
                                           Hash hash,
                                           KeyEqual key_equal)
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const tile_size  = 8;
  auto const grid_size  = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_mutable_view();

  *num_successes_ = 0;
  int device_id;
  CUCO_CUDA_TRY(cudaGetDevice(&device_id));
  CUCO_CUDA_TRY(cudaMemPrefetchAsync(num_successes_, sizeof(atomic_ctr_type), device_id));

  detail::insert<block_size, tile_size>
    <<<grid_size, block_size>>>(first, first + num_keys, num_successes_, view, hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  size_ += num_successes_->load(cuda::std::memory_order_relaxed);
}

template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void static_map<Key, Value, Scope>::find(
  InputIt first, InputIt last, OutputIt output_begin, Hash hash, KeyEqual key_equal) noexcept
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const tile_size  = 8;
  auto const grid_size  = (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_view();

  detail::find<block_size, tile_size, Value>
    <<<grid_size, block_size>>>(first, last, output_begin, view, hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}

template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void static_map<Key, Value, Scope>::contains(
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

template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename Hash, typename KeyEqual>
__device__ bool static_map<Key, Value, Scope>::device_mutable_view::insert(
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
      return true;
    } else if (value_success) {
      slot_value.store(this->get_empty_value_sentinel(), memory_order_relaxed);
    }

    // if the key was already inserted by another thread, than this instance is a
    // duplicate, so the insert fails
    if (key_equal(insert_pair.first, expected_key)) { return false; }

    // if we couldn't insert the key, but it wasn't a duplicate, then there must
    // have been some other key there, so we keep looking for a slot
    current_slot = next_slot(current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename CG, typename Hash, typename KeyEqual>
__device__ bool static_map<Key, Value, Scope>::device_mutable_view::insert(
  CG g, value_type const& insert_pair, Hash hash, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, insert_pair.first, hash);

  while (true) {
    key_type const existing_key = current_slot->first;
    uint32_t existing           = g.ballot(key_equal(existing_key, insert_pair.first));

    // the key we are trying to insert is already in the map, so we return
    // with failure to insert
    if (existing) { return false; }

    uint32_t empty = g.ballot(existing_key == this->get_empty_key_sentinel());

    // we found an empty slot, but not the key we are inserting, so this must
    // be an empty slot into which we can insert the key
    if (empty) {
      // the first lane in the group with an empty slot will attempt the insert
      insert_result status{insert_result::CONTINUE};
      uint32_t src_lane = __ffs(empty) - 1;

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

        // our key was already present in the slot, so our key is a duplicate
        if (key_equal(insert_pair.first, expected_key)) { status = insert_result::DUPLICATE; }
        // another key was inserted in the slot we wanted to try
        // so we need to try the next empty slot in the window
      }

      uint32_t res_status = g.shfl(static_cast<uint32_t>(status), src_lane);
      status              = static_cast<insert_result>(res_status);

      // successful insert
      if (status == insert_result::SUCCESS) { return true; }
      // duplicate present during insert
      if (status == insert_result::DUPLICATE) { return false; }
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
template <typename Hash, typename KeyEqual>
__device__ typename static_map<Key, Value, Scope>::device_view::iterator
static_map<Key, Value, Scope>::device_view::find(Key const& k,
                                                 Hash hash,
                                                 KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);
    // Key exists, return iterator to location
    if (key_equal(existing_key, k)) { return current_slot; }

    // Key doesn't exist, return end()
    if (existing_key == this->get_empty_key_sentinel()) { return this->end(); }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename Hash, typename KeyEqual>
__device__ typename static_map<Key, Value, Scope>::device_view::const_iterator
static_map<Key, Value, Scope>::device_view::find(Key const& k, Hash hash, KeyEqual key_equal) const
  noexcept
{
  auto current_slot = initial_slot(k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);
    // Key exists, return iterator to location
    if (key_equal(existing_key, k)) { return current_slot; }

    // Key doesn't exist, return end()
    if (existing_key == this->get_empty_key_sentinel()) { return this->end(); }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename CG, typename Hash, typename KeyEqual>
__device__ typename static_map<Key, Value, Scope>::device_view::iterator
static_map<Key, Value, Scope>::device_view::find(CG g,
                                                 Key const& k,
                                                 Hash hash,
                                                 KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);
    uint32_t existing       = g.ballot(key_equal(existing_key, k));

    // the key we were searching for was found by one of the threads,
    // so we return an iterator to the entry
    if (existing) {
      uint32_t src_lane = __ffs(existing) - 1;
      // TODO: This shouldn't cast an iterator to an int to shuffle. Instead, get the index of the
      // current_slot and shuffle that instead.
      intptr_t res_slot = g.shfl(reinterpret_cast<intptr_t>(current_slot), src_lane);
      return reinterpret_cast<iterator>(res_slot);
    }

    // we found an empty slot, meaning that the key we're searching
    // for isn't in this submap, so we should move onto the next one
    uint32_t empty = g.ballot(existing_key == this->get_empty_key_sentinel());
    if (empty) { return this->end(); }

    // otherwise, all slots in the current window are full with other keys,
    // so we move onto the next window in the current submap

    current_slot = next_slot(g, current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename CG, typename Hash, typename KeyEqual>
__device__ typename static_map<Key, Value, Scope>::device_view::const_iterator
static_map<Key, Value, Scope>::device_view::find(CG g,
                                                 Key const& k,
                                                 Hash hash,
                                                 KeyEqual key_equal) const noexcept
{
  auto current_slot = initial_slot(g, k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);
    uint32_t existing       = g.ballot(key_equal(existing_key, k));

    // the key we were searching for was found by one of the threads,
    // so we return an iterator to the entry
    if (existing) {
      uint32_t src_lane = __ffs(existing) - 1;
      // TODO: This shouldn't cast an iterator to an int to shuffle. Instead, get the index of the
      // current_slot and shuffle that instead.
      intptr_t res_slot = g.shfl(reinterpret_cast<intptr_t>(current_slot), src_lane);
      return reinterpret_cast<const_iterator>(res_slot);
    }

    // we found an empty slot, meaning that the key we're searching
    // for isn't in this submap, so we should move onto the next one
    uint32_t empty = g.ballot(existing_key == this->get_empty_key_sentinel());
    if (empty) { return this->end(); }

    // otherwise, all slots in the current window are full with other keys,
    // so we move onto the next window in the current submap

    current_slot = next_slot(g, current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename Hash, typename KeyEqual>
__device__ bool static_map<Key, Value, Scope>::device_view::contains(Key const& k,
                                                                     Hash hash,
                                                                     KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    if (key_equal(existing_key, k)) { return true; }

    if (existing_key == this->get_empty_key_sentinel()) { return false; }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename CG, typename Hash, typename KeyEqual>
__device__ bool static_map<Key, Value, Scope>::device_view::contains(CG g,
                                                                     Key const& k,
                                                                     Hash hash,
                                                                     KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k, hash);

  while (true) {
    key_type const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);
    uint32_t existing           = g.ballot(key_equal(existing_key, k));

    // the key we were searching for was found by one of the threads,
    // so we return an iterator to the entry
    if (existing) { return true; }

    // we found an empty slot, meaning that the key we're searching
    // for isn't in this submap, so we should move onto the next one
    uint32_t empty = g.ballot(existing_key == this->get_empty_key_sentinel());
    if (empty) { return false; }

    // otherwise, all slots in the current window are full with other keys,
    // so we move onto the next window in the current submap
    current_slot = next_slot(g, current_slot);
  }
}
}  // namespace cuco
