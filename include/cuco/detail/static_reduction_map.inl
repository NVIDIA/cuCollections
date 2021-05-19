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

template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator>
static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::static_reduction_map(
  std::size_t capacity, Key empty_key_sentinel, ReductionOp reduction_op, Allocator const& alloc)
  : capacity_{capacity},
    empty_key_sentinel_{empty_key_sentinel},
    empty_value_sentinel_{ReductionOp::identity},
    op_{reduction_op},
    slot_allocator_{alloc}
{
  slots_ = std::allocator_traits<slot_allocator_type>::allocate(slot_allocator_, capacity);

  auto constexpr block_size = 256;
  auto constexpr stride     = 4;
  auto const grid_size      = (capacity + stride * block_size - 1) / (stride * block_size);
  detail::initialize<atomic_key_type, atomic_mapped_type><<<grid_size, block_size>>>(
    slots_, get_empty_key_sentinel(), get_empty_value_sentinel(), get_capacity());

  CUCO_CUDA_TRY(cudaMallocManaged(&num_successes_, sizeof(atomic_ctr_type)));
}

template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator>
static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::~static_reduction_map()
{
  std::allocator_traits<slot_allocator_type>::deallocate(slot_allocator_, slots_, capacity_);
  CUCO_CUDA_TRY(cudaFree(num_successes_));
}

template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename Hash, typename KeyEqual>
void static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::insert(InputIt first,
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

  *num_successes_ = 0;
  int device_id;
  CUCO_CUDA_TRY(cudaGetDevice(&device_id));
  CUCO_CUDA_TRY(cudaMemPrefetchAsync(num_successes_, sizeof(atomic_ctr_type), device_id));

  detail::insert<block_size, tile_size>
    <<<grid_size, block_size>>>(first, first + num_keys, num_successes_, view, hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  size_ += num_successes_->load(cuda::std::memory_order_relaxed);
}

template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::find(
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

namespace detail {
template <typename Key, typename Value>
struct slot_to_tuple {
  template <typename S>
  __device__ thrust::tuple<Key, Value> operator()(S const& s)
  {
    return thrust::tuple<Key, Value>(s.first, s.second);
  }
};

template <typename Key>
struct slot_is_filled {
  Key empty_key_sentinel;
  template <typename S>
  __device__ bool operator()(S const& s)
  {
    return thrust::get<0>(s) != empty_key_sentinel;
  }
};
}  // namespace detail

template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename KeyOut, typename ValueOut>
void static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::retrieve_all(
  KeyOut keys_out, ValueOut values_out)
{
  // Convert pair_type to thrust::tuple to allow assigning to a zip iterator
  auto begin      = thrust::make_transform_iterator(raw_slots_begin(), detail::slot_to_tuple<Key, Value>{});
  auto end        = begin + get_capacity();
  auto filled     = detail::slot_is_filled<Key>{get_empty_key_sentinel()};
  auto zipped_out = thrust::make_zip_iterator(thrust::make_tuple(keys_out, values_out));

  thrust::copy_if(thrust::device, begin, end, zipped_out, filled);
}

template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
void static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::contains(
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

template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ Value
static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::device_mutable_view::insert(
  value_type const& insert_pair, Hash hash, KeyEqual key_equal) noexcept
{
  auto current_slot{initial_slot(insert_pair.first, hash)};

  while (true) {
    using cuda::std::memory_order_relaxed;
    auto expected_key   = this->get_empty_key_sentinel();
    auto expected_value = this->get_empty_value_sentinel();
    auto& slot_key      = current_slot->first;
    auto& slot_value    = current_slot->second;

    auto const key_success =
      slot_key.compare_exchange_strong(expected_key, insert_pair.first, memory_order_relaxed);

    if (key_success or key_equal(insert_pair.first, expected_key)) {
      return this->get_op().apply(slot_value, insert_pair.second);
    }

    // if we couldn't insert the key, but it wasn't a duplicate, then there must
    // have been some other key there, so we keep looking for a slot
    current_slot = next_slot(current_slot);
  }
}

template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ bool
static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::device_mutable_view::insert(
  CG g, value_type const& insert_pair, Hash hash, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, insert_pair.first, hash);

  while (true) {
    auto& slot_key         = current_slot->first;
    auto& slot_value       = current_slot->second;
    auto const current_key = slot_key.load(cuda::std::memory_order_relaxed);

    // The user provided `key_equal` should never be used to compare against `empty_key_sentinel` as
    // the sentinel is not a valid key value. Therefore, first check for the sentinel
    // TODO: Use memcmp
    auto const slot_is_empty = (current_key == this->get_empty_key_sentinel());

    auto const key_exists = not slot_is_empty and key_equal(current_key, insert_pair.first);

    // Key already exists, aggregate with it's value
    if (key_exists) { this->get_op().apply(slot_value, insert_pair.second); }

    // If key already exists in the CG window, all threads exit
    if (g.ballot(key_exists)) { return false; }

    auto const window_empty_mask = g.ballot(slot_is_empty);

    if (window_empty_mask) {
      // the first lane in the group with an empty slot will attempt the insert
      auto const src_lane = __ffs(window_empty_mask) - 1;

      auto const attempt_update = [&]() {
        auto expected_key = this->get_empty_key_sentinel();

        auto const key_success = slot_key.compare_exchange_strong(
          expected_key, insert_pair.first, cuda::memory_order_relaxed);

        if (key_success or key_equal(insert_pair.first, expected_key)) {
          this->get_op().apply(slot_value, insert_pair.second);
          return key_success ? insert_result::SUCCESS : insert_result::DUPLICATE;
        }
        return insert_result::CONTINUE;
      };

      auto const update_result =
        (g.thread_rank() == src_lane) ? attempt_update() : insert_result::CONTINUE;

      auto const window_result = g.shfl(update_result, src_lane);

      // If the update succeeded, the thread group exits
      if (window_result != insert_result::CONTINUE) {
        return (window_result == insert_result::SUCCESS);
      }

      // A different key took the current slot. Look for an empty slot in the current window
    } else {
      // No empty slots in the current window, move onto the next window
      current_slot = next_slot(g, current_slot);
    }
  }
}

template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename Hash, typename KeyEqual>
__device__
  typename static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::device_view::iterator
  static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::device_view::find(
    Key const& k, Hash hash, KeyEqual key_equal) noexcept
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

template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ typename static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::device_view::
  const_iterator
  static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::device_view::find(
    Key const& k, Hash hash, KeyEqual key_equal) const noexcept
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

template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__
  typename static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::device_view::iterator
  static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::device_view::find(
    CG g, Key const& k, Hash hash, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as
    // the sentinel is not a valid key value. Therefore, first check for the sentinel
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

template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ typename static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::device_view::
  const_iterator
  static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::device_view::find(
    CG g, Key const& k, Hash hash, KeyEqual key_equal) const noexcept
{
  auto current_slot = initial_slot(g, k, hash);

  while (true) {
    auto const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as
    // the sentinel is not a valid key value. Therefore, first check for the sentinel
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

template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename Hash, typename KeyEqual>
__device__ bool
static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::device_view::contains(
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

template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename Hash, typename KeyEqual>
__device__ bool
static_reduction_map<ReductionOp, Key, Value, Scope, Allocator>::device_view::contains(
  CG g, Key const& k, Hash hash, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k, hash);

  while (true) {
    key_type const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as
    // the sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty = (existing_key == this->get_empty_key_sentinel());

    // the key we were searching for was found by one of the threads, so we return an iterator to
    // the entry
    if (g.ballot(not slot_is_empty and key_equal(existing_key, k))) { return true; }

    // we found an empty slot, meaning that the key we're searching for isn't present
    if (g.ballot(slot_is_empty)) { return false; }

    // otherwise, all slots in the current window are full with other keys, so we move onto the
    // next window
    current_slot = next_slot(g, current_slot);
  }
}
}  // namespace cuco
