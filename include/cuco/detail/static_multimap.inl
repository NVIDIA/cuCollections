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

#include <cuco/detail/bitwise_compare.cuh>

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

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::static_multimap(
  std::size_t capacity, Key empty_key_sentinel, Value empty_value_sentinel, Allocator const& alloc)
  : empty_key_sentinel_{empty_key_sentinel},
    empty_value_sentinel_{empty_value_sentinel},
    slot_allocator_{alloc}
{
  if constexpr (is_vector_load()) {
    capacity_ = cuco::detail::get_valid_capacity<cg_size() * 2>(capacity);
  } else {
    capacity_ = cuco::detail::get_valid_capacity<cg_size()>(capacity);
  }

  slots_ = std::allocator_traits<slot_allocator_type>::allocate(slot_allocator_, get_capacity());

  auto constexpr block_size = 256;
  auto constexpr stride     = 4;
  auto const grid_size      = (get_capacity() + stride * block_size - 1) / (stride * block_size);
  detail::initialize<atomic_key_type, atomic_mapped_type>
    <<<grid_size, block_size>>>(slots_, empty_key_sentinel, empty_value_sentinel, get_capacity());
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::~static_multimap()
{
  std::allocator_traits<slot_allocator_type>::deallocate(slot_allocator_, slots_, capacity_);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename KeyEqual>
void static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::insert(InputIt first,
                                                                          InputIt last,
                                                                          cudaStream_t stream,
                                                                          KeyEqual key_equal)
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const grid_size  = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_mutable_view();

  detail::insert<block_size, cg_size(), is_vector_load()>
    <<<grid_size, block_size, 0, stream>>>(first, first + num_keys, view, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename OutputIt, typename KeyEqual>
void static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::contains(
  InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream, KeyEqual key_equal)
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const grid_size  = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_view();

  detail::contains<block_size, cg_size(), is_vector_load()>
    <<<grid_size, block_size, 0, stream>>>(first, last, output_begin, view, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename KeyEqual>
std::size_t static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::count(InputIt first,
                                                                                InputIt last,
                                                                                cudaStream_t stream,
                                                                                KeyEqual key_equal)
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const grid_size  = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_view();

  atomic_ctr_type* num_matches;
  CUCO_CUDA_TRY(cudaMallocManaged(&num_matches, sizeof(atomic_ctr_type)));
  *num_matches = 0;
  int device_id;
  CUCO_CUDA_TRY(cudaGetDevice(&device_id));
  CUCO_CUDA_TRY(cudaMemPrefetchAsync(num_matches, sizeof(atomic_ctr_type), device_id));

  detail::count<block_size, cg_size(), Key, Value, is_vector_load()>
    <<<grid_size, block_size, 0, stream>>>(first, last, num_matches, view, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  size_t result = *num_matches;
  CUCO_CUDA_TRY(cudaFree(num_matches));

  return result;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename KeyEqual>
std::size_t static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::count_outer(
  InputIt first, InputIt last, cudaStream_t stream, KeyEqual key_equal)
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const grid_size  = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_view();

  constexpr bool is_outer = true;

  atomic_ctr_type* num_matches;
  CUCO_CUDA_TRY(cudaMallocManaged(&num_matches, sizeof(atomic_ctr_type)));
  *num_matches = 0;
  int device_id;
  CUCO_CUDA_TRY(cudaGetDevice(&device_id));
  CUCO_CUDA_TRY(cudaMemPrefetchAsync(num_matches, sizeof(atomic_ctr_type), device_id));

  detail::count<block_size, cg_size(), Key, Value, is_vector_load(), is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, num_matches, view, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  size_t result = *num_matches;
  CUCO_CUDA_TRY(cudaFree(num_matches));

  return result;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename PairEqual>
std::size_t static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::pair_count(
  InputIt first, InputIt last, PairEqual pair_equal, cudaStream_t stream)
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const grid_size  = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_view();

  atomic_ctr_type* num_matches;
  CUCO_CUDA_TRY(cudaMallocManaged(&num_matches, sizeof(atomic_ctr_type)));
  *num_matches = 0;
  int device_id;
  CUCO_CUDA_TRY(cudaGetDevice(&device_id));
  CUCO_CUDA_TRY(cudaMemPrefetchAsync(num_matches, sizeof(atomic_ctr_type), device_id));

  detail::pair_count<block_size, cg_size(), Key, Value, is_vector_load()>
    <<<grid_size, block_size, 0, stream>>>(first, last, num_matches, view, pair_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  size_t result = *num_matches;
  CUCO_CUDA_TRY(cudaFree(num_matches));

  return result;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename PairEqual>
std::size_t static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::pair_count_outer(
  InputIt first, InputIt last, PairEqual pair_equal, cudaStream_t stream)
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  auto const stride     = 1;
  auto const grid_size  = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view             = get_device_view();

  constexpr bool is_outer = true;

  atomic_ctr_type* num_matches;
  CUCO_CUDA_TRY(cudaMallocManaged(&num_matches, sizeof(atomic_ctr_type)));
  *num_matches = 0;
  int device_id;
  CUCO_CUDA_TRY(cudaGetDevice(&device_id));
  CUCO_CUDA_TRY(cudaMemPrefetchAsync(num_matches, sizeof(atomic_ctr_type), device_id));

  detail::pair_count<block_size, cg_size(), Key, Value, is_vector_load(), is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, num_matches, view, pair_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  size_t result = *num_matches;
  CUCO_CUDA_TRY(cudaFree(num_matches));

  return result;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename OutputIt, typename KeyEqual>
OutputIt static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::retrieve(
  InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream, KeyEqual key_equal)
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  // Using per-warp buffer for vector loads and per-CG buffer for scalar loads
  auto const buffer_size = is_vector_load() ? (32u * 3u) : (cg_size() * 3u);
  auto const stride      = 1;
  auto const grid_size   = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view              = get_device_view();

  atomic_ctr_type* num_matches;
  CUCO_CUDA_TRY(cudaMallocManaged(&num_matches, sizeof(atomic_ctr_type)));
  *num_matches = 0;
  int device_id;
  CUCO_CUDA_TRY(cudaGetDevice(&device_id));
  CUCO_CUDA_TRY(cudaMemPrefetchAsync(num_matches, sizeof(atomic_ctr_type), device_id));

  detail::retrieve<block_size, cg_size(), buffer_size, Key, Value, is_vector_load()>
    <<<grid_size, block_size, 0, stream>>>(first, last, output_begin, num_matches, view, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  auto output_end = output_begin + *num_matches;
  CUCO_CUDA_TRY(cudaFree(num_matches));

  return output_end;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename OutputIt, typename KeyEqual>
OutputIt static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::retrieve_outer(
  InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream, KeyEqual key_equal)
{
  auto num_keys         = std::distance(first, last);
  auto const block_size = 128;
  // Using per-warp buffer for vector loads and per-CG buffer for scalar loads
  auto const buffer_size = is_vector_load() ? (32u * 3u) : (cg_size() * 3u);
  auto const stride      = 1;
  auto const grid_size   = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view              = get_device_view();

  constexpr bool is_outer = true;

  atomic_ctr_type* num_matches;
  CUCO_CUDA_TRY(cudaMallocManaged(&num_matches, sizeof(atomic_ctr_type)));
  *num_matches = 0;
  int device_id;
  CUCO_CUDA_TRY(cudaGetDevice(&device_id));
  CUCO_CUDA_TRY(cudaMemPrefetchAsync(num_matches, sizeof(atomic_ctr_type), device_id));

  detail::retrieve<block_size, cg_size(), buffer_size, Key, Value, is_vector_load(), is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, output_begin, num_matches, view, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  auto output_end = output_begin + *num_matches;
  CUCO_CUDA_TRY(cudaFree(num_matches));

  return output_end;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool is_vector_load, typename CG, typename KeyEqual>
__device__ std::enable_if_t<is_vector_load, void>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::insert(
  CG g, value_type const& insert_pair, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, insert_pair.first);
  while (true) {
    // key_type const existing_key = current_slot->first.load(cuda::memory_order_relaxed);
    pair<Key, Value> arr[2];
    if constexpr (sizeof(Key) == 4) {
      auto const tmp = *reinterpret_cast<uint4 const*>(current_slot);
      memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
    } else {
      auto const tmp = *reinterpret_cast<ulonglong4 const*>(current_slot);
      memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
    }

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as
    // the sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const first_slot_is_empty =
      (detail::bitwise_compare(arr[0].first, this->get_empty_value_sentinel()));
    auto const second_slot_is_empty =
      (detail::bitwise_compare(arr[1].first, this->get_empty_value_sentinel()));
    auto const window_contains_empty = g.ballot(first_slot_is_empty or second_slot_is_empty);
    if (window_contains_empty) {
      // the first lane in the group with an empty slot will attempt the insert
      insert_result status{insert_result::CONTINUE};
      uint32_t src_lane = __ffs(window_contains_empty) - 1;
      if (g.thread_rank() == src_lane) {
        using cuda::std::memory_order_relaxed;
        auto expected_key    = this->get_empty_key_sentinel();
        auto expected_value  = this->get_empty_value_sentinel();
        auto insert_location = first_slot_is_empty ? current_slot : current_slot + 1;

        if constexpr (sizeof(Key) == 4 and sizeof(Value) == 4) {
          static_assert(sizeof(cuda::atomic<uint64_t>) ==
                        (sizeof(cuda::atomic<Key>) + sizeof(cuda::atomic<Value>)));

          cuco::pair2uint64<Key, Value> converter;
          auto slot = reinterpret_cast<cuda::atomic<uint64_t>*>(insert_location);

          converter.pair =
            cuco::make_pair<Key, Value>(std::move(expected_key), std::move(expected_value));
          auto empty_sentinel = converter.uint64;
          converter.pair      = insert_pair;
          auto tmp_pair       = converter.uint64;

          bool success =
            slot->compare_exchange_strong(empty_sentinel, tmp_pair, memory_order_relaxed);
          if (success) { status = insert_result::SUCCESS; }
        } else {
          auto& slot_key   = insert_location->first;
          auto& slot_value = insert_location->second;
          bool key_success =
            slot_key.compare_exchange_strong(expected_key, insert_pair.first, memory_order_relaxed);
          bool value_success = slot_value.compare_exchange_strong(
            expected_value, insert_pair.second, memory_order_relaxed);
          if (key_success) {
            while (not value_success) {
              value_success = slot_value.compare_exchange_strong(
                expected_value = this->get_empty_value_sentinel(),
                insert_pair.second,
                memory_order_relaxed);
            }
            status = insert_result::SUCCESS;
          } else if (value_success) {
            slot_value.store(this->get_empty_value_sentinel(), memory_order_relaxed);
          }
        }
        // another key was inserted in both slots we wanted to try
        // so we need to try the next empty slots in the window
      }

      // successful insert
      if (g.any(status == insert_result::SUCCESS)) { return; }
      // if we've gotten this far, a different key took our spot
      // before we could insert. We need to retry the insert on the
      // same window
    }
    // if there are no empty slots in the current window,
    // we move onto the next window
    else {
      current_slot = next_slot(current_slot);
    }
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool is_vector_load, typename CG, typename KeyEqual>
__device__ std::enable_if_t<not is_vector_load, void>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::insert(
  CG g, value_type const& insert_pair, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, insert_pair.first);

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

        if constexpr (sizeof(Key) == 4 and sizeof(Value) == 4) {
          static_assert(sizeof(cuda::atomic<uint64_t>) ==
                        (sizeof(cuda::atomic<Key>) + sizeof(cuda::atomic<Value>)));

          cuco::pair2uint64<Key, Value> converter;
          auto slot = reinterpret_cast<cuda::atomic<uint64_t>*>(current_slot);

          converter.pair =
            cuco::make_pair<Key, Value>(std::move(expected_key), std::move(expected_value));
          auto empty_sentinel = converter.uint64;
          converter.pair      = insert_pair;
          auto tmp_pair       = converter.uint64;

          bool success =
            slot->compare_exchange_strong(empty_sentinel, tmp_pair, memory_order_relaxed);
          if (success) { status = insert_result::SUCCESS; }
        } else {
          auto& slot_key   = current_slot->first;
          auto& slot_value = current_slot->second;

          bool key_success =
            slot_key.compare_exchange_strong(expected_key, insert_pair.first, memory_order_relaxed);
          bool value_success = slot_value.compare_exchange_strong(
            expected_value, insert_pair.second, memory_order_relaxed);

          if (key_success) {
            while (not value_success) {
              value_success = slot_value.compare_exchange_strong(
                expected_value = this->get_empty_value_sentinel(),
                insert_pair.second,
                memory_order_relaxed);
            }
            status = insert_result::SUCCESS;
          } else if (value_success) {
            slot_value.store(this->get_empty_value_sentinel(), memory_order_relaxed);
          }
        }
        // another key was inserted in the slot we wanted to try
        // so we need to try the next empty slot in the window
      }

      // successful insert
      if (g.any(status == insert_result::SUCCESS)) { return; }
      // if we've gotten this far, a different key took our spot
      // before we could insert. We need to retry the insert on the
      // same window
    }
    // if there are no empty slots in the current window,
    // we move onto the next window
    else {
      current_slot = next_slot(current_slot);
    }
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool is_vector_load, typename CG, typename KeyEqual>
__device__ std::enable_if_t<is_vector_load, bool>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::contains(
  CG g, Key const& k, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k);

  while (true) {
    pair<Key, Value> arr[2];
    if constexpr (sizeof(Key) == 4) {
      auto const tmp = *reinterpret_cast<uint4 const*>(current_slot);
      memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
    } else {
      auto const tmp = *reinterpret_cast<ulonglong4 const*>(current_slot);
      memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
    }

    auto const first_slot_is_empty  = (arr[0].first == this->get_empty_key_sentinel());
    auto const second_slot_is_empty = (arr[1].first == this->get_empty_key_sentinel());
    auto const first_equals         = (not first_slot_is_empty and key_equal(arr[0].first, k));
    auto const second_equals        = (not second_slot_is_empty and key_equal(arr[1].first, k));

    // the key we were searching for was found by one of the threads, so we return true
    if (g.any(first_equals or second_equals)) { return true; }

    // we found an empty slot, meaning that the key we're searching for isn't present
    if (g.any(first_slot_is_empty or second_slot_is_empty)) { return false; }

    // otherwise, all slots in the current window are full with other keys, so we move onto the next
    // window
    current_slot = next_slot(current_slot);
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool is_vector_load, typename CG, typename KeyEqual>
__device__ std::enable_if_t<not is_vector_load, bool>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::contains(
  CG g, Key const& k, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k);

  while (true) {
    key_type const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as the
    // sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty = (existing_key == this->get_empty_key_sentinel());

    auto const equals = (not slot_is_empty and key_equal(existing_key, k));

    // the key we were searching for was found by one of the threads, so we return true
    if (g.any(equals)) { return true; }

    // we found an empty slot, meaning that the key we're searching for isn't present
    if (g.any(slot_is_empty)) { return false; }

    // otherwise, all slots in the current window are full with other keys, so we move onto the next
    // window
    current_slot = next_slot(current_slot);
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool is_vector_load, bool is_outer, typename CG, typename KeyEqual>
__device__ std::enable_if_t<is_vector_load, void>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::count(
  CG const& g, std::size_t& thread_num_matches, Key const& k, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k);

  if constexpr (is_outer) {
    bool found_match = false;

    while (true) {
      pair<Key, Value> arr[2];
      if constexpr (sizeof(Key) == 4) {
        auto const tmp = *reinterpret_cast<uint4 const*>(current_slot);
        memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
      } else {
        auto const tmp = *reinterpret_cast<ulonglong4 const*>(current_slot);
        memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
      }

      auto const first_slot_is_empty  = (arr[0].first == this->get_empty_key_sentinel());
      auto const second_slot_is_empty = (arr[1].first == this->get_empty_key_sentinel());
      auto const first_equals         = (not first_slot_is_empty and key_equal(arr[0].first, k));
      auto const second_equals        = (not second_slot_is_empty and key_equal(arr[1].first, k));

      if (g.any(first_equals or second_equals)) { found_match = true; }

      thread_num_matches += (first_equals + second_equals);

      if (g.any(first_slot_is_empty or second_slot_is_empty)) {
        if ((not found_match) && (g.thread_rank() == 0)) { thread_num_matches++; }
        return;
      }

      current_slot = next_slot(current_slot);
    }
  } else {
    while (true) {
      pair<Key, Value> arr[2];
      if constexpr (sizeof(Key) == 4) {
        auto const tmp = *reinterpret_cast<uint4 const*>(current_slot);
        memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
      } else {
        auto const tmp = *reinterpret_cast<ulonglong4 const*>(current_slot);
        memcpy(&arr[0], &tmp, 2 * sizeof(pair<Key, Value>));
      }

      auto const first_slot_is_empty  = (arr[0].first == this->get_empty_key_sentinel());
      auto const second_slot_is_empty = (arr[1].first == this->get_empty_key_sentinel());
      auto const first_equals         = (not first_slot_is_empty and key_equal(arr[0].first, k));
      auto const second_equals        = (not second_slot_is_empty and key_equal(arr[1].first, k));

      thread_num_matches += (first_equals + second_equals);

      if (g.any(first_slot_is_empty or second_slot_is_empty)) { return; }

      current_slot = next_slot(current_slot);
    }
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool is_vector_load, bool is_outer, typename CG, typename KeyEqual>
__device__ std::enable_if_t<not is_vector_load, void>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::count(
  CG const& g, std::size_t& thread_num_matches, Key const& k, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k);

  if constexpr (is_outer) {
    bool found_match = false;

    while (true) {
      pair<Key, Value> slot_contents =
        *reinterpret_cast<cuco::pair_type<Key, Value> const*>(current_slot);
      auto const& current_key = slot_contents.first;

      auto const slot_is_empty = (current_key == this->get_empty_key_sentinel());
      auto const equals        = not slot_is_empty and key_equal(current_key, k);

      if (g.any(equals)) { found_match = true; }

      thread_num_matches += equals;

      if (g.any(slot_is_empty)) {
        if ((not found_match) && (g.thread_rank() == 0)) { thread_num_matches++; }
        break;
      }

      current_slot = next_slot(current_slot);
    }
  } else {
    while (true) {
      pair<Key, Value> slot_contents =
        *reinterpret_cast<cuco::pair_type<Key, Value> const*>(current_slot);
      auto const& current_key = slot_contents.first;

      auto const slot_is_empty = (current_key == this->get_empty_key_sentinel());
      auto const equals        = not slot_is_empty and key_equal(current_key, k);

      thread_num_matches += equals;

      if (g.any(slot_is_empty)) { break; }

      current_slot = next_slot(current_slot);
    }
  }
}
}  // namespace cuco
