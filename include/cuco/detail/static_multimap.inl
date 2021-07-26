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

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::static_multimap(
  std::size_t capacity, Key empty_key_sentinel, Value empty_value_sentinel, Allocator const& alloc)
  : empty_key_sentinel_{empty_key_sentinel},
    empty_value_sentinel_{empty_value_sentinel},
    slot_allocator_{alloc},
    counter_allocator_{alloc}
{
  if constexpr (uses_vector_load()) {
    capacity_ = cuco::detail::get_valid_capacity<cg_size() * vector_width()>(capacity);
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

  detail::insert<block_size, cg_size()>
    <<<grid_size, block_size, 0, stream>>>(first, first + num_keys, view, key_equal);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
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

  detail::contains<block_size, cg_size()>
    <<<grid_size, block_size, 0, stream>>>(first, last, output_begin, view, key_equal);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
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

  constexpr bool is_outer = false;

  atomic_ctr_type *h_num_matches, *d_num_matches;

  CUCO_CUDA_TRY(cudaMallocHost((void**)&h_num_matches, sizeof(atomic_ctr_type)));
  d_num_matches = std::allocator_traits<counter_allocator_type>::allocate(counter_allocator_, 1);

  h_num_matches->store(0ull, cuda::std::memory_order_relaxed);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    d_num_matches, h_num_matches, sizeof(atomic_ctr_type), cudaMemcpyHostToDevice, stream));

  detail::count<block_size, cg_size(), Key, Value, is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, d_num_matches, view, key_equal);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    h_num_matches, d_num_matches, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  auto result = h_num_matches->load(cuda::std::memory_order_relaxed);
  CUCO_CUDA_TRY(cudaFreeHost(h_num_matches));
  std::allocator_traits<counter_allocator_type>::deallocate(counter_allocator_, d_num_matches, 1);

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

  atomic_ctr_type *h_num_matches, *d_num_matches;

  CUCO_CUDA_TRY(cudaMallocHost((void**)&h_num_matches, sizeof(atomic_ctr_type)));
  d_num_matches = std::allocator_traits<counter_allocator_type>::allocate(counter_allocator_, 1);

  h_num_matches->store(0ull, cuda::std::memory_order_relaxed);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    d_num_matches, h_num_matches, sizeof(atomic_ctr_type), cudaMemcpyHostToDevice, stream));

  detail::count<block_size, cg_size(), Key, Value, is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, d_num_matches, view, key_equal);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    h_num_matches, d_num_matches, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  auto result = h_num_matches->load(cuda::std::memory_order_relaxed);
  CUCO_CUDA_TRY(cudaFreeHost(h_num_matches));
  std::allocator_traits<counter_allocator_type>::deallocate(counter_allocator_, d_num_matches, 1);

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

  constexpr bool is_outer = false;

  atomic_ctr_type *h_num_matches, *d_num_matches;

  CUCO_CUDA_TRY(cudaMallocHost((void**)&h_num_matches, sizeof(atomic_ctr_type)));
  d_num_matches = std::allocator_traits<counter_allocator_type>::allocate(counter_allocator_, 1);

  h_num_matches->store(0ull, cuda::std::memory_order_relaxed);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    d_num_matches, h_num_matches, sizeof(atomic_ctr_type), cudaMemcpyHostToDevice, stream));

  detail::pair_count<block_size, cg_size(), Key, Value, is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, d_num_matches, view, pair_equal);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    h_num_matches, d_num_matches, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  auto result = h_num_matches->load(cuda::std::memory_order_relaxed);
  CUCO_CUDA_TRY(cudaFreeHost(h_num_matches));
  std::allocator_traits<counter_allocator_type>::deallocate(counter_allocator_, d_num_matches, 1);

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

  atomic_ctr_type *h_num_matches, *d_num_matches;

  CUCO_CUDA_TRY(cudaMallocHost((void**)&h_num_matches, sizeof(atomic_ctr_type)));
  d_num_matches = std::allocator_traits<counter_allocator_type>::allocate(counter_allocator_, 1);

  h_num_matches->store(0ull, cuda::std::memory_order_relaxed);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    d_num_matches, h_num_matches, sizeof(atomic_ctr_type), cudaMemcpyHostToDevice, stream));

  detail::pair_count<block_size, cg_size(), Key, Value, is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, d_num_matches, view, pair_equal);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    h_num_matches, d_num_matches, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  auto result = h_num_matches->load(cuda::std::memory_order_relaxed);
  CUCO_CUDA_TRY(cudaFreeHost(h_num_matches));
  std::allocator_traits<counter_allocator_type>::deallocate(counter_allocator_, d_num_matches, 1);

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
  auto const buffer_size = uses_vector_load() ? (warp_size() * 3u) : (cg_size() * 3u);
  auto const stride      = 1;
  auto const grid_size   = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view              = get_device_view();

  constexpr bool is_outer = false;

  atomic_ctr_type *h_num_matches, *d_num_matches;

  CUCO_CUDA_TRY(cudaMallocHost((void**)&h_num_matches, sizeof(atomic_ctr_type)));
  d_num_matches = std::allocator_traits<counter_allocator_type>::allocate(counter_allocator_, 1);

  h_num_matches->store(0ull, cuda::std::memory_order_relaxed);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    d_num_matches, h_num_matches, sizeof(atomic_ctr_type), cudaMemcpyHostToDevice, stream));

  if constexpr (uses_vector_load()) {
    detail::
      vectorized_retrieve<block_size, warp_size(), cg_size(), buffer_size, Key, Value, is_outer>
      <<<grid_size, block_size, 0, stream>>>(
        first, last, output_begin, d_num_matches, view, key_equal);
  } else {
    detail::retrieve<block_size, warp_size(), cg_size(), buffer_size, Key, Value, is_outer>
      <<<grid_size, block_size, 0, stream>>>(
        first, last, output_begin, d_num_matches, view, key_equal);
  }
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    h_num_matches, d_num_matches, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  auto output_end = output_begin + h_num_matches->load(cuda::std::memory_order_relaxed);
  CUCO_CUDA_TRY(cudaFreeHost(h_num_matches));
  std::allocator_traits<counter_allocator_type>::deallocate(counter_allocator_, d_num_matches, 1);

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
  auto const buffer_size = uses_vector_load() ? (warp_size() * 3u) : (cg_size() * 3u);
  auto const stride      = 1;
  auto const grid_size   = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view              = get_device_view();

  constexpr bool is_outer = true;

  atomic_ctr_type *h_num_matches, *d_num_matches;

  CUCO_CUDA_TRY(cudaMallocHost((void**)&h_num_matches, sizeof(atomic_ctr_type)));
  d_num_matches = std::allocator_traits<counter_allocator_type>::allocate(counter_allocator_, 1);

  h_num_matches->store(0ull, cuda::std::memory_order_relaxed);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    d_num_matches, h_num_matches, sizeof(atomic_ctr_type), cudaMemcpyHostToDevice, stream));

  if constexpr (uses_vector_load()) {
    detail::
      vectorized_retrieve<block_size, warp_size(), cg_size(), buffer_size, Key, Value, is_outer>
      <<<grid_size, block_size, 0, stream>>>(
        first, last, output_begin, d_num_matches, view, key_equal);
  } else {
    detail::retrieve<block_size, warp_size(), cg_size(), buffer_size, Key, Value, is_outer>
      <<<grid_size, block_size, 0, stream>>>(
        first, last, output_begin, d_num_matches, view, key_equal);
  }
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    h_num_matches, d_num_matches, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  auto output_end = output_begin + h_num_matches->load(cuda::std::memory_order_relaxed);
  CUCO_CUDA_TRY(cudaFreeHost(h_num_matches));
  std::allocator_traits<counter_allocator_type>::deallocate(counter_allocator_, d_num_matches, 1);

  return output_end;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view_base::load_pair_array(
  value_type* arr, const_iterator current_slot) noexcept
{
  if constexpr (sizeof(value_type) == 4) {
    auto const tmp = *reinterpret_cast<ushort4 const*>(current_slot);
    memcpy(&arr[0], &tmp, 2 * sizeof(value_type));
  } else {
    auto const tmp = *reinterpret_cast<uint4 const*>(current_slot);
    memcpy(&arr[0], &tmp, 2 * sizeof(value_type));
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename KeyEqual>
__device__
  static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::insert_result
  static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::packed_cas(
    iterator current_slot, value_type const& insert_pair, KeyEqual key_equal) noexcept
{
  auto expected_key   = this->get_empty_key_sentinel();
  auto expected_value = this->get_empty_value_sentinel();

  cuco::detail::pair_converter<value_type> expected_pair{
    cuco::make_pair<Key, Value>(std::move(expected_key), std::move(expected_value))};
  cuco::detail::pair_converter<value_type> new_pair{insert_pair};

  auto slot =
    reinterpret_cast<cuda::atomic<typename cuco::detail::pair_converter<value_type>::packed_type>*>(
      current_slot);

  bool success = slot->compare_exchange_strong(
    expected_pair.packed, new_pair.packed, cuda::std::memory_order_relaxed);
  if (success) { return insert_result::SUCCESS; }

  return insert_result::CONTINUE;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename KeyEqual>
__device__
  static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::insert_result
  static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::
    back_to_back_cas(iterator current_slot,
                     value_type const& insert_pair,
                     KeyEqual key_equal) noexcept
{
  using cuda::std::memory_order_relaxed;

  auto expected_key   = this->get_empty_key_sentinel();
  auto expected_value = this->get_empty_value_sentinel();

  // Back-to-back CAS for 8B/8B key/value pairs
  auto& slot_key   = current_slot->first;
  auto& slot_value = current_slot->second;

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
    return insert_result::SUCCESS;
  } else if (value_success) {
    slot_value.store(this->get_empty_value_sentinel(), memory_order_relaxed);
  }

  return insert_result::CONTINUE;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename KeyEqual>
__device__
  static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::insert_result
  static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::
    cas_dependent_write(iterator current_slot,
                        value_type const& insert_pair,
                        KeyEqual key_equal) noexcept
{
  using cuda::std::memory_order_relaxed;
  auto expected_key = this->get_empty_key_sentinel();

  auto& slot_key = current_slot->first;

  auto const key_success =
    slot_key.compare_exchange_strong(expected_key, insert_pair.first, memory_order_relaxed);

  if (key_success) {
    auto& slot_value = current_slot->second;
    slot_value.store(insert_pair.second, memory_order_relaxed);
    return insert_result::SUCCESS;
  }

  return insert_result::CONTINUE;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool uses_vector_load, typename CG, typename KeyEqual>
__device__ std::enable_if_t<uses_vector_load, void>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::insert_impl(
  CG g, value_type const& insert_pair, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, insert_pair.first);
  while (true) {
    value_type arr[2];
    load_pair_array(&arr[0], current_slot);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as
    // the sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const first_slot_is_empty =
      (detail::bitwise_compare(arr[0].first, this->get_empty_key_sentinel()));
    auto const second_slot_is_empty =
      (detail::bitwise_compare(arr[1].first, this->get_empty_key_sentinel()));
    auto const window_contains_empty = g.ballot(first_slot_is_empty or second_slot_is_empty);

    if (window_contains_empty) {
      // the first lane in the group with an empty slot will attempt the insert
      insert_result status{insert_result::CONTINUE};
      uint32_t src_lane = __ffs(window_contains_empty) - 1;
      if (g.thread_rank() == src_lane) {
        auto insert_location = first_slot_is_empty ? current_slot : current_slot + 1;
        // One single CAS operation since vector loads are dedicated to packable pairs
        status = packed_cas(insert_location, insert_pair, key_equal);
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
  }  // while true
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool uses_vector_load, typename CG, typename KeyEqual>
__device__ std::enable_if_t<not uses_vector_load, void>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::insert_impl(
  CG g, value_type const& insert_pair, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, insert_pair.first);

  while (true) {
    key_type const existing_key = current_slot->first.load(cuda::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as the
    // sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty =
      detail::bitwise_compare(existing_key, this->get_empty_key_sentinel());
    auto const window_contains_empty = g.ballot(slot_is_empty);

    if (window_contains_empty) {
      // the first lane in the group with an empty slot will attempt the insert
      insert_result status{insert_result::CONTINUE};
      uint32_t src_lane = __ffs(window_contains_empty) - 1;

      if (g.thread_rank() == src_lane) {
#if __CUDA_ARCH__ < 700
        status = cas_dependent_write(current_slot, insert_pair, key_equal);
#else
        status = back_to_back_cas(current_slot, insert_pair, key_equal);
#endif
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
  }  // while true
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename KeyEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::insert(
  CG g, value_type const& insert_pair, KeyEqual key_equal) noexcept
{
  insert_impl<uses_vector_load()>(g, insert_pair, key_equal);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool uses_vector_load, typename CG, typename KeyEqual>
__device__ std::enable_if_t<uses_vector_load, bool>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::contains_impl(
  CG g, Key const& k, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k);

  while (true) {
    value_type arr[2];
    load_pair_array(&arr[0], current_slot);

    auto const first_slot_is_empty =
      detail::bitwise_compare(arr[0].first, this->get_empty_key_sentinel());
    auto const second_slot_is_empty =
      detail::bitwise_compare(arr[1].first, this->get_empty_key_sentinel());
    auto const first_equals  = (not first_slot_is_empty and key_equal(arr[0].first, k));
    auto const second_equals = (not second_slot_is_empty and key_equal(arr[1].first, k));

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
template <bool uses_vector_load, typename CG, typename KeyEqual>
__device__ std::enable_if_t<not uses_vector_load, bool>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::contains_impl(
  CG g, Key const& k, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k);

  while (true) {
    key_type const existing_key = current_slot->first.load(cuda::std::memory_order_relaxed);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as the
    // sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty =
      detail::bitwise_compare(existing_key, this->get_empty_key_sentinel());

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
template <bool uses_vector_load, bool is_outer, typename CG, typename KeyEqual>
__device__ std::enable_if_t<uses_vector_load, void>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::count_impl(
  CG const& g, Key const& k, std::size_t& thread_num_matches, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k);

  [[maybe_unused]] bool found_match = false;

  while (true) {
    value_type arr[2];
    load_pair_array(&arr[0], current_slot);

    auto const first_slot_is_empty =
      detail::bitwise_compare(arr[0].first, this->get_empty_key_sentinel());
    auto const second_slot_is_empty =
      detail::bitwise_compare(arr[1].first, this->get_empty_key_sentinel());
    auto const first_equals  = (not first_slot_is_empty and key_equal(arr[0].first, k));
    auto const second_equals = (not second_slot_is_empty and key_equal(arr[1].first, k));

    if constexpr (is_outer) {
      if (g.any(first_equals or second_equals)) { found_match = true; }
    }

    thread_num_matches += (first_equals + second_equals);

    if (g.any(first_slot_is_empty or second_slot_is_empty)) {
      if constexpr (is_outer) {
        if ((not found_match) && (g.thread_rank() == 0)) { thread_num_matches++; }
      }
      break;
    }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool uses_vector_load, bool is_outer, typename CG, typename KeyEqual>
__device__ std::enable_if_t<not uses_vector_load, void>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::count_impl(
  CG const& g, Key const& k, std::size_t& thread_num_matches, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k);

  [[maybe_unused]] bool found_match = false;

  while (true) {
    pair<Key, Value> slot_contents =
      *reinterpret_cast<cuco::pair_type<Key, Value> const*>(current_slot);
    auto const& current_key = slot_contents.first;

    auto const slot_is_empty = detail::bitwise_compare(current_key, this->get_empty_key_sentinel());
    auto const equals        = not slot_is_empty and key_equal(current_key, k);

    if constexpr (is_outer) {
      if (g.any(equals)) { found_match = true; }
    }

    thread_num_matches += equals;

    if (g.any(slot_is_empty)) {
      if constexpr (is_outer) {
        if ((not found_match) && (g.thread_rank() == 0)) { thread_num_matches++; }
      }
      break;
    }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool uses_vector_load, bool is_outer, typename CG, typename PairEqual>
__device__ std::enable_if_t<uses_vector_load, void>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::pair_count_impl(
  CG const& g,
  value_type const& pair,
  std::size_t& thread_num_matches,
  PairEqual pair_equal) noexcept
{
  auto key          = pair.first;
  auto current_slot = initial_slot(g, key);

  [[maybe_unused]] bool found_match = false;

  while (true) {
    value_type arr[2];
    load_pair_array(&arr[0], current_slot);

    auto const first_slot_is_empty =
      detail::bitwise_compare(arr[0].first, this->get_empty_key_sentinel());
    auto const second_slot_is_empty =
      detail::bitwise_compare(arr[1].first, this->get_empty_key_sentinel());

    auto const first_slot_equals  = (not first_slot_is_empty and pair_equal(arr[0], pair));
    auto const second_slot_equals = (not second_slot_is_empty and pair_equal(arr[1], pair));

    if constexpr (is_outer) {
      if (g.any(first_slot_equals or second_slot_equals)) { found_match = true; }
    }

    thread_num_matches += (first_slot_equals + second_slot_equals);

    if (g.any(first_slot_is_empty or second_slot_is_empty)) {
      if constexpr (is_outer) {
        if ((not found_match) && (g.thread_rank() == 0)) { thread_num_matches++; }
      }
      break;
    }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool uses_vector_load, bool is_outer, typename CG, typename PairEqual>
__device__ std::enable_if_t<not uses_vector_load, void>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::pair_count_impl(
  CG const& g,
  value_type const& pair,
  std::size_t& thread_num_matches,
  PairEqual pair_equal) noexcept
{
  auto key          = pair.first;
  auto current_slot = initial_slot(g, key);

  [[maybe_unused]] bool found_match = false;

  while (true) {
    auto slot_contents = *reinterpret_cast<cuco::pair_type<Key, Value> const*>(current_slot);

    auto const slot_is_empty = (slot_contents.first == this->get_empty_key_sentinel());

    auto const equals = not slot_is_empty and pair_equal(slot_contents, pair);

    if constexpr (is_outer) {
      if (g.any(equals)) { found_match = true; }
    }

    thread_num_matches += equals;

    if (g.any(slot_is_empty)) {
      if constexpr (is_outer) {
        if ((not found_match) && (g.thread_rank() == 0)) { thread_num_matches++; }
      }
      break;
    }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t buffer_size,
          bool is_outer,
          typename warpT,
          typename CG,
          typename atomicT,
          typename OutputIt,
          typename KeyEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::retrieve_impl(
  warpT const& warp,
  CG const& g,
  Key const& k,
  uint32_t* warp_counter,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin,
  KeyEqual key_equal) noexcept
{
  const uint32_t cg_lane_id = g.thread_rank();

  auto current_slot = initial_slot(g, k);

  bool running                      = true;
  [[maybe_unused]] bool found_match = false;

  while (warp.any(running)) {
    if (running) {
      value_type arr[2];
      load_pair_array(&arr[0], current_slot);

      auto const first_slot_is_empty =
        detail::bitwise_compare(arr[0].first, this->get_empty_key_sentinel());
      auto const second_slot_is_empty =
        detail::bitwise_compare(arr[1].first, this->get_empty_key_sentinel());
      auto const first_equals  = (not first_slot_is_empty and key_equal(arr[0].first, k));
      auto const second_equals = (not second_slot_is_empty and key_equal(arr[1].first, k));
      auto const first_exists  = g.ballot(first_equals);
      auto const second_exists = g.ballot(second_equals);

      if (first_exists or second_exists) {
        if constexpr (is_outer) { found_match = true; }

        auto num_first_matches  = __popc(first_exists);
        auto num_second_matches = __popc(second_exists);

        uint32_t output_idx;
        if (0 == cg_lane_id) {
          output_idx = atomicAdd(warp_counter, (num_first_matches + num_second_matches));
        }
        output_idx = g.shfl(output_idx, 0);

        if (first_equals) {
          auto lane_offset = __popc(first_exists & ((1 << cg_lane_id) - 1));
          Key key          = k;
          output_buffer[output_idx + lane_offset] =
            cuco::make_pair<Key, Value>(std::move(key), std::move(arr[0].second));
        }
        if (second_equals) {
          auto lane_offset = __popc(second_exists & ((1 << cg_lane_id) - 1));
          Key key          = k;
          output_buffer[output_idx + num_first_matches + lane_offset] =
            cuco::make_pair<Key, Value>(std::move(key), std::move(arr[1].second));
        }
      }
      if (g.any(first_slot_is_empty or second_slot_is_empty)) {
        running = false;
        if constexpr (is_outer) {
          if ((not found_match) && (cg_lane_id == 0)) {
            auto output_idx           = atomicAdd(warp_counter, 1);
            Key key                   = k;
            output_buffer[output_idx] = cuco::make_pair<Key, Value>(
              std::move(key), std::move(this->get_empty_value_sentinel()));
          }
        }
      }
    }  // if running

    warp.sync();
    if (*warp_counter + warp.size() * vector_width() > buffer_size) {
      flush_output_buffer(warp, *warp_counter, output_buffer, num_matches, output_begin);
      // First lane reset warp-level counter
      if (warp.thread_rank() == 0) { *warp_counter = 0; }
    }

    current_slot = next_slot(current_slot);
  }  // while running
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t cg_size,
          uint32_t buffer_size,
          bool is_outer,
          typename CG,
          typename atomicT,
          typename OutputIt,
          typename KeyEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::retrieve_impl(
  CG const& g,
  Key const& k,
  uint32_t* cg_counter,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin,
  KeyEqual key_equal) noexcept
{
  const uint32_t lane_id = g.thread_rank();

  auto current_slot = initial_slot(g, k);

  bool running                      = true;
  [[maybe_unused]] bool found_match = false;

  while (running) {
    // TODO: Replace reinterpret_cast with atomic ref when possible. The current implementation
    // is unsafe!
    static_assert(sizeof(Key) == sizeof(cuda::atomic<Key>));
    static_assert(sizeof(Value) == sizeof(cuda::atomic<Value>));
    pair<Key, Value> slot_contents = *reinterpret_cast<pair<Key, Value> const*>(current_slot);

    auto const slot_is_empty =
      detail::bitwise_compare(slot_contents.first, this->get_empty_key_sentinel());
    auto const equals = (not slot_is_empty and key_equal(slot_contents.first, k));
    auto const exists = g.ballot(equals);

    if (exists) {
      if constexpr (is_outer) { found_match = true; }
      auto num_matches    = __popc(exists);
      uint32_t output_idx = *cg_counter;
      if (equals) {
        // Each match computes its lane-level offset
        auto lane_offset = __popc(exists & ((1 << lane_id) - 1));
        Key key          = k;
        output_buffer[output_idx + lane_offset] =
          cuco::make_pair<Key, Value>(std::move(key), std::move(slot_contents.second));
      }
      if (0 == lane_id) { (*cg_counter) += num_matches; }
    }
    if (g.any(slot_is_empty)) {
      running = false;
      if constexpr (is_outer) {
        if ((not found_match) && (lane_id == 0)) {
          auto output_idx           = (*cg_counter)++;
          Key key                   = k;
          output_buffer[output_idx] = cuco::make_pair<Key, Value>(
            std::move(key), std::move(this->get_empty_value_sentinel()));
        }
      }
    }

    g.sync();

    // Flush if the next iteration won't fit into buffer
    if ((*cg_counter + cg_size) > buffer_size) {
      flush_output_buffer(g, *cg_counter, output_buffer, num_matches, output_begin);
      // First lane reset CG-level counter
      if (lane_id == 0) { *cg_counter = 0; }
    }
    current_slot = next_slot(current_slot);
  }  // while running
}

// public APIs

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename atomicT, typename OutputIt>
__inline__ __device__
  std::enable_if_t<thrust::is_contiguous_iterator<OutputIt>::value and SUPPORTS_CG_MEMCPY_ASYNC,
                   void>
  static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::flush_output_buffer(
    CG const& g,
    uint32_t const num_outputs,
    value_type* output_buffer,
    atomicT* num_matches,
    OutputIt output_begin) noexcept
{
  std::size_t offset;
  const auto lane_id = g.thread_rank();
  if (0 == lane_id) {
    offset = num_matches->fetch_add(num_outputs, cuda::std::memory_order_relaxed);
  }
  offset = g.shfl(offset, 0);

#if defined(CUCO_HAS_CUDA_BARRIER)
  cooperative_groups::memcpy_async(g,
                                   output_begin + offset,
                                   output_buffer,
                                   cuda::aligned_size_t<alignof(cuco::pair_type<Key, Value>)>(
                                     sizeof(cuco::pair_type<Key, Value>) * num_outputs));
#else
  cooperative_groups::memcpy_async(
    g, output_begin + offset, output_buffer, sizeof(cuco::pair_type<Key, Value>) * num_outputs);
#endif
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename atomicT, typename OutputIt>
__inline__ __device__ std::enable_if_t<not(thrust::is_contiguous_iterator<OutputIt>::value and
                                           SUPPORTS_CG_MEMCPY_ASYNC),
                                       void>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::flush_output_buffer(
  CG const& g,
  uint32_t const num_outputs,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin) noexcept
{
  std::size_t offset;
  const auto lane_id = g.thread_rank();
  if (0 == lane_id) {
    offset = num_matches->fetch_add(num_outputs, cuda::std::memory_order_relaxed);
  }
  offset = g.shfl(offset, 0);

  for (auto index = lane_id; index < num_outputs; index += g.size()) {
    *(output_begin + offset + index) = output_buffer[index];
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename KeyEqual>
__device__ bool static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::contains(
  CG g, Key const& k, KeyEqual key_equal) noexcept
{
  return contains_impl<uses_vector_load()>(g, k, key_equal);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename KeyEqual>
__device__ void static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::count(
  CG const& g, Key const& k, std::size_t& thread_num_matches, KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = false;
  count_impl<uses_vector_load(), is_outer>(g, k, thread_num_matches, key_equal);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename KeyEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::count_outer(
  CG const& g, Key const& k, std::size_t& thread_num_matches, KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = true;
  count_impl<uses_vector_load(), is_outer>(g, k, thread_num_matches, key_equal);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename PairEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::pair_count(
  CG const& g,
  value_type const& pair,
  std::size_t& thread_num_matches,
  PairEqual pair_equal) noexcept
{
  constexpr bool is_outer = false;
  pair_count_impl<uses_vector_load(), is_outer>(g, pair, thread_num_matches, pair_equal);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename PairEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::pair_count_outer(
  CG const& g,
  value_type const& pair,
  std::size_t& thread_num_matches,
  PairEqual pair_equal) noexcept
{
  constexpr bool is_outer = true;
  pair_count_impl<uses_vector_load(), is_outer>(g, pair, thread_num_matches, pair_equal);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t buffer_size,
          typename warpT,
          typename CG,
          typename atomicT,
          typename OutputIt,
          typename KeyEqual>
__device__ void static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::retrieve(
  warpT const& warp,
  CG const& g,
  Key const& k,
  uint32_t* warp_counter,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin,
  KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = false;
  retrieve_impl<buffer_size, is_outer>(
    warp, g, k, warp_counter, output_buffer, num_matches, output_begin);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t buffer_size,
          typename warpT,
          typename CG,
          typename atomicT,
          typename OutputIt,
          typename KeyEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::retrieve_outer(
  warpT const& warp,
  CG const& g,
  Key const& k,
  uint32_t* warp_counter,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin,
  KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = true;
  retrieve_impl<buffer_size, is_outer>(
    warp, g, k, warp_counter, output_buffer, num_matches, output_begin);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t cg_size,
          uint32_t buffer_size,
          typename CG,
          typename atomicT,
          typename OutputIt,
          typename KeyEqual>
__device__ void static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::retrieve(
  CG const& g,
  Key const& k,
  uint32_t* cg_counter,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin,
  KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = false;
  retrieve_impl<cg_size, buffer_size, is_outer>(
    g, k, cg_counter, output_buffer, num_matches, output_begin);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t cg_size,
          uint32_t buffer_size,
          typename CG,
          typename atomicT,
          typename OutputIt,
          typename KeyEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::retrieve_outer(
  CG const& g,
  Key const& k,
  uint32_t* cg_counter,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin,
  KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = true;
  retrieve_impl<cg_size, buffer_size, is_outer>(
    g, k, cg_counter, output_buffer, num_matches, output_begin);
}

}  // namespace cuco
