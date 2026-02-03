/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include <cuco/detail/dynamic_map/kernels.cuh>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/detail/utils.hpp>
#include <cuco/operator.hpp>

#include <cuda/std/atomic>
#include <cuda/stream_ref>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cuco {

template <typename Key,
          typename T,
          typename Extent,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename Allocator,
          typename Storage>
constexpr dynamic_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  dynamic_map(Extent initial_capacity,
              empty_key<Key> empty_key_sentinel,
              empty_value<T> empty_value_sentinel,
              KeyEqual const& pred,
              ProbingScheme const& probing_scheme,
              cuda_thread_scope<Scope> scope,
              Storage storage,
              Allocator const& alloc,
              cuda::stream_ref stream)
  : size_{0},
    capacity_{initial_capacity},
    min_insert_size_{static_cast<size_type>(1E4)},
    max_load_factor_{0.60f},
    alloc_{alloc}
{
  submaps_.push_back(std::make_unique<map_type>(initial_capacity,
                                                empty_key_sentinel,
                                                empty_value_sentinel,
                                                pred,
                                                probing_scheme,
                                                scope,
                                                storage,
                                                alloc,
                                                stream));
}

template <typename Key,
          typename T,
          typename Extent,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename Allocator,
          typename Storage>
constexpr dynamic_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  dynamic_map(Extent initial_capacity,
              empty_key<Key> empty_key_sentinel,
              empty_value<T> empty_value_sentinel,
              erased_key<Key> erased_key_sentinel,
              KeyEqual const& pred,
              ProbingScheme const& probing_scheme,
              cuda_thread_scope<Scope> scope,
              Storage storage,
              Allocator const& alloc,
              cuda::stream_ref stream)
  : size_{0},
    capacity_{initial_capacity},
    min_insert_size_{static_cast<size_type>(1E4)},
    max_load_factor_{0.60f},
    alloc_{alloc}
{
  CUCO_EXPECTS(empty_key_sentinel.value != erased_key_sentinel.value,
               "The empty key sentinel and erased key sentinel cannot be the same value.",
               std::runtime_error);

  submaps_.push_back(std::make_unique<map_type>(initial_capacity,
                                                empty_key_sentinel,
                                                empty_value_sentinel,
                                                erased_key_sentinel,
                                                pred,
                                                probing_scheme,
                                                scope,
                                                storage,
                                                alloc,
                                                stream));
}

template <typename Key,
          typename T,
          typename Extent,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename Allocator,
          typename Storage>
template <typename InputIt>
void dynamic_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::insert(
  InputIt first, InputIt last, cuda::stream_ref stream)
{
  auto num_to_insert = cuco::detail::distance(first, last);
  this->reserve(size_ + num_to_insert, stream);

  // Fast path: single submap, no cross-submap duplicate check needed
  if (submaps_.size() == 1) {
    size_ += submaps_.front()->insert(first, last, stream);
    return;
  }

  // Multiple submaps: use kernel to check for duplicates across all submaps
  using ref_type = decltype(submaps_.front()->ref(cuco::op::contains, cuco::op::insert));

  using ref_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<ref_type>;
  auto ref_allocator = ref_allocator_type{alloc_};

  using counter_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<cuda::atomic<std::size_t>>;
  auto counter_allocator = counter_allocator_type{alloc_};

  std::size_t submap_idx = 0;
  while (num_to_insert > 0) {
    auto& cur = submaps_[submap_idx];

    auto capacity_remaining = max_load_factor_ * cur->capacity() - cur->size();
    if (capacity_remaining >= min_insert_size_) {
      auto const n = std::min(static_cast<detail::index_type>(capacity_remaining), num_to_insert);

      // Allocate and initialize device counter
      auto* d_num_successes = counter_allocator.allocate(1, stream);
      CUCO_CUDA_TRY(
        cudaMemsetAsync(d_num_successes, 0, sizeof(cuda::atomic<std::size_t>), stream.get()));

      // Allocate and copy refs for all submaps (with both contains and insert ops)
      auto* d_submap_refs = ref_allocator.allocate(submaps_.size(), stream);
      std::vector<ref_type> h_submap_refs;
      h_submap_refs.reserve(submaps_.size());
      for (auto const& submap : submaps_) {
        h_submap_refs.push_back(submap->ref(cuco::op::contains, cuco::op::insert));
      }
      CUCO_CUDA_TRY(cudaMemcpyAsync(d_submap_refs,
                                    h_submap_refs.data(),
                                    sizeof(ref_type) * submaps_.size(),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));

      auto constexpr cg_size    = ProbingScheme::cg_size;
      auto constexpr block_size = cuco::detail::default_block_size();
      auto const grid_size      = cuco::detail::grid_size(n, cg_size);

      detail::dynamic_map_ns::insert<cg_size, block_size>
        <<<grid_size, block_size, 0, stream.get()>>>(first,
                                                     n,
                                                     d_num_successes,
                                                     d_submap_refs,
                                                     static_cast<uint32_t>(submap_idx),
                                                     static_cast<uint32_t>(submaps_.size()));

      // Read back success count
      std::size_t h_num_successes = 0;
      CUCO_CUDA_TRY(cudaMemcpyAsync(&h_num_successes,
                                    d_num_successes,
                                    sizeof(std::size_t),
                                    cudaMemcpyDeviceToHost,
                                    stream.get()));
      CUCO_CUDA_TRY(cudaStreamSynchronize(stream.get()));

      ref_allocator.deallocate(d_submap_refs, submaps_.size(), stream);
      counter_allocator.deallocate(d_num_successes, 1, stream);

      size_ += h_num_successes;
      first += n;
      num_to_insert -= n;
    }
    submap_idx++;
  }
}

template <typename Key,
          typename T,
          typename Extent,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename Allocator,
          typename Storage>
template <typename InputIt>
void dynamic_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  insert_or_assign(InputIt first, InputIt last, cuda::stream_ref stream)
{
  auto num_to_insert = cuco::detail::distance(first, last);
  this->reserve(size_ + num_to_insert, stream);

  // Fast path: single submap
  if (submaps_.size() == 1) {
    auto const old_size = submaps_.front()->size(stream);
    submaps_.front()->insert_or_assign(first, last, stream);
    auto const new_size = submaps_.front()->size(stream);
    size_ += (new_size - old_size);
    return;
  }

  // Multiple submaps: use kernel to check for existing keys across all submaps
  using ref_type = decltype(submaps_.front()->ref(
    cuco::op::contains, cuco::op::insert, cuco::op::insert_or_assign));

  using ref_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<ref_type>;
  auto ref_allocator = ref_allocator_type{alloc_};

  using counter_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<cuda::atomic<std::size_t>>;
  auto counter_allocator = counter_allocator_type{alloc_};

  std::size_t submap_idx = 0;
  while (num_to_insert > 0) {
    auto& cur = submaps_[submap_idx];

    auto capacity_remaining = max_load_factor_ * cur->capacity() - cur->size();
    if (capacity_remaining >= min_insert_size_) {
      auto const n = std::min(static_cast<detail::index_type>(capacity_remaining), num_to_insert);

      // Allocate and initialize device counter for new insertions
      auto* d_num_insertions = counter_allocator.allocate(1, stream);
      CUCO_CUDA_TRY(
        cudaMemsetAsync(d_num_insertions, 0, sizeof(cuda::atomic<std::size_t>), stream.get()));

      // Allocate and copy refs for all submaps
      auto* d_submap_refs = ref_allocator.allocate(submaps_.size(), stream);
      std::vector<ref_type> h_submap_refs;
      h_submap_refs.reserve(submaps_.size());
      for (auto const& submap : submaps_) {
        h_submap_refs.push_back(
          submap->ref(cuco::op::contains, cuco::op::insert, cuco::op::insert_or_assign));
      }
      CUCO_CUDA_TRY(cudaMemcpyAsync(d_submap_refs,
                                    h_submap_refs.data(),
                                    sizeof(ref_type) * submaps_.size(),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));

      auto constexpr cg_size    = ProbingScheme::cg_size;
      auto constexpr block_size = cuco::detail::default_block_size();
      auto const grid_size      = cuco::detail::grid_size(n, cg_size);

      detail::dynamic_map_ns::insert_or_assign<cg_size, block_size>
        <<<grid_size, block_size, 0, stream.get()>>>(first,
                                                     n,
                                                     d_num_insertions,
                                                     d_submap_refs,
                                                     static_cast<uint32_t>(submap_idx),
                                                     static_cast<uint32_t>(submaps_.size()));

      // Read back insertion count (only new insertions, not assignments)
      std::size_t h_num_insertions = 0;
      CUCO_CUDA_TRY(cudaMemcpyAsync(&h_num_insertions,
                                    d_num_insertions,
                                    sizeof(std::size_t),
                                    cudaMemcpyDeviceToHost,
                                    stream.get()));
      CUCO_CUDA_TRY(cudaStreamSynchronize(stream.get()));

      ref_allocator.deallocate(d_submap_refs, submaps_.size(), stream);
      counter_allocator.deallocate(d_num_insertions, 1, stream);

      size_ += h_num_insertions;
      first += n;
      num_to_insert -= n;
    }
    submap_idx++;
  }
}

template <typename Key,
          typename T,
          typename Extent,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename Allocator,
          typename Storage>
void dynamic_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::reserve(
  size_type n, cuda::stream_ref stream)
{
  auto const& ref            = *submaps_.front();
  auto const empty_key_val   = ref.empty_key_sentinel();
  auto const empty_value_val = ref.empty_value_sentinel();
  auto const erased_key_val  = ref.erased_key_sentinel();
  auto const pred            = ref.key_eq();
  auto const probing_scheme  = ProbingScheme{ref.hash_function()};
  auto const has_erased_key  = empty_key_val != erased_key_val;

  std::size_t submap_idx = 0;
  while (n > 0) {
    std::size_t submap_capacity;

    if (submap_idx < submaps_.size()) {
      submap_capacity = submaps_[submap_idx]->capacity();
    } else {
      submap_capacity = capacity_;

      if (has_erased_key) {
        submaps_.push_back(std::make_unique<map_type>(submap_capacity,
                                                      empty_key<Key>{empty_key_val},
                                                      empty_value<T>{empty_value_val},
                                                      erased_key<Key>{erased_key_val},
                                                      pred,
                                                      probing_scheme,
                                                      cuda_thread_scope<Scope>{},
                                                      Storage{},
                                                      alloc_,
                                                      stream));
      } else {
        submaps_.push_back(std::make_unique<map_type>(submap_capacity,
                                                      empty_key<Key>{empty_key_val},
                                                      empty_value<T>{empty_value_val},
                                                      pred,
                                                      probing_scheme,
                                                      cuda_thread_scope<Scope>{},
                                                      Storage{},
                                                      alloc_,
                                                      stream));
      }
      capacity_ *= 2;
    }

    auto const usable_capacity =
      static_cast<size_type>(max_load_factor_ * submap_capacity) - min_insert_size_;
    if (usable_capacity >= n) { break; }
    n -= usable_capacity;
    submap_idx++;
  }
}

template <typename Key,
          typename T,
          typename Extent,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename Allocator,
          typename Storage>
template <typename InputIt>
void dynamic_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::erase(
  InputIt first, InputIt last, cuda::stream_ref stream)
{
  auto const& ref = *submaps_.front();
  CUCO_EXPECTS(ref.empty_key_sentinel() != ref.erased_key_sentinel(),
               "Erase requires a unique erased key sentinel to be provided at construction.",
               std::runtime_error);

  auto const num_keys = cuco::detail::distance(first, last);
  if (num_keys == 0) { return; }

  // Fast path: single submap
  if (submaps_.size() == 1) {
    auto const size_before = submaps_.front()->size(stream);
    submaps_.front()->erase(first, last, stream);
    auto const size_after = submaps_.front()->size(stream);
    size_ -= (size_before - size_after);
    return;
  }

  // Multiple submaps: use kernel to erase from all submaps in parallel
  using erase_ref_type = decltype(submaps_.front()->ref(cuco::op::erase));

  using ref_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<erase_ref_type>;
  auto ref_allocator = ref_allocator_type{alloc_};

  using counter_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<cuda::atomic<std::size_t>>;
  auto counter_allocator = counter_allocator_type{alloc_};

  // Allocate and initialize device counter
  auto* d_num_successes = counter_allocator.allocate(1, stream);
  CUCO_CUDA_TRY(
    cudaMemsetAsync(d_num_successes, 0, sizeof(cuda::atomic<std::size_t>), stream.get()));

  // Allocate and copy erase refs for all submaps
  auto* d_refs = ref_allocator.allocate(submaps_.size(), stream);
  std::vector<erase_ref_type> h_refs;
  h_refs.reserve(submaps_.size());
  for (auto const& submap : submaps_) {
    h_refs.push_back(submap->ref(cuco::op::erase));
  }
  CUCO_CUDA_TRY(cudaMemcpyAsync(d_refs,
                                h_refs.data(),
                                sizeof(erase_ref_type) * submaps_.size(),
                                cudaMemcpyHostToDevice,
                                stream.get()));

  auto constexpr cg_size    = ProbingScheme::cg_size;
  auto constexpr block_size = cuco::detail::default_block_size();
  auto const grid_size      = cuco::detail::grid_size(num_keys, cg_size);

  detail::dynamic_map_ns::erase<cg_size, block_size><<<grid_size, block_size, 0, stream.get()>>>(
    first, num_keys, d_num_successes, d_refs, static_cast<uint32_t>(submaps_.size()));

  // Read back success count
  std::size_t h_num_successes = 0;
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_num_successes, d_num_successes, sizeof(std::size_t), cudaMemcpyDeviceToHost, stream.get()));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream.get()));

  ref_allocator.deallocate(d_refs, submaps_.size(), stream);
  counter_allocator.deallocate(d_num_successes, 1, stream);

  size_ -= h_num_successes;
}

template <typename Key,
          typename T,
          typename Extent,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename Allocator,
          typename Storage>
template <typename InputIt>
void dynamic_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::erase_async(
  InputIt first, InputIt last, cuda::stream_ref stream)
{
  auto const& ref = *submaps_.front();
  CUCO_EXPECTS(ref.empty_key_sentinel() != ref.erased_key_sentinel(),
               "Erase requires a unique erased key sentinel to be provided at construction.",
               std::runtime_error);

  auto const num_keys = cuco::detail::distance(first, last);
  if (num_keys == 0) { return; }

  // Fast path: single submap
  if (submaps_.size() == 1) {
    submaps_.front()->erase_async(first, last, stream);
    return;
  }

  // Multiple submaps: use kernel to erase from all submaps in parallel
  using erase_ref_type = decltype(submaps_.front()->ref(cuco::op::erase));

  using ref_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<erase_ref_type>;
  auto ref_allocator = ref_allocator_type{alloc_};

  // Allocate and copy erase refs for all submaps
  auto* d_refs = ref_allocator.allocate(submaps_.size(), stream);
  std::vector<erase_ref_type> h_refs;
  h_refs.reserve(submaps_.size());
  for (auto const& submap : submaps_) {
    h_refs.push_back(submap->ref(cuco::op::erase));
  }
  CUCO_CUDA_TRY(cudaMemcpyAsync(d_refs,
                                h_refs.data(),
                                sizeof(erase_ref_type) * submaps_.size(),
                                cudaMemcpyHostToDevice,
                                stream.get()));

  auto constexpr cg_size    = ProbingScheme::cg_size;
  auto constexpr block_size = cuco::detail::default_block_size();
  auto const grid_size      = cuco::detail::grid_size(num_keys, cg_size);

  // For async, we don't track success count
  using counter_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<cuda::atomic<std::size_t>>;
  auto counter_allocator = counter_allocator_type{alloc_};
  auto* d_num_successes  = counter_allocator.allocate(1, stream);
  CUCO_CUDA_TRY(
    cudaMemsetAsync(d_num_successes, 0, sizeof(cuda::atomic<std::size_t>), stream.get()));

  detail::dynamic_map_ns::erase<cg_size, block_size><<<grid_size, block_size, 0, stream.get()>>>(
    first, num_keys, d_num_successes, d_refs, static_cast<uint32_t>(submaps_.size()));

  // Deallocate asynchronously (counter value is discarded for async)
  ref_allocator.deallocate(d_refs, submaps_.size(), stream);
  counter_allocator.deallocate(d_num_successes, 1, stream);
}

template <typename Key,
          typename T,
          typename Extent,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename Allocator,
          typename Storage>
template <typename InputIt, typename OutputIt>
void dynamic_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::find(
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const
{
  find_async(first, last, output_begin, stream);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream.get()));
}

template <typename Key,
          typename T,
          typename Extent,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename Allocator,
          typename Storage>
template <typename InputIt, typename OutputIt>
void dynamic_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::find_async(
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const
{
  auto const num_keys = cuco::detail::distance(first, last);
  if (num_keys == 0) { return; }

  if (submaps_.size() == 1) {
    submaps_.front()->find_async(first, last, output_begin, stream);
    return;
  }

  using ref_type = decltype(submaps_.front()->ref(cuco::op::find));

  using ref_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<ref_type>;
  auto ref_allocator = ref_allocator_type{alloc_};
  auto* d_refs       = ref_allocator.allocate(submaps_.size(), stream);

  std::vector<ref_type> h_refs;
  h_refs.reserve(submaps_.size());
  for (auto const& submap : submaps_) {
    h_refs.push_back(submap->ref(cuco::op::find));
  }
  CUCO_CUDA_TRY(cudaMemcpyAsync(d_refs,
                                h_refs.data(),
                                sizeof(ref_type) * submaps_.size(),
                                cudaMemcpyHostToDevice,
                                stream.get()));

  auto constexpr cg_size    = ProbingScheme::cg_size;
  auto constexpr block_size = cuco::detail::default_block_size();
  auto const grid_size      = cuco::detail::grid_size(num_keys, cg_size);

  detail::dynamic_map_ns::find<cg_size, block_size><<<grid_size, block_size, 0, stream.get()>>>(
    first, num_keys, output_begin, d_refs, static_cast<uint32_t>(submaps_.size()));

  ref_allocator.deallocate(d_refs, submaps_.size(), stream);
}

template <typename Key,
          typename T,
          typename Extent,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename Allocator,
          typename Storage>
template <typename InputIt, typename OutputIt>
void dynamic_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::contains(
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const
{
  contains_async(first, last, output_begin, stream);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream.get()));
}

template <typename Key,
          typename T,
          typename Extent,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename Allocator,
          typename Storage>
template <typename InputIt, typename OutputIt>
void dynamic_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  contains_async(InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const
{
  auto const num_keys = cuco::detail::distance(first, last);
  if (num_keys == 0) { return; }

  if (submaps_.size() == 1) {
    submaps_.front()->contains_async(first, last, output_begin, stream);
    return;
  }

  using ref_type = decltype(submaps_.front()->ref(cuco::op::contains));

  using ref_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<ref_type>;
  auto ref_allocator = ref_allocator_type{alloc_};
  auto* d_refs       = ref_allocator.allocate(submaps_.size(), stream);

  std::vector<ref_type> h_refs;
  h_refs.reserve(submaps_.size());
  for (auto const& submap : submaps_) {
    h_refs.push_back(submap->ref(cuco::op::contains));
  }
  CUCO_CUDA_TRY(cudaMemcpyAsync(d_refs,
                                h_refs.data(),
                                sizeof(ref_type) * submaps_.size(),
                                cudaMemcpyHostToDevice,
                                stream.get()));

  auto constexpr cg_size    = ProbingScheme::cg_size;
  auto constexpr block_size = cuco::detail::default_block_size();
  auto const grid_size      = cuco::detail::grid_size(num_keys, cg_size);

  detail::dynamic_map_ns::contains<cg_size, block_size><<<grid_size, block_size, 0, stream.get()>>>(
    first, num_keys, output_begin, d_refs, static_cast<uint32_t>(submaps_.size()));

  ref_allocator.deallocate(d_refs, submaps_.size(), stream);
}

template <typename Key,
          typename T,
          typename Extent,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename Allocator,
          typename Storage>
template <typename KeyOut, typename ValueOut>
std::pair<KeyOut, ValueOut>
dynamic_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::retrieve_all(
  KeyOut keys_out, ValueOut values_out, cuda::stream_ref stream) const
{
  if (size_ == 0) { return {keys_out, values_out}; }

  // Fast path: single submap
  if (submaps_.size() == 1) { return submaps_.front()->retrieve_all(keys_out, values_out, stream); }

  // Multiple submaps: use kernel
  using slot_type = typename map_type::value_type;

  // Compute capacity prefix sums and total capacity
  std::vector<detail::index_type> h_capacity_prefix_sum(submaps_.size());
  detail::index_type total_capacity = 0;
  for (std::size_t i = 0; i < submaps_.size(); ++i) {
    total_capacity += submaps_[i]->capacity();
    h_capacity_prefix_sum[i] = total_capacity;
  }

  // Collect slot pointers
  std::vector<slot_type const*> h_slot_arrays(submaps_.size());
  for (std::size_t i = 0; i < submaps_.size(); ++i) {
    h_slot_arrays[i] = submaps_[i]->data();
  }

  // Allocate device memory
  using slot_ptr_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<slot_type const*>;
  auto slot_ptr_allocator = slot_ptr_allocator_type{alloc_};
  auto* d_slot_arrays     = slot_ptr_allocator.allocate(submaps_.size(), stream);

  using index_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<detail::index_type>;
  auto index_allocator        = index_allocator_type{alloc_};
  auto* d_capacity_prefix_sum = index_allocator.allocate(submaps_.size(), stream);

  using counter_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<cuda::atomic<std::size_t>>;
  auto counter_allocator = counter_allocator_type{alloc_};
  auto* d_num_out        = counter_allocator.allocate(1, stream);

  // Copy data to device
  CUCO_CUDA_TRY(cudaMemcpyAsync(d_slot_arrays,
                                h_slot_arrays.data(),
                                sizeof(slot_type const*) * submaps_.size(),
                                cudaMemcpyHostToDevice,
                                stream.get()));
  CUCO_CUDA_TRY(cudaMemcpyAsync(d_capacity_prefix_sum,
                                h_capacity_prefix_sum.data(),
                                sizeof(detail::index_type) * submaps_.size(),
                                cudaMemcpyHostToDevice,
                                stream.get()));
  CUCO_CUDA_TRY(cudaMemsetAsync(d_num_out, 0, sizeof(cuda::atomic<std::size_t>), stream.get()));

  auto constexpr block_size = detail::default_block_size();
  auto const grid_size      = detail::grid_size(total_capacity);

  detail::dynamic_map_ns::retrieve_all<block_size, key_type, mapped_type>
    <<<grid_size, block_size, 0, stream.get()>>>(keys_out,
                                                 values_out,
                                                 d_slot_arrays,
                                                 static_cast<uint32_t>(submaps_.size()),
                                                 total_capacity,
                                                 d_num_out,
                                                 d_capacity_prefix_sum,
                                                 empty_key_sentinel(),
                                                 erased_key_sentinel());

  // Read back count
  std::size_t h_num_out = 0;
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_num_out, d_num_out, sizeof(std::size_t), cudaMemcpyDeviceToHost, stream.get()));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream.get()));

  // Deallocate
  slot_ptr_allocator.deallocate(d_slot_arrays, submaps_.size(), stream);
  index_allocator.deallocate(d_capacity_prefix_sum, submaps_.size(), stream);
  counter_allocator.deallocate(d_num_out, 1, stream);

  return {keys_out + h_num_out, values_out + h_num_out};
}

}  // namespace cuco
