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

#include <cuco/detail/bitwise_compare.cuh>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/detail/utils.hpp>

#include <cub/device/device_for.cuh>
#include <cuda/functional>
#include <cuda/stream_ref>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

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

  std::size_t submap_idx = 0;
  while (num_to_insert > 0) {
    auto& cur = submaps_[submap_idx];

    auto capacity_remaining = max_load_factor_ * cur->capacity() - cur->size();
    if (capacity_remaining >= min_insert_size_) {
      auto const n = std::min(static_cast<detail::index_type>(capacity_remaining), num_to_insert);

      std::size_t h_num_successes = cur->insert(first, first + n, stream);

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

  auto num_keys = cuco::detail::distance(first, last);
  if (num_keys == 0) { return; }

  for (std::size_t submap_idx = 0; submap_idx < submaps_.size(); ++submap_idx) {
    auto& cur              = submaps_[submap_idx];
    auto const size_before = cur->size(stream);
    cur->erase(first, last, stream);
    auto const size_after = cur->size(stream);
    size_ -= (size_before - size_after);
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
void dynamic_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::erase_async(
  InputIt first, InputIt last, cuda::stream_ref stream)
{
  auto const& ref = *submaps_.front();
  CUCO_EXPECTS(ref.empty_key_sentinel() != ref.erased_key_sentinel(),
               "Erase requires a unique erased key sentinel to be provided at construction.",
               std::runtime_error);

  auto num_keys = cuco::detail::distance(first, last);
  if (num_keys == 0) { return; }

  for (std::size_t submap_idx = 0; submap_idx < submaps_.size(); ++submap_idx) {
    submaps_[submap_idx]->erase_async(first, last, stream);
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

  auto const empty_val = submaps_.front()->empty_value_sentinel();

  CUCO_CUDA_TRY(cub::DeviceFor::ForEachN(
    output_begin,
    num_keys,
    [empty_val] __device__(mapped_type & val) { val = empty_val; },
    stream.get()));

  if (submaps_.size() == 1) {
    submaps_.front()->find_async(first, last, output_begin, stream);
    return;
  }

  using temp_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<mapped_type>;
  auto temp_allocator = temp_allocator_type{alloc_};
  auto* temp          = temp_allocator.allocate(num_keys, stream);

  auto* output_ptr = thrust::raw_pointer_cast(&*output_begin);

  for (std::size_t submap_idx = 0; submap_idx < submaps_.size(); ++submap_idx) {
    submaps_[submap_idx]->find_async(first, last, temp, stream);

    CUCO_CUDA_TRY(cub::DeviceFor::ForEachN(
      thrust::make_counting_iterator<std::size_t>(0),
      num_keys,
      [output_ptr, temp, empty_val] __device__(std::size_t i) {
        if (cuco::detail::bitwise_compare(output_ptr[i], empty_val)) { output_ptr[i] = temp[i]; }
      },
      stream.get()));
  }

  temp_allocator.deallocate(temp, num_keys, stream);
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

  CUCO_CUDA_TRY(cub::DeviceFor::ForEachN(
    output_begin, num_keys, [] __device__(bool& val) { val = false; }, stream.get()));

  if (submaps_.size() == 1) {
    submaps_.front()->contains_async(first, last, output_begin, stream);
    return;
  }

  using temp_allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<bool>;
  auto temp_allocator = temp_allocator_type{alloc_};
  auto* temp          = temp_allocator.allocate(num_keys, stream);

  auto* output_ptr = thrust::raw_pointer_cast(&*output_begin);

  for (std::size_t submap_idx = 0; submap_idx < submaps_.size(); ++submap_idx) {
    submaps_[submap_idx]->contains_async(first, last, temp, stream);

    CUCO_CUDA_TRY(cub::DeviceFor::ForEachN(
      thrust::make_counting_iterator<std::size_t>(0),
      num_keys,
      [output_ptr, temp] __device__(std::size_t i) { output_ptr[i] = output_ptr[i] || temp[i]; },
      stream.get()));
  }

  temp_allocator.deallocate(temp, num_keys, stream);
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
  KeyOut keys_current   = keys_out;
  ValueOut vals_current = values_out;

  for (std::size_t submap_idx = 0; submap_idx < submaps_.size(); ++submap_idx) {
    auto [keys_end, vals_end] =
      submaps_[submap_idx]->retrieve_all(keys_current, vals_current, stream);
    keys_current = keys_end;
    vals_current = vals_end;
  }

  return {keys_current, vals_current};
}

}  // namespace cuco
