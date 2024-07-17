/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cuco/detail/static_map/kernels.cuh>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/detail/utils.hpp>
#include <cuco/operator.hpp>
#include <cuco/static_map_ref.cuh>

#include <cuda/stream_ref>

#include <cstddef>
#include <iostream>

namespace cuco {
namespace experimental {

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
              Storage,
              Allocator const& alloc,
              cuda::stream_ref stream)
  : empty_key_sentinel_(empty_key_sentinel.value),
    empty_value_sentinel_(empty_value_sentinel.value),
    erased_key_sentinel_(empty_key_sentinel.value),
    size_(0),
    capacity_(initial_capacity),
    min_insert_size_(1E4),
    max_load_factor_(0.60),
    alloc_{alloc}
{
  submaps_.push_back(
    std::make_unique<
      cuco::static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>>(
      initial_capacity,
      empty_key<Key>{empty_key_sentinel},
      empty_value<T>{empty_value_sentinel},
      pred,
      probing_scheme,
      scope,
      Storage{},
      alloc,
      stream));

  submap_mutable_views_.push_back(submaps_[0]->ref(op::insert));
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
              Storage,
              Allocator const& alloc,
              cuda::stream_ref stream)
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

  submaps_.push_back(
    std::make_unique<
      cuco::static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>>(
      initial_capacity,
      empty_key<Key>{empty_key_sentinel_},
      empty_value<T>{empty_value_sentinel_},
      erased_key<Key>{erased_key_sentinel_},
      pred,
      probing_scheme,
      scope,
      alloc,
      stream));
  submap_mutable_views_.push_back(submaps_[0]->ref(op::insert));
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
  std::size_t num_to_insert = std::distance(first, last);

  reserve(size_ + num_to_insert, stream);

  uint32_t submap_idx = 0;
  while (num_to_insert > 0) {
    auto& cur = submaps_[submap_idx];
    //
    std::size_t capacity_remaining = max_load_factor_ * cur->capacity() - cur->size();
    // If we are tying to insert some of the remaining keys into this submap, we can insert
    // only if we meet the minimum insert size.
    if (capacity_remaining >= min_insert_size_) {
      auto const n                = std::min(capacity_remaining, num_to_insert);
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
  std::size_t n, cuda::stream_ref stream)
{
  int64_t num_elements_remaining = n;
  uint32_t submap_idx            = 0;
  while (num_elements_remaining > 0) {
    std::size_t submap_capacity;

    // if the submap already exists
    if (submap_idx < submaps_.size()) {
      submap_capacity = submaps_[submap_idx]->capacity();
    }
    // if the submap does not exist yet, create it
    else {
      submap_capacity = capacity_;
      if (erased_key_sentinel_ != empty_key_sentinel_) {
        submaps_.push_back(
          std::make_unique<
            cuco::static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>>(
            submap_capacity,
            empty_key<Key>{empty_key_sentinel_},
            empty_value<T>{empty_value_sentinel_},
            erased_key<Key>{erased_key_sentinel_},
            KeyEqual{},
            ProbingScheme{},
            cuda_thread_scope<Scope>{},
            Storage{},
            alloc_,
            stream));

      } else {
        submaps_.push_back(
          std::make_unique<
            cuco::static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>>(
            submap_capacity,
            empty_key<Key>{empty_key_sentinel_},
            empty_value<T>{empty_value_sentinel_},
            KeyEqual{},
            ProbingScheme{},
            cuda_thread_scope<Scope>{},
            Storage{},
            alloc_,
            stream));
      }
      submap_mutable_views_.push_back(submaps_[submap_idx]->ref(op::insert));
      capacity_ *= 2;
    }

    num_elements_remaining -= max_load_factor_ * submap_capacity - min_insert_size_;
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
template <typename InputIt, typename OutputIt>
void dynamic_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::contains(
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const
{
  auto num_keys       = std::distance(first, last);
  int increment       = 0;
  uint32_t submap_idx = 0;
  while (num_keys > 0) {
    const auto& cur       = submaps_[submap_idx];
    const size_t cur_size = cur->size(stream);

    CUCO_CUDA_TRY(cudaStreamSynchronize(stream.get()));

    cur->contains(first,
                  first + increment,  // should I do bounds checking?
                  output_begin + increment,
                  stream);

    increment += cur_size;
    num_keys -= cur_size;
    submap_idx++;
    first += cur_size;
  }
}
/*
while (num_to_insert > 0) {
    auto& cur = submaps_[submap_idx];
    std::size_t capacity_remaining = max_load_factor_ * cur->capacity() - cur->size();

    if (capacity_remaining >= min_insert_size_) {
      auto const n                = std::min(capacity_remaining, num_to_insert);
      std::size_t h_num_successes = cur->insert(first, first + n, stream);

      size_ += h_num_successes;
      first += n;
      num_to_insert -= n;
    }
    submap_idx++;
  }
*/

}  // namespace experimental
}  // namespace cuco
