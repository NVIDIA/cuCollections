/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cuco/detail/utils.cuh>
#include <cuco/detail/utils.hpp>

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>

#include <iterator>

namespace cuco {

template <class Key,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbeSequence,
          class Allocator,
          class Storage>
static_set<Key, Scope, KeyEqual, ProbeSequence, Allocator, Storage>::static_set(
  std::size_t capacity,
  sentinel::empty_key<Key> empty_key_sentinel,
  Allocator const& alloc,
  cudaStream_t stream)
  : capacity_{cuco::detail::get_valid_capacity<cg_size(), vector_width(), uses_vector_load()>(
      capacity)},
    empty_key_sentinel_{empty_key_sentinel.value},
    counter_allocator_{alloc},
    slot_allocator_{alloc},
    delete_counter_{counter_allocator_},
    delete_slots_{slot_allocator_, capacity_},
    d_counter_{counter_allocator_.allocate(1), delete_counter_},
    slots_{slot_allocator_.allocate(capacity_), delete_slots_}
{
}

template <class Key,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbeSequence,
          class Allocator,
          class Storage>
template <typename InputIt>
void static_set<Key, Scope, KeyEqual, ProbeSequence, Allocator, Storage>::insert(
  InputIt first, InputIt last, cudaStream_t stream)
{
}

template <class Key,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbeSequence,
          class Allocator,
          class Storage>
template <typename InputIt, typename OutputIt, typename KeyEqual>
void static_set<Key, Scope, KeyEqual, ProbeSequence, Allocator, Storage>::contains(
  InputIt first, InputIt last, OutputIt output_begin, KeyEqual key_equal, cudaStream_t stream) const
{
}

template <class Key,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbeSequence,
          class Allocator,
          class Storage>
std::size_t static_set<Key, Scope, KeyEqual, ProbeSequence, Allocator, Storage>::size(
  cudaStream_t stream) const noexcept
{
  return 0;
}

template <class Key,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbeSequence,
          class Allocator,
          class Storage>
float static_set<Key, Scope, KeyEqual, ProbeSequence, Allocator, Storage>::load_factor(
  cudaStream_t stream) const noexcept
{
  return 0;
}

}  // namespace cuco
