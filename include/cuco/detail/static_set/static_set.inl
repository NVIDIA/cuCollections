/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuco/detail/error.hpp>
#include <cuco/detail/prime.hpp>
#include <cuco/detail/static_set/kernels.cuh>
#include <cuco/detail/tuning.cuh>  // TODO .hpp?
#include <cuco/function.hpp>
#include <cuco/static_set_ref.cuh>

#include <cstddef>

namespace cuco {
namespace experimental {

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::static_set(
  Extent capacity,
  sentinel::empty_key<Key> empty_key_sentinel,
  KeyEqual pred,
  ProbingScheme probing_scheme,
  Allocator const& alloc,
  cudaStream_t stream)
  : size_{0},
    empty_key_sentinel_{empty_key_sentinel.value},
    predicate_{pred},
    probing_scheme_{probing_scheme},
    allocator_{alloc},
    window_storage_{cuco::detail::get_num_windows<cg_size, window_size, size_type>(capacity),
                    allocator_}
{
  window_storage_.initialize(empty_key_sentinel_, stream);
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt>
void static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::insert(
  InputIt first, InputIt last, cudaStream_t stream)
{
  auto num_keys = std::distance(first, last);
  if (num_keys == 0) { return; }

  auto const grid_size =
    (cg_size * num_keys + detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
    (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE);

  detail::insert<cg_size, detail::CUCO_DEFAULT_BLOCK_SIZE>
    <<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
      first, first + num_keys, reference_with_functions<function::insert>());
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename OutputIt>
void static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::contains(
  InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream) const
{
  auto num_keys = std::distance(first, last);
  if (num_keys == 0) { return; }

  auto const grid_size =
    (cg_size * num_keys + detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
    (detail::CUCO_DEFAULT_STRIDE * detail::CUCO_DEFAULT_BLOCK_SIZE);

  detail::contains<cg_size, detail::CUCO_DEFAULT_BLOCK_SIZE>
    <<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
      first, first + num_keys, output_begin, reference_with_functions<function::contains>());
}

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename... Functions>
auto static_set<Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::reference()
  const noexcept
{
  return cuco::experimental::static_set_ref<key_type,
                                            Scope,
                                            key_equal,
                                            probing_scheme_type,
                                            window_reference_type,
                                            Functions...>{
    cuco::sentinel::empty_key<Key>(empty_key_sentinel_),
    predicate_,
    probing_scheme_,
    window_storage_.reference()};
}
}  // namespace experimental
}  // namespace cuco
