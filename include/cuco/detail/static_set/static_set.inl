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

#include <cuco/detail/common_kernels.cuh>
#include <cuco/detail/defaults.cuh>
#include <cuco/detail/error.hpp>
#include <cuco/detail/static_set/kernels.cuh>
#include <cuco/reference.cuh>

#include <cstddef>

namespace cuco {

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
    counter_{allocator_},
    slot_storage_{cuco::detail::get_valid_capacity<ProbingScheme>(capacity), allocator_}
{
  auto constexpr stride = 4;
  auto const grid_size  = (this->capacity() + stride * detail::CUCO_DEFAULT_BLOCK_SIZE - 1) /
                         (stride * detail::CUCO_DEFAULT_BLOCK_SIZE);

  detail::initialize<<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
    slot_storage_.slots(), empty_key_sentinel_, this->capacity());
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

  counter_.reset(stream);
  size_type h_num_successes{};

  auto ref =
    cuco::static_set_ref(empty_key_sentinel_, predicate_, probing_scheme_, slot_storage_.view());

  detail::insert<detail::CUCO_DEFAULT_BLOCK_SIZE>
    <<<grid_size, detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
      first, first + num_keys, counter_.get(), ref);

  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_num_successes, counter_.get(), sizeof(size_type), cudaMemcpyDeviceToHost, stream));

  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));  // stream sync to ensure h_num_successes is updated

  size_ += h_num_successes;
}

}  // namespace cuco
