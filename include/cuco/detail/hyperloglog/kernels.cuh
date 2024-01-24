/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#pragma once

#include <cuco/detail/utility/cuda.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cstddef>

#include <cooperative_groups.h>

namespace cuco::hyperloglog_ns::detail {
CUCO_SUPPRESS_KERNEL_WARNINGS

template <class RefType>
CUCO_KERNEL void clear(RefType ref)
{
  auto const block = cooperative_groups::this_thread_block();
  if (block.group_index().x == 0) { ref.clear(block); }
}

template <class InputIt, class RefType>
CUCO_KERNEL void add_shmem(InputIt first, cuco::detail::index_type n, RefType ref)
{
  using local_ref_type = typename RefType::with_scope<cuda::thread_scope_block>;

  __shared__ typename local_ref_type::storage_type local_storage;

  auto const loop_stride = cuco::detail::grid_stride();
  auto idx               = cuco::detail::global_thread_id();
  auto const block       = cooperative_groups::this_thread_block();

  local_ref_type local_ref(local_storage);
  local_ref.clear(block);
  block.sync();

  while (idx < n) {
    local_ref.add(*(first + idx));
    idx += loop_stride;
  }
  block.sync();

  ref.merge(block, local_ref);
}

template <class OtherRefType, class RefType>
CUCO_KERNEL void merge(OtherRefType other_ref, RefType ref)
{
  auto const block = cooperative_groups::this_thread_block();
  if (block.group_index().x == 0) { ref.merge(block, other_ref); }
}

// TODO this kernel currently isn't being used
template <class RefType>
CUCO_KERNEL void estimate(std::size_t* cardinality, RefType ref)
{
  auto const block = cooperative_groups::this_thread_block();
  if (block.group_index().x == 0) {
    auto const estimate = ref.estimate(block);
    if (block.thread_rank() == 0) { *cardinality = estimate; }
  }
}
}  // namespace cuco::hyperloglog_ns::detail