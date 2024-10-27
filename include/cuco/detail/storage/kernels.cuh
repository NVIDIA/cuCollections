/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cstddef>

namespace cuco {
namespace detail {

CUCO_SUPPRESS_KERNEL_WARNINGS

/**
 * @brief Initializes each slot in the bucket storage to contain `value`.
 *
 * @tparam BucketT Bucket type
 *
 * @param buckets Pointer to flat storage for buckets
 * @param n Number of input buckets
 * @param value Value to which all values in `slots` are initialized
 */
template <typename BucketT>
CUCO_KERNEL void initialize(BucketT* buckets,
                            cuco::detail::index_type n,
                            typename BucketT::value_type value)
{
  auto const loop_stride = cuco::detail::grid_stride();
  auto idx               = cuco::detail::global_thread_id();

  while (idx < n) {
    auto& bucket_slots = *(buckets + idx);
#pragma unroll
    for (auto& slot : bucket_slots) {
      slot = value;
    }
    idx += loop_stride;
  }
}

}  // namespace detail
}  // namespace cuco
