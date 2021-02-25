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

#pragma once

#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include <cuco/detail/error.hpp>

namespace cuco {

template <typename Key,
          typename Value,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          typename Allocator       = cuco::cuda_allocator<char>>
class static_multi_map : public static_map<Key, Value, Scope, Allocator> {
 public:
  static_multi_map(std::size_t capacity,
                   Key empty_key_sentinel,
                   Value empty_value_sentinel,
                   Allocator const& alloc = Allocator{});

  ~static_multi_map();

};  // class static_multi_map
}  // namespace cuco

#include <cuco/detail/static_map.inl>
