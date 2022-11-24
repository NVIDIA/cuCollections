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

#pragma once

#include <cuco/detail/equal_wrapper.cuh>
#include <cuco/detail/open_address_container/open_address_container_ref.cuh>
#include <cuco/detail/pair.cuh>
#include <cuco/function.hpp>
#include <cuco/sentinel.cuh>  // TODO .hpp

#include <thrust/distance.h>

#include <cooperative_groups.h>
#include <cuda/std/atomic>
#include <type_traits>

namespace cuco {
namespace experimental {
namespace detail {

// use `insert` from `open_address_container_ref` base class
template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Functions>
class function_impl<function::insert,
                    static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Functions...>>
  : public function_impl<
      function::insert,
      open_address_container_ref<
        static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Functions...>,
        Key,
        Scope,
        KeyEqual,
        ProbingScheme,
        StorageRef,
        Functions...>> {
};

// use `contains` from `open_address_container_ref` base class
template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Functions>
class function_impl<function::contains,
                    static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Functions...>>
  : public function_impl<
      function::contains,
      open_address_container_ref<
        static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Functions...>,
        Key,
        Scope,
        KeyEqual,
        ProbingScheme,
        StorageRef,
        Functions...>> {
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
