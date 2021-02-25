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
#include <cuda/std/atomic>

#include <cuco/allocator.hpp>
#include <cuco/detail/error.hpp>

#include <cuco/static_map.cuh>

namespace cuco {

template <typename Key,
          typename Value,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          typename Allocator       = cuco::cuda_allocator<char>>
class static_multimap : public static_map<Key, Value, Scope, Allocator> {
 public:
  using typename static_map<Key, Value, Scope, Allocator>::value_type;
  using typename static_map<Key, Value, Scope, Allocator>::key_type;
  using typename static_map<Key, Value, Scope, Allocator>::mapped_type;
  using typename static_map<Key, Value, Scope, Allocator>::atomic_key_type;
  using typename static_map<Key, Value, Scope, Allocator>::atomic_mapped_type;
  using typename static_map<Key, Value, Scope, Allocator>::pair_atomic_type;
  using typename static_map<Key, Value, Scope, Allocator>::atomic_ctr_type;
  using typename static_map<Key, Value, Scope, Allocator>::allocator_type;
  using typename static_map<Key, Value, Scope, Allocator>::slot_allocator_type;

  using static_map<Key, Value, Scope, Allocator>::get_device_mutable_view;

  static_multimap(std::size_t capacity,
                  Key empty_key_sentinel,
                  Value empty_value_sentinel,
                  Allocator const& alloc = Allocator{});

  ~static_multimap();

  /**
   * @brief Inserts all key/value pairs in the range `[first, last)`.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `value_type`
   * @tparam Hash Unary callable type
   * @tparam KeyEqual Binary callable type
   * @param first Beginning of the sequence of key/value pairs
   * @param last End of the sequence of key/value pairs
   * @param hash The unary function to apply to hash each key
   * @param key_equal The binary function to compare two keys for equality
   */
  template <typename InputIt,
            typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void insert(InputIt first,
              InputIt last,
              Hash hash          = Hash{},
              KeyEqual key_equal = KeyEqual{}) override;

 private:
  using static_map<Key, Value, Scope, Allocator>::num_successes_;
  using static_map<Key, Value, Scope, Allocator>::size_;

};  // class static_multi_map
}  // namespace cuco

#include <cuco/detail/static_multimap.inl>
