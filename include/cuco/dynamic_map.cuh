/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>

#include "../cu_collections/hash_functions.cuh"
#include "detail/error.hpp"
#include <cuda/std/atomic>
#include <atomic>
#include <cooperative_groups.h>
#include <../thirdparty/cub/cub/cub.cuh>

#include <cuco/static_map.cuh>

namespace cuco {



template <typename Key, typename Value, cuda::thread_scope Scope = cuda::thread_scope_device>
class dynamic_map {
  static_assert(std::is_arithmetic<Key>::value, "Unsupported, non-arithmetic key type.");

  public:
  using key_type           = Key;
  using mapped_type        = Value;
  dynamic_map(dynamic_map const&) = delete;
  dynamic_map(dynamic_map&&)      = delete;
  dynamic_map& operator=(dynamic_map const&) = delete;
  dynamic_map& operator=(dynamic_map&&) = delete;

  /**
  * @brief Construct a fixed-size map with the specified capacity and sentinel values.
  *
  * details here...
  *
  * @param capacity The total number of slots in the map
  * @param empty_key_sentinel The reserved key value for empty slots
  * @param empty_value_sentinel The reserved mapped value for empty slots
  */
  dynamic_map(std::size_t capacity, Key empty_key_sentinel, Value empty_value_sentinel);

  ~dynamic_map();

  template <typename InputIt,
            typename Hash = MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void insert(InputIt first, InputIt last, 
              Hash hash = Hash{},
              KeyEqual key_equal = KeyEqual{});
              
  template <typename InputIt,
            typename Hash = MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void insertReduce(InputIt first, InputIt last, 
                    mapped_type f(mapped_type, mapped_type),                
                    Hash hash = Hash{},
                    KeyEqual key_equal = KeyEqual{});


  template <typename InputIt, typename OutputIt, 
            typename Hash = MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void find(
    InputIt first, InputIt last, OutputIt output_begin,
    Hash hash = Hash{}, 
    KeyEqual key_equal = KeyEqual{}) noexcept;

  template <typename InputIt, typename OutputIt, 
            typename Hash = MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void contains(
    InputIt first, InputIt last, OutputIt output_begin,
    Hash hash = Hash{}, 
    KeyEqual key_equal = KeyEqual{}) noexcept;


  std::size_t get_capacity() const noexcept;

  float get_load_factor() const noexcept;

  private:
  static_map<key_type, mapped_type, Scope>* submaps_{nullptr};    ///< Pointer to flat slots storage
};
}  // namespace cuco 