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
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/remove.h>

#include <cuco/detail/error.hpp>
#include <cuda/std/atomic>
#include <atomic>
#include <cooperative_groups.h>
#include <../thirdparty/cub/cub/cub.cuh>

#include <cuco/static_map.cuh>
#include <cuco/detail/dynamic_map_kernels.cuh>

namespace cuco {



template <typename Key, typename Value, cuda::thread_scope Scope = cuda::thread_scope_device>
class dynamic_map {
  static_assert(std::is_arithmetic<Key>::value, "Unsupported, non-arithmetic key type.");

  public:
  using key_type           = Key;
  using mapped_type        = Value;
  using atomic_ctr_type = cuda::atomic<std::size_t, Scope>;
  using view_type = typename static_map<Key, Value, Scope>::device_view;
  using mutable_view_type = typename static_map<Key, Value, Scope>::device_mutable_view;
  dynamic_map(dynamic_map const&) = delete;
  dynamic_map(dynamic_map&&)      = delete;
  dynamic_map& operator=(dynamic_map const&) = delete;
  dynamic_map& operator=(dynamic_map&&) = delete;

  // used for insert, not too sure where to put it
  struct already_exists {
    __host__ __device__ 
    bool operator()(thrust::tuple<bool, cuco::pair_type<Key, Value>> x) {
      return thrust::get<0>(x);
    }
  };

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

  void reserve(std::size_t num_to_insert);

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
  key_type empty_key_sentinel_{};
  mapped_type empty_value_sentinel_{};
    
  static constexpr std::size_t MAX_NUM_SUBMAPS_ = 16;
  std::vector<static_map<key_type, mapped_type, Scope> *> submaps_;
  std::vector<std::size_t> submap_caps_;
  std::vector<view_type> submap_views_;
  std::vector<mutable_view_type> submap_mutable_views_;
  std::size_t num_elements_{};
  std::size_t min_insert_size_{};
  float max_load_factor_{};
  atomic_ctr_type *num_successes_;
};
}  // namespace cuco

#include <cuco/detail/dynamic_map.inl>