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

#include <cuco/hash_functions.cuh>
#include <cuco/detail/error.hpp>
#include <cuda/std/atomic>
#include <atomic>
#include <cooperative_groups.h>
#include <../thirdparty/cub/cub/cub.cuh>

#include <cuco/detail/static_map_kernels.cuh>
#include <cuco/detail/cuda_memcmp.cuh>
#include <cuco/detail/pair.cuh>

namespace cuco {

  

template <typename Key, typename Value, cuda::thread_scope Scope = cuda::thread_scope_device>
class static_map {
  static_assert(std::is_arithmetic<Key>::value, "Unsupported, non-arithmetic key type.");

  public:
  using value_type         = cuco::pair_type<Key, Value>;
  using key_type           = Key;
  using mapped_type        = Value;
  using atomic_key_type    = cuda::atomic<key_type, Scope>;
  using atomic_mapped_type = cuda::atomic<mapped_type, Scope>;
  using pair_atomic_type   = cuco::pair_type<atomic_key_type, atomic_mapped_type>;
  using atomic_ctr_type    = cuda::atomic<std::size_t, Scope>;
  
  static_map(static_map const&) = delete;
  static_map(static_map&&)      = delete;
  static_map& operator=(static_map const&) = delete;
  static_map& operator=(static_map&&) = delete;

  /**
   * @brief Construct a fixed-size map with the specified capacity and sentinel values.
   *
   * details here...
   *
   * @param capacity The total number of slots in the map
   * @param empty_key_sentinel The reserved key value for empty slots
   * @param empty_value_sentinel The reserved mapped value for empty slots
   */
  static_map(std::size_t capacity, Key empty_key_sentinel, Value empty_value_sentinel);

  ~static_map();

  template <typename InputIt,
            typename Hash = MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void insert(InputIt first, InputIt last, 
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

  class device_mutable_view {
  public:
    using iterator       = pair_atomic_type*;
    using const_iterator = pair_atomic_type const*;
    
    device_mutable_view(pair_atomic_type* slots,
                        std::size_t capacity,
                        Key empty_key_sentinel,
                        Value empty_value_sentinel) noexcept :
      slots_{slots},
      capacity_{capacity},
      empty_key_sentinel_{empty_key_sentinel},
      empty_value_sentinel_{empty_value_sentinel} {}

    template <typename Iterator = iterator,
              typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ thrust::pair<Iterator, bool> insert(value_type const& insert_pair,
                                                      Hash hash = Hash{},
                                                      KeyEqual key_equal = KeyEqual{}) noexcept;

    template <typename CG,
              typename Iterator = iterator,
              typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ thrust::pair<Iterator, bool> insert(CG g,
                                                   value_type const& insert_pair,
                                                   Hash hash,
                                                   KeyEqual key_equal) noexcept;

    std::size_t get_capacity() const noexcept { return capacity_; }

    Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

    Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }
    
    /**
     * @brief Returns iterator to one past the last element.
     *
     */
    __host__ __device__ const_iterator end() const noexcept { return slots_ + capacity_; } 

    /**
     * @brief Returns iterator to one past the last element.
     *
     */
    __host__ __device__ iterator end() noexcept { return slots_ + capacity_; }

  private:
    pair_atomic_type* slots_{};
    std::size_t const capacity_{};
    Key const empty_key_sentinel_{};
    Value const empty_value_sentinel_{};
    
    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * @tparam Hash
     * @param k The key to get the slot for
     * @param hash Hash to use to determine the slot
     * @return Pointer to the initial slot for `k`
     */
    template <typename Hash, typename Iterator = iterator>
    __device__ Iterator initial_slot(Key const& k, Hash hash) const noexcept;

    template<typename CG, typename Hash, typename Iterator = iterator>
    __device__ Iterator initial_slot(CG g, Key const& k, Hash hash) const noexcept;

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot.
     *
     * @param s The slot to advance
     * @return The next slot after `s`
     */
     template<typename Iterator = iterator>
    __device__ Iterator next_slot(Iterator s) const noexcept;

    template<typename CG, typename Iterator = iterator>
    __device__ Iterator next_slot(CG g, Iterator s) const noexcept;
  }; // class device mutable view

  class device_view {
  public:
    using iterator       = pair_atomic_type*;
    using const_iterator = pair_atomic_type const*;

    device_view(pair_atomic_type* slots,
                std::size_t capacity,
                Key empty_key_sentinel,
                Value empty_value_sentinel) noexcept :
      slots_{slots},
      capacity_{capacity},
      empty_key_sentinel_{empty_key_sentinel},
      empty_value_sentinel_{empty_value_sentinel} {}


    template <typename Iterator = iterator,
              typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ Iterator find(Key const& k,
                             Hash hash = Hash{},
                             KeyEqual key_equal = KeyEqual{}) noexcept;

    template <typename CG,
              typename Iterator = iterator,
              typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ Iterator find(CG g, Key const& k,
                             Hash hash = Hash{},
                             KeyEqual key_equal = KeyEqual{}) noexcept;

    template<typename Hash = MurmurHash3_32<key_type>,
             typename KeyEqual = thrust::equal_to<key_type>>
    __device__ bool contains(Key const& k, Hash hash, KeyEqual key_equal) noexcept;

    template <typename CG, 
              typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ bool contains(CG g, Key const& k, Hash hash, KeyEqual key_equal) noexcept;


    std::size_t get_capacity() const noexcept { return capacity_; }

    Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

    Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }
    
    /**
     * @brief Returns iterator to one past the last element.
     *
     */
    __host__ __device__ const_iterator end() const noexcept { return slots_ + capacity_; }

    /**
     * @brief Returns iterator to one past the last element.
     *
     */
    __host__ __device__ iterator end() noexcept { return slots_ + capacity_; }

  private:
    pair_atomic_type* slots_{};
    std::size_t const capacity_{};
    Key const empty_key_sentinel_{};
    Value const empty_value_sentinel_{};
    
    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * @tparam Hash
     * @param k The key to get the slot for
     * @param hash Hash to use to determine the slot
     * @return Pointer to the initial slot for `k`
     */
    template <typename Hash, typename Iterator = iterator>
    __device__ Iterator initial_slot(Key const& k, Hash hash) const noexcept;
    
    template<typename CG, typename Hash, typename Iterator = iterator>
    __device__ Iterator initial_slot(CG g, Key const& k, Hash hash) const noexcept;

    template<typename Iterator = iterator>
    __device__ Iterator next_slot(Iterator s) const noexcept;
    
    template<typename CG, typename Iterator = iterator>
    __device__ Iterator next_slot(CG g, Iterator s) const noexcept;
  }; // class device_view

  Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

  Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }

  device_view get_device_view() const noexcept {
    return device_view(slots_, capacity_, empty_key_sentinel_, empty_value_sentinel_);
  }

  device_mutable_view get_device_mutable_view() const noexcept {
    return device_mutable_view(slots_, capacity_, empty_key_sentinel_, empty_value_sentinel_);
  }

  std::size_t get_capacity() const noexcept { return capacity_; }
  
  float get_load_factor() const noexcept { return static_cast<float>(size_) / capacity_; } 

  private:
  pair_atomic_type* slots_{nullptr};    ///< Pointer to flat slots storage
  std::size_t capacity_{};              ///< Total number of slots
  std::size_t size_{};                  ///< number of keys in map
  Key const empty_key_sentinel_{};      ///< Key value that represents an empty slot
  Value const empty_value_sentinel_{};  ///< Initial value of empty slot
  atomic_ctr_type *d_num_successes_{};
};
}  // namespace cuco 

#include "detail/static_map.inl"