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

#include <cu_collections/hash_functions.cuh>
#include <cuco/detail/error.hpp>
#include <cuda/std/atomic>
#include <atomic>
#include <cooperative_groups.h>
#include <cub/cub/cub.cuh>

#include "static_map_kernels.cuh"

namespace cuco {

/**
 * @brief Gives a power of 2 value equal to or greater than `v`.
 *
 */
constexpr std::size_t next_pow2(std::size_t v) noexcept;

/**
 * @brief Gives value to use as alignment for a pair type that is at least the
 * size of the sum of the size of the first type and second type, or 16,
 * whichever is smaller.
 */
template <typename First, typename Second>
constexpr std::size_t pair_alignment();

/**
 * @brief Custom pair type
 *
 * This is necessary because `thrust::pair` is under aligned.
 *
 * @tparam First
 * @tparam Second
 */
template <typename First, typename Second>
struct alignas(pair_alignment<First, Second>()) pair {
  using first_type = First;
  using second_type = Second;
  First first{};
  Second second{};
  pair() = default;
  __host__ __device__ constexpr pair(First f, Second s) noexcept
      : first{f}, second{s} {}
};

template <typename K, typename V>
using pair_type = cuco::pair<K, V>;

template <typename F, typename S>
__host__ __device__ pair_type<F, S> make_pair(F f, S s) noexcept {
  return pair_type<F, S>{f, s};
}
  
/**---------------------------------------------------------------------------*
  * @brief Enumeration of the possible results of attempting to insert into
  *a hash bucket
  *---------------------------------------------------------------------------**/
enum class insert_result {
  CONTINUE,  ///< Insert did not succeed, continue trying to insert
              ///< (collision)
  SUCCESS,   ///< New pair inserted successfully
  DUPLICATE  ///< Insert did not succeed, key is already present
};

// TODO: Allocator
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
  
  template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
  void contains(
    InputIt first, InputIt last, OutputIt output_begin, Hash hash, KeyEqual key_equal) noexcept;

  class device_mutable_view {
  public:
    using iterator       = pair_atomic_type*;
    using const_iterator = pair_atomic_type const*;

    device_mutable_view(pair_atomic_type* slots,
                        std::size_t capacity,
                        Key empty_key_sentinel,
                        Value empty_value_sentinel) noexcept; 

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

    std::size_t capacity() const noexcept { return capacity_; }

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
    pair_atomic_type* __restrict__ slots_{};
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
                Value empty_value_sentinel) noexcept;

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

    std::size_t capacity();

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
    pair_atomic_type* __restrict__ slots_{};
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

  device_view get_device_view() const noexcept;

  device_mutable_view get_device_mutable_view() const noexcept;

  std::size_t get_capacity();

 private:
  pair_atomic_type* slots_{nullptr};    ///< Pointer to flat slots storage
  std::size_t capacity_{};              ///< Total number of slots
  std::size_t size_{};                  ///< number of keys in map
  Key const empty_key_sentinel_{};      ///< Key value that represents an empty slot
  Value const empty_value_sentinel_{};  ///< Initial value of empty slot
  cuda::std::atomic<std::size_t> *d_num_successes_{};
};
}  // namespace cuco 

#include "static_map.inl"