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
#include <simt/atomic>

#include "static_map_kernels.cuh"

namespace cuco {

// TODO: replace with custom pair type
template <typename K, typename V>
using pair_type = thrust::pair<K, V>;

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
  static_map(std::size_t capacity, Key empty_key_sentinel, Value empty_value_sentinel) : 
    capacity_{capacity},
    empty_key_sentinel_{empty_key_sentinel},
    empty_value_sentinel_{empty_value_sentinel} {
      cudaMalloc(&slots_, capacity * sizeof(pair_atomic_type));
      
      auto constexpr block_size = 256;
      auto constexpr stride = 4;
      auto const grid_size = (capacity + stride * block_size - 1) / (stride * block_size);
      initializeKernel
      <atomic_key_type, atomic_mapped_type>
      <<<grid_size, block_size>>>(slots_, empty_key_sentinel,
                                            empty_value_sentinel, capacity);
    }

  ~static_map() {
    cudaFree(slots_);
  }

  template <typename InputIt,
            typename Hash = MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void insert(InputIt first, InputIt last, 
              Hash hash = Hash{},
              KeyEqual key_equal = KeyEqual{}) {
    
    auto num_keys = std::distance(first, last);

    auto const block_size = 128;
    auto const stride = 1;
    auto const grid_size = (num_keys + stride * block_size - 1) / (stride * block_size);
    auto view = get_device_mutable_view();
    //insertKernel<<<grid_size, block_size>>>(it, it + num_keys, view, hash, key_equal);
    cudaDeviceSynchronize();
  }


  template <typename InputIt, typename OutputIt, 
            typename Hash = MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void find(
    InputIt first, InputIt last, OutputIt output_begin,
    Hash hash = Hash{}, 
    KeyEqual key_equal = KeyEqual{}) noexcept {
    
    auto num_keys = std::distance(first, last);
    auto const block_size = 128;
    auto const stride = 1;
    auto const grid_size = (num_keys + stride * block_size - 1) / (stride * block_size);
    auto view = get_device_view();
    findKernel<<<grid_size, block_size>>>(first, last, output_begin, view, hash, key_equal);
    cudaDeviceSynchronize();    
  }

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
                        Value empty_value_sentinel) noexcept :
                        slots_{slots},
                        capacity_{capacity},
                        empty_key_sentinel_{empty_key_sentinel},
                        empty_value_sentinel_{empty_value_sentinel} {}

    template <typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ cuco::pair_type<iterator, bool> insert(value_type const& insert_pair,
                                                      Hash hash = Hash{},
                                                      KeyEqual key_equal = KeyEqual{}) noexcept {

      iterator current_slot{initial_slot(insert_pair.first, hash)};

      while (true) {
        auto expected = pair_atomic_type{empty_key_sentinel_, empty_value_sentinel_};

        if (current_slot->compare_exchange_strong(expected, insert_pair)) {
          return thrust::make_pair(current_slot, true);
        }

        if (key_equal(insert_pair.first, expected.first)) {
          return thrust::make_pair(current_slot, false);
        }

        current_slot = next_slot(current_slot);
      }
    }

    template <typename CG, typename Hash, typename KeyEqual>
    __device__ cuco::pair_type<iterator, bool> insert(CG cg,
                                                 value_type const& insert_pair,
                                                 KeyEqual key_equal,
                                                 Hash hash) noexcept;

    std::size_t capacity() const noexcept { return capacity_; }

    Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

    Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }
    
    /**
     * @brief Returns iterator to one past the last element.
     *
     */
    __host__ __device__ const_iterator end() const noexcept {
      return slots_ + capacity_;
    }

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
    template <typename Hash>
    __device__ iterator initial_slot(Key const& k, Hash hash) const noexcept {
      return &slots_[hash(k) % capacity_];
    }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot.
     *
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    __device__ iterator next_slot(iterator s) const noexcept {
      // TODO: Since modulus is expensive, I think this should be more
      // efficient than doing (++index % capacity_)
      return (++s < end()) ? s : slots_;
    }
    
     };

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

    template <typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ iterator find(Key const& k,
                             KeyEqual key_equal = KeyEqual{},
                             Hash hash = Hash{}) noexcept {
      return nullptr;
    }

    template <typename CG, 
              typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ iterator find(CG cg, Key const& k,
                             KeyEqual key_equal = KeyEqual{}, Hash hash = Hash{}) {
      auto current_slot = initial_slot(k, hash);

      while (true) {
        auto const current_key =
            current_slot->load(cuda::std::memory_order_relaxed).first;
        // Key exists, return iterator to location
        if (key_equal(k, current_key)) {
          return current_slot;
        }

        // Key doesn't exist, return end()
        if (key_equal(empty_key_sentinel_, current_key)) {
          return end();
        }

        // TODO: Add check for full hash map?

        // Slot is occupied by a different key---collision
        // Advance to next slot
        current_slot = next_slot(current_slot);
      }
    }
    
    
    template <typename Hash, typename KeyEqual>
    __device__ bool contains(Key const& k, KeyEqual key_equal, Hash hash) noexcept;

    template <typename CG, typename Hash, typename KeyEqual>
    __device__ bool contains(CG cg, Key const& k, KeyEqual key_equal, Hash hash) noexcept;

    std::size_t capacity();

    Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

    Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }
    
    /**
     * @brief Returns iterator to one past the last element.
     *
     */
    __host__ __device__ const_iterator end() const noexcept {
      return slots_ + capacity_;
    }

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
    template <typename Hash>
    __device__ iterator initial_slot(Key const& k, Hash hash) const noexcept {
      return &slots_[hash(k) % capacity_];
    }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot.
     *
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    __device__ iterator next_slot(iterator s) const noexcept {
      // TODO: Since modulus is expensive, I think this should be more
      // efficient than doing (++index % capacity_)
      return (++s < end()) ? s : slots_;
    }
  }; // class device_view

  Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

  Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }

  device_view get_device_view() const noexcept {
    return device_view(slots_, capacity_, empty_key_sentinel_, empty_value_sentinel_);
  }

  device_mutable_view get_device_mutable_view() const noexcept {
    return device_mutable_view(slots_, capacity_, empty_key_sentinel_, empty_value_sentinel_);
  }

  std::size_t get_capacity();

 private:
  pair_atomic_type* slots_{nullptr};    ///< Pointer to flat slots storage
  std::size_t capacity_{};              ///< Total number of slots
  Key const empty_key_sentinel_{};      ///< Key value that represents an empty slot
  Value const empty_value_sentinel_{};  ///< Initial value of empty slot
};

}  // namespace cuco 