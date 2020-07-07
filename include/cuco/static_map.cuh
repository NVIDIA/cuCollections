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
#include <cooperative_groups.h>
#include <cub/cub/cub.cuh>

#include "static_map_kernels.cuh"

namespace cuco {

/**
 * @brief Gives a power of 2 value equal to or greater than `v`.
 *
 */
constexpr std::size_t next_pow2(std::size_t v) noexcept {
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return ++v;
}

/**
 * @brief Gives value to use as alignment for a pair type that is at least the
 * size of the sum of the size of the first type and second type, or 16,
 * whichever is smaller.
 */
template <typename First, typename Second>
constexpr std::size_t pair_alignment() {
  return std::min(std::size_t{16}, next_pow2(sizeof(First) + sizeof(Second)));
}

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
    using atomicT = cuda::std::atomic<std::size_t>;

    auto num_keys = std::distance(first, last);
    auto const block_size = 128;
    auto const stride = 4;
    auto const tile_size = 4;
    auto const grid_size = (tile_size * num_keys + stride * block_size - 1) /
                           (stride * block_size);
    auto view = get_device_mutable_view();

    atomicT h_num_successes;
    atomicT *d_num_successes;
    cudaMalloc((void**)&d_num_successes, sizeof(atomicT));
    cudaMemset(d_num_successes, 0x00, sizeof(atomicT));

    insertKernel<block_size, tile_size>
    <<<grid_size, block_size>>>(first, first + num_keys, d_num_successes, view, 
                                hash, key_equal);
    
    cudaMemcpy(&h_num_successes, d_num_successes, sizeof(atomicT), cudaMemcpyDeviceToHost);
    size_ += h_num_successes;
    cudaFree(d_num_successes);
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
    auto const tile_size = 4;
    auto const grid_size = (tile_size * num_keys + stride * block_size - 1) /
                           (stride * block_size);
    auto view = get_device_view();
    findKernel<tile_size><<<grid_size, block_size>>>(first, last, output_begin,
                                                     view, hash, key_equal);
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
    __device__ thrust::pair<iterator, bool> insert(value_type const& insert_pair,
                                                      Hash hash = Hash{},
                                                      KeyEqual key_equal = KeyEqual{}) noexcept {

      iterator current_slot{initial_slot(insert_pair.first, hash)};

      while (true) {
        using cuda::std::memory_order_relaxed;
        auto expected_key = empty_key_sentinel_;
        auto expected_value = empty_value_sentinel_;
        auto& slot_key = current_slot->first;
        auto& slot_value = current_slot->second;

        bool key_success = slot_key.compare_exchange_strong(expected_key,
                                                            insert_pair.first,
                                                            memory_order_relaxed);
        bool value_success = slot_value.compare_exchange_strong(expected_value,
                                                                insert_pair.second,
                                                                memory_order_relaxed);

        if(key_success) {
          while(not value_success) {
            value_success = slot_value.compare_exchange_strong(expected_value = empty_value_sentinel_,
                                                               insert_pair.second,
                                                               memory_order_relaxed);
          }
          return thrust::make_pair(current_slot, true);
        }
        else if(value_success) {
          slot_value.store(empty_value_sentinel_, memory_order_relaxed);
        }
        
        // if the key was already inserted by another thread, than this instance is a
        // duplicate, so the insert fails
        if (key_equal(insert_pair.first, expected_key)) {
          return thrust::make_pair(current_slot, false);
        }
        
        // if we couldn't insert the key, but it wasn't a duplicate, then there must
        // have been some other key there, so we keep looking for a slot
        current_slot = next_slot(current_slot);
      }
    }

    template <typename CG, typename Hash, typename KeyEqual>
    __device__ thrust::pair<iterator, bool> insert(CG g,
                                                   value_type const& insert_pair,
                                                   Hash hash,
                                                   KeyEqual key_equal) noexcept {
      std::size_t const key_hash = hash(insert_pair.first);
      uint32_t window_idx = 0;
      
      while(true) {
        std::size_t index = (key_hash + window_idx * g.size() + g.thread_rank()) % capacity_;
        iterator current_slot = &slots_[index];
        key_type const existing_key = current_slot->first;
        uint32_t existing = g.ballot(key_equal(existing_key, insert_pair.first));
        
        // the key we are trying to insert is already in the map, so we return
        // with failure to insert
        if(existing) {
          return thrust::make_pair(current_slot, false);
        }
        
        uint32_t empty = g.ballot(key_equal(existing_key, empty_key_sentinel_));

        // we found an empty slot, but not the key we are inserting, so this must
        // be an empty slot into which we can insert the key
        if(empty) {
          // the first lane in the group with an empty slot will attempt the insert
          insert_result status{insert_result::CONTINUE};
          uint32_t srcLane = __ffs(empty) - 1;

          if(g.thread_rank() == srcLane) {
            using cuda::std::memory_order_relaxed;
            auto expected_key = empty_key_sentinel_;
            auto expected_value = empty_value_sentinel_;
            auto& slot_key = current_slot->first;
            auto& slot_value = current_slot->second;

            bool key_success = slot_key.compare_exchange_strong(expected_key,
                                                                insert_pair.first,
                                                                memory_order_relaxed);
            bool value_success = slot_value.compare_exchange_strong(expected_value,
                                                                    insert_pair.second,
                                                                    memory_order_relaxed);

            if(key_success) {
              while(not value_success) {
                value_success = slot_value.compare_exchange_strong(expected_value = empty_value_sentinel_,
                                                                  insert_pair.second,
                                                                memory_order_relaxed);
              }
              status = insert_result::SUCCESS;
            }
            else if(value_success) {
              slot_value.store(empty_value_sentinel_, memory_order_relaxed);
            }
            
            // our key was already present in the slot, so our key is a duplicate
            if(key_equal(insert_pair.first, expected_key)) {
              status = insert_result::DUPLICATE;
            }
            // another key was inserted in the slot we wanted to try
            // so we need to try the next empty slot in the window
          }

          uint32_t res_status = g.shfl(static_cast<uint32_t>(status), srcLane);
          status = static_cast<insert_result>(res_status);

          // successful insert
          if(status == insert_result::SUCCESS) {
            intptr_t res_slot = g.shfl(reinterpret_cast<intptr_t>(current_slot), srcLane);
            return thrust::make_pair(reinterpret_cast<iterator>(res_slot), true);
          }
          // duplicate present during insert
          if(status == insert_result::DUPLICATE) {
            intptr_t res_slot = g.shfl(reinterpret_cast<intptr_t>(current_slot), srcLane);
            return thrust::make_pair(reinterpret_cast<iterator>(res_slot), false);
          }
          // if we've gotten this far, a different key took our spot 
          // before we could insert. We need to retry the insert on the
          // same window
        }
        // if there are no empty slots in the current window,
        // we move onto the next window
        else {
          window_idx++;
        }
      }
    }

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

    template <typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ iterator find(Key const& k,
                             Hash hash = Hash{},
                             KeyEqual key_equal = KeyEqual{}) noexcept {
      auto current_slot = initial_slot(k, hash);

      while (true) {
        auto const current_key =
            current_slot->first.load(cuda::std::memory_order_relaxed);
        // Key exists, return iterator to location
        if (key_equal(k, current_key)) {
          return current_slot;
        }

        // Key doesn't exist, return end()
        if (key_equal(empty_key_sentinel_, current_key)) {
          return end();
        }

        current_slot = next_slot(current_slot);
      }
    }

    template <typename CG, 
              typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ iterator find(CG g, Key const& k,
                             Hash hash = Hash{},
                             KeyEqual key_equal = KeyEqual{}) noexcept {
      uint32_t const key_hash = hash(k);
      uint32_t window_idx = 0;

      while(true) {
        uint32_t index = (key_hash + window_idx * g.size() + g.thread_rank()) % capacity_;
        auto const current_bucket = &slots_[index];
        key_type const existing_key = current_bucket->first.load(cuda::std::memory_order_relaxed);
        uint32_t existing = g.ballot(key_equal(existing_key, k));
        
        // the key we were searching for was found by one of the threads,
        // so we return an iterator to the entry
        if(existing) {
          uint32_t src_lane = __ffs(existing) - 1;
          intptr_t res_bucket = g.shfl(reinterpret_cast<intptr_t>(current_bucket), src_lane);
          return reinterpret_cast<pair_atomic_type*>(res_bucket);
        }
        
        // we found an empty slot, meaning that the key we're searching 
        // for isn't in this submap, so we should move onto the next one
        uint32_t empty = g.ballot(key_equal(existing_key, empty_key_sentinel_));
        if(empty) {
          return end();
        }

        // otherwise, all slots in the current window are full with other keys,
        // so we move onto the next window in the current submap
        window_idx++;
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
  std::size_t size_{};                  ///< number of keys in map
  Key const empty_key_sentinel_{};      ///< Key value that represents an empty slot
  Value const empty_value_sentinel_{};  ///< Initial value of empty slot
};
}  // namespace cuco 