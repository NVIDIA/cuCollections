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
#include <cub/cub.cuh>

#include <cuco/detail/pair.cuh>
#include <cuco/detail/static_map_kernels.cuh>
#include <cuco/detail/cuda_memcmp.cuh>

namespace cuco {

template<typename Key, typename Value, cuda::thread_scope Scope>
class dynamic_map;

/**
 * @brief A GPU-accelerated, unordered, associative container of key-value
 * pairs with unique keys.
 *
 * Allows constant time concurrent inserts or concurrent find operations (not
 * concurrent insert and find) from threads in device code.
 *
 * Current limitations:
 * - Requires keys that are Arithmetic
 * - Does not support erasing keys
 * - Capacity is fixed and will not grow automatically
 * - Requires the user to specify sentinel values for both key and mapped value
 * to indicate empty slots
 * - Does not support concurrent insert and find operations
 *
 * The `static_map` supports two types of operations:
 * - Host-side "bulk" operations
 * - Device-side "singular" operations
 *
 * The host-side bulk operations include `insert`, `find`, and `contains`. These
 * APIs should be used when there are a large number of keys to insert or lookup
 * in the map. For example, given a range of keys specified by device-accessible
 * iterators, the bulk `insert` function will insert all keys into the map.
 *
 * The singular device-side operations allow individual threads to to perform
 * independent insert or find/contains operations from device code. These
 * operations are accessed through non-owning, trivially copyable "view" types:
 * `device_view` and `mutable_device_view`. The `device_view` class is an
 * immutable view that allows only non-modifying operations such as `find` or
 * `contains`. The `mutable_device_view` class only allows `insert` operations.
 * The two types are separate to prevent erroneous concurrent insert/find
 * operations.
 *
 * Example:
 * \code{.cpp}
 * int empty_key_sentinel = -1;
 * int empty_value_sentine = -1;
 *
 * // Constructs a map with 100,000 slots using -1 and -1 as the empty key/value
 * // sentinels. Note the capacity is chosen knowing we will insert 50,000 keys,
 * // for an load factor of 50%.
 * static_map<int, int> m{100'000, empty_key_sentinel, empty_value_sentinel};
 *
 * // Create a sequence of pairs {{0,0}, {1,1}, ... {i,i}}
 * thrust::device_vector<thrust::pair<int,int>> pairs(50,000);
 * thrust::transform(thrust::make_counting_iterator(0),
 *                   thrust::make_counting_iterator(pairs.size()),
 *                   pairs.begin(),
 *                   []__device__(auto i){ return thrust::make_pair(i,i); };
 *
 *
 * // Inserts all pairs into the map
 * m.insert(pairs.begin(), pairs.end());
 *
 * // Get a `device_view` and passes it to a kernel where threads may perform
 * // `find/contains` lookups
 * kernel<<<...>>>(m.get_device_view());
 * \endcode
 *
 *
 * @tparam Key Arithmetic type used for key
 * @tparam Value Type of the mapped values
 * @tparam Scope The scope in which insert/find operations will be performed by
 * individual threads.
 */
template <typename Key, typename Value, cuda::thread_scope Scope = cuda::thread_scope_device>
class static_map {
  static_assert(std::is_arithmetic<Key>::value, "Unsupported, non-arithmetic key type.");
  friend class dynamic_map<Key, Value, Scope>;

  friend class dynamic_map<Key, Value, Scope>;

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
   * @brief Construct a statically sized map with the specified number of slots
   * and sentinel values.
   *
   * The capacity of the map is fixed. Insert operations will not automatically
   * grow the map. Attempting to insert more unique keys than the capacity of
   * the map results in undefined behavior.
   *
   * details here...
   * Performance begins to degrade significantly beyond a load factor of ~70%.
   * For best performance, choose a capacity that will keep the load factor
   * below 70%. E.g., if inserting `N` unique keys, choose a capacity of
   * `N * (1/0.7)`.
   *
   * The `empty_key_sentinel` and `empty_value_sentinel` values are reserved and
   * undefined behavior results from attempting to insert any key/value pair
   * that contains either.
   *
   * @param capacity The total number of slots in the map
   * @param empty_key_sentinel The reserved key value for empty slots
   * @param empty_value_sentinel The reserved mapped value for empty slots
   */
  static_map(std::size_t capacity, Key empty_key_sentinel, Value empty_value_sentinel);
  
  /**
   * @brief Destroys the map and frees its contents.
   *
   */
  ~static_map();

  /**
   * @brief Doubles the capacity of the hash map by allocating a new
   * slots_ array of twice the size as before, and initializing it to 
   * pairs of empty_key_value and empty_key_sentinel. Any previously
   * inserted key/value pairs will not be present in the new slots_ array.
   */
  void resize();
  
  /**
   * @brief Doubles the capacity of the hash map by allocating a new
   * slots_ array twice the size as before and initializing it appropriately.
   * All preexisting key/value pairs are reinserted into the new slots_ array.
   * 
   * @tparam Hash Unary callable type
   * @tparam KeyEqual Binary callable type
   * @param hash The unary function to apply to hash each key
   * @param key_equal The binary function to compare two keys for equality
   */
  template <typename Hash = MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void rehash(Hash hash = Hash{},
              KeyEqual key_equal = KeyEqual{});
  
  /**
   * @brief Inserts all key/value pairs in the range `[first, last)`.
   *
   * If multiple keys in `[first, last)` compare equal, it is unspecified which
   * element is inserted.
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
            typename Hash = MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void insert(InputIt first, InputIt last, 
              Hash hash = Hash{},
              KeyEqual key_equal = KeyEqual{});
  
  /**
   * @brief Inserts all key/value pairs in the range `[first, last)`. 
   *
   * If multiple keys in `[first, last)` compare equal, their 
   * corresponding values are summed together and mapped to that key.
   *
   * Example:
   *
   * Suppose we have a `static_map` m containing the pairs `{{2, 2}, {3, 1}}`
   *
   * If we have a sequence of `pairs` of `{{1,1}, {1,1}, {1,2}, {2, 1}}`, then
   * performing `m.insertAdd(pairs.begin(), pairs.end())` results in 
   * `m` containing the pairs `{{1, 4}, {2, 3}, {3, 1}}`.
   * 
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `value_type`
   * @tparam Hash Unary callable type
   * @tparam KeyEqual Binary callable type
   * @param first Beginning of the sequence of key/value pairs
   * @param last End of the sequence of key/value pairs
   * @param hash The unary function to apply to hash each key
   * @param key_equal The binary function used to compare two keys for equality
   */
  template <typename InputIt,
            typename Hash = MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void insertSumReduce(InputIt first, InputIt last, 
              Hash hash = Hash{},
              KeyEqual key_equal = KeyEqual{});

  /**
   * @brief Finds the values corresponding to all keys in the range `[first, last)`.
   * 
   * If the key `*(first + i)` exists in the map, copies its associated value to `(output_begin + i)`. 
   * Else, copies the empty value sentinel. 
   * 
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is 
   * convertible to the map's `mapped_type`
   * @tparam Hash Unary callable type
   * @tparam KeyEqual Binary callable type 
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of values retrieved for each key
   * @param hash The unary function to apply to hash each key
   * @param key_equal The binary function to compare two keys for equality
   */
  template <typename InputIt, typename OutputIt, 
            typename Hash = MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void find(
    InputIt first, InputIt last, OutputIt output_begin,
    Hash hash = Hash{}, 
    KeyEqual key_equal = KeyEqual{}) noexcept;
  
  /**
   * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
   * 
   * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is 
   * convertible to the map's `mapped_type`
   * @tparam Hash Unary callable type
   * @tparam KeyEqual Binary callable type 
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param hash The unary function to apply to hash each key
   * @param key_equal The binary function to compare two keys for equality
   */
  template <typename InputIt, typename OutputIt, 
            typename Hash = MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void contains(
    InputIt first, InputIt last, OutputIt output_begin,
    Hash hash = Hash{}, 
    KeyEqual key_equal = KeyEqual{}) noexcept;
  
  /**
   * @brief Mutable, non-owning view-type that may be used in device code to
   * perform singular inserts into the map.
   *
   * `device_mutable_view` is trivially-copyable and is intended to be passed by
   * value.   
   *
   * Example:
   * \code{.cpp}
   * cuco::static_map<int,int> m{100'000, -1, -1};
   *
   * // Inserts a sequence of pairs {{0,0}, {1,1}, ... {i,i}}
   * thrust::for_each(thrust::make_counting_iterator(0),
   *                  thrust::make_counting_iterator(50'000),
   *                  [map = m.get_mutable_device_view()]
   *                  __device__ (auto i) mutable {
   *                     map.insert(thrust::make_pair(i,i));
   *                  });
   * \endcode
   */
  class device_mutable_view {
  public:
    using iterator       = pair_atomic_type*;
    using const_iterator = pair_atomic_type const*;
    
    __device__ device_mutable_view& operator=(device_mutable_view const& lhs) {
      slots_ = lhs.slots_;
      capacity_ = lhs.capacity_;
      empty_key_sentinel_ = lhs.empty_key_sentinel_;
      empty_value_sentinel_ = lhs.empty_value_sentinel_;
      return *this;
    }
    
    device_mutable_view() noexcept {}

    /**
     * @brief Construct a mutable view of the first `capacity` slots of the
     * slots array pointed to by `slots`.
     *
     * @param slots Pointer to beginning of initialized slots array
     * @param capacity The number of slots viewed by this object
     * @param empty_key_sentinel The reserved value for keys to represent empty
     * slots
     * @param empty_value_sentinel The reserved value for mapped values to
     * represent empty slots
     */
    device_mutable_view(pair_atomic_type* slots,
                        std::size_t capacity,
                        Key empty_key_sentinel,
                        Value empty_value_sentinel) noexcept :
      slots_{slots},
      capacity_{capacity},
      empty_key_sentinel_{empty_key_sentinel},
      empty_value_sentinel_{empty_value_sentinel} {}
    
    /**
     * @brief Inserts the specified key/value pair into the map.
     *
     * Returns a pair consisting of an iterator to the inserted element (or to
     * the element that prevented the insertion) and a `bool` denoting whether
     * the insertion took place.
     *
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param insert_pair The pair to insert
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys for
     * equality
     * @return A pair containing an iterator to the inserted key and a bool indicating if
     * the insertion was successful
     */
    template <typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ thrust::pair<iterator, bool> insert(
      value_type const& insert_pair,
      Hash hash = Hash{},
      KeyEqual key_equal = KeyEqual{}) noexcept;
    
    /**
     * @brief Inserts the specified key/value pair into the map.
     *
     * Returns a pair consisting of an iterator to the inserted element (or to
     * the element that prevented the insertion) and a `bool` denoting whether
     * the insertion took place. Uses the CUDA Cooperative Groups API to 
     * to leverage multiple threads to perform a single insert. This provides a 
     * significant boost in throughput compared to the non Cooperative Group
     * `insert` at moderate to high load factors.
     *
     * @tparam Cooperative Group type
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     *
     * @param g The Cooperative Group that performs the insert
     * @param insert_pair The pair to insert
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys for
     * equality
     * @return A pair containing an iterator to the inserted key and a bool indicating if
     * the insertion was successful
     */
    template <typename CG,
              typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ thrust::pair<iterator, bool> insert(
      CG g,
      value_type const& insert_pair,
      Hash hash = Hash{},
      KeyEqual key_equal = KeyEqual{}) noexcept;
      
    /**
     * @brief Gets the maximum number of elements the hash map can hold.
     * 
     * @return The maximum number of elements the hash map can hold
     */
    std::size_t get_capacity() const noexcept { return capacity_; }

    /**
     * @brief Gets the sentinel value used to represent an empty key slot.
     *
     * @return The sentinel value used to represent an empty key slot
     */
    Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }
    
    /**
     * @brief Gets the sentinel value used to represent an empty value slot.
     *
     * @return The sentinel value used to represent an empty value slot
     */
    Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }
    
    /**
     * @brief Returns a const_iterator to one past the last element.
     *
     * @return A const_iterator to one past the last element
     */
    __host__ __device__ const_iterator end() const noexcept { return slots_ + capacity_; } 

    /**
     * @brief Returns an iterator to one past the last element.
     *
     * @return An iterator to one past the last element
     */
    __host__ __device__ iterator end() noexcept { return slots_ + capacity_; }

  private:
    pair_atomic_type* slots_{};          ///< Pointer to flat slots storage
    std::size_t capacity_{};       ///< Total number of slots
    Key empty_key_sentinel_{};     ///< Key value that represents an empty slot
    Value empty_value_sentinel_{}; ///< Initial Value of empty slot
    
    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * @tparam Hash Unary callable type
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the initial slot for `k`
     */
    template <typename Hash>
    __device__ iterator initial_slot(Key const& k, Hash hash) const noexcept;
    
    /**
     * @brief Returns the initial slot for a given key `k`
     * 
     * To be used for Cooperative Group based probing.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @param g the Cooperative Group for which the initial slot is needed
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the initial slot for `k`
     */
    template<typename CG, typename Hash>
    __device__ iterator initial_slot(CG g, Key const& k, Hash hash) const noexcept;

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot.
     *
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    __device__ iterator next_slot(iterator s) const noexcept;
    
    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot. To
     * be used for Cooperative Group based probing.
     *
     * @tparam CG The Cooperative Group type
     * @param g The Cooperative Group for which the next slot is needed
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    template<typename CG>
    __device__ iterator next_slot(CG g, iterator s) const noexcept;
  }; // class device mutable view
  
  /**
   * @brief Non-owning view-type that may be used in device code to
   * perform singular find and contains operations for the map.
   *
   * `device_view` is trivially-copyable and is intended to be passed by
   * value.
   *
   */
  class device_view {
  public:
    using iterator       = pair_atomic_type*;
    using const_iterator = pair_atomic_type const*;

    __device__ device_view& operator=(device_view const& lhs) {
      slots_ = lhs.slots_;
      capacity_ = lhs.capacity_;
      empty_key_sentinel_ = lhs.empty_key_sentinel_;
      empty_value_sentinel_ = lhs.empty_value_sentinel_;
      return *this;
    }
    
    device_view() noexcept {}
    
    /**
     * @brief Construct a view of the first `capacity` slots of the
     * slots array pointed to by `slots`.
     *
     * @param slots Pointer to beginning of initialized slots array
     * @param capacity The number of slots viewed by this object
     * @param empty_key_sentinel The reserved value for keys to represent empty
     * slots
     * @param empty_value_sentinel The reserved value for mapped values to
     * represent empty slots
     */
    device_view(pair_atomic_type* slots,
                std::size_t capacity,
                Key empty_key_sentinel,
                Value empty_value_sentinel) noexcept :
      slots_{slots},
      capacity_{capacity},
      empty_key_sentinel_{empty_key_sentinel},
      empty_value_sentinel_{empty_value_sentinel} {}

    /**
     * @brief Finds the value corresponding to the key `k`.
     * 
     * Returns an iterator to the pair whose key is equivalent to `k`. 
     * If no such pair exists, returns `end()`.  
     *
     * @tparam Hash Unary callable type 
     * @tparam KeyEqual Binary callable type 
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return An iterator to the position at which the key/value pair
     * containing `k` was inserted 
     */
    template <typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ iterator find(Key const& k,
                             Hash hash = Hash{},
                             KeyEqual key_equal = KeyEqual{}) noexcept;
    
    /**
     * @brief Finds the value corresponding to the key `k`.
     * 
     * Returns an iterator to the pair whose key is equivalent to `k`. 
     * If no such pair exists, returns `end()`. Uses the CUDA Cooperative Groups API to 
     * to leverage multiple threads to perform a single find. This provides a 
     * significant boost in throughput compared to the non Cooperative Group
     * `find` at moderate to high load factors.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type 
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the find
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return An iterator to the position at which the key/value pair
     * containing `k` was inserted 
     */
    template <typename CG,
              typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ iterator find(CG g, Key const& k,
                             Hash hash = Hash{},
                             KeyEqual key_equal = KeyEqual{}) noexcept;
    
    /**
     * @brief Indicates whether the key `k` was inserted into the map.
     * 
     * If the key `k` was inserted into the map, find returns
     * true. Otherwise, it returns false.
     *
     * @tparam Hash Unary callable type 
     * @tparam KeyEqual Binary callable type 
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return A boolean indicating whether the key/value pair
     * containing `k` was inserted 
     */
    template<typename Hash = MurmurHash3_32<key_type>,
             typename KeyEqual = thrust::equal_to<key_type>>
    __device__ bool contains(Key const& k, Hash hash, KeyEqual key_equal) noexcept;
    
    /**
     * @brief Indicates whether the key `k` was inserted into the map.
     *
     * If the key `k` was inserted into the map, find returns
     * true. Otherwise, it returns false. Uses the CUDA Cooperative Groups API to 
     * to leverage multiple threads to perform a single contains operation. This provides a 
     * significant boost in throughput compared to the non Cooperative Group
     * `contains` at moderate to high load factors.
     * 
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type 
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the contains operation
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return A boolean indicating whether the key/value pair
     * containing `k` was inserted 
     */
    template <typename CG, 
              typename Hash = MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ bool contains(CG g, Key const& k, Hash hash, KeyEqual key_equal) noexcept;
    
    /**
     * @brief Gets the maximum number of elements the hash map can hold.
     * 
     * @return The maximum number of elements the hash map can hold
     */
    __host__ __device__ std::size_t get_capacity() const noexcept { return capacity_; }

    /**
     * @brief Gets the sentinel value used to represent an empty key slot.
     *
     * @return The sentinel value used to represent an empty key slot
     */
    __host__ __device__ Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }
    
    /**
     * @brief Gets the sentinel value used to represent an empty value slot.
     *
     * @return The sentinel value used to represent an empty value slot
     */
    __host__ __device__ Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }
    
    /**
     * @brief Returns a const_iterator to one past the last element.
     *
     * @return A const_iterator to one past the last element
     */
    __host__ __device__ const_iterator end() const noexcept { return slots_ + capacity_; } 

    /**
     * @brief Returns an iterator to one past the last element.
     *
     * @return An iterator to one past the last element
     */
    __host__ __device__ iterator end() noexcept { return slots_ + capacity_; }

  private:
    pair_atomic_type* slots_{};          ///< Pointer to flat slots storage
    std::size_t capacity_{};       ///< Total number of slots
    Key empty_key_sentinel_{};     ///< Key value that represents an empty slot
    Value empty_value_sentinel_{}; ///< Initial Value of empty slot
    
    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * @tparam Hash Unary callable type
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the initial slot for `k`
     */
    template <typename Hash>
    __device__ iterator initial_slot(Key const& k, Hash hash) const noexcept;
    
    /**
     * @brief Returns the initial slot for a given key `k`
     * 
     * To be used for Cooperative Group based probing.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @param g the Cooperative Group for which the initial slot is needed
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the initial slot for `k`
     */
    template<typename CG, typename Hash>
    __device__ iterator initial_slot(CG g, Key const& k, Hash hash) const noexcept;

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot.
     *
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    __device__ iterator next_slot(iterator s) const noexcept;
    
    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot. To
     * be used for Cooperative Group based probing.
     *
     * @tparam CG The Cooperative Group type
     * @param g The Cooperative Group for which the next slot is needed
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    template<typename CG>
    __device__ iterator next_slot(CG g, iterator s) const noexcept;
  }; // class device_view
  
  /**
   * @brief Gets the maximum number of elements the hash map can hold.
   * 
   * @return The maximum number of elements the hash map can hold
   */
  std::size_t get_capacity() const noexcept { return capacity_; }

  /**
   * @brief Gets the number of elements in the hash map.
   * 
   * @return The number of elements in the map
   */
  std::size_t get_size() const noexcept { return size_; }
  
  /**
   * @brief Gets the load factor of the hash map.
   * 
   * @return The load factor of the hash map
   */
  float get_load_factor() const noexcept { return static_cast<float>(size_) / capacity_; } 

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

  /**
   * @brief Gets the sentinel value used to represent an empty value slot.
   *
   * @return The sentinel value used to represent an empty value slot
   */
  Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }

  /**
   * @brief Constructs a device_view object based on the members of the `static_map` object.
   * 
   * @return A device_view object based on the members of the `static_map` object
   */
  device_view get_device_view() const noexcept {
    return device_view(slots_, capacity_, empty_key_sentinel_, empty_value_sentinel_);
  }

  /**
   * @brief Constructs a device_mutable_view object based on the members of the `static_map` object
   * 
   * @return A device_mutable_view object based on the members of the `static_map` object
   */
  device_mutable_view get_device_mutable_view() const noexcept {
    return device_mutable_view(slots_, capacity_, empty_key_sentinel_, empty_value_sentinel_);
  }


  private:
  pair_atomic_type* slots_{nullptr};    ///< Pointer to flat slots storage
  std::size_t capacity_{};              ///< Total number of slots
  std::size_t size_{};                  ///< Number of keys in map
  Key empty_key_sentinel_{};      ///< Key value that represents an empty slot
  Value empty_value_sentinel_{};  ///< Initial value of empty slot
  atomic_ctr_type *num_successes_{};
};
}  // namespace cuco 

#include <cuco/detail/static_map.inl>