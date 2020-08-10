/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
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

#include <thrust/pair.h>
#include <cassert>
#include <iostream>
#include <iterator>
#include <limits>
#include <type_traits>
#include <stdint.h>

//#include <utilities/legacy/device_atomics.cuh>
#include "../cudf_include/helper_functions.cuh"
#include "../cudf_include/allocator.cuh"
#include "../thirdparty/cub/cub/cub.cuh"
#include <cooperative_groups.h>
#include "chain_kernels.cuh"
#include <cu_collections/hash_functions.cuh>
  
namespace cg = cooperative_groups;

#ifndef AN
#define AN
namespace {
template <std::size_t N>
struct packed {
  using type = void;
};
template <>
struct packed<sizeof(unsigned long long int)> {
  using type = unsigned long long int;
};
template <>
struct packed<sizeof(unsigned int)> {
  using type = unsigned int;
};
template <typename pair_type>
using packed_t = typename packed<sizeof(pair_type)>::type;

/**---------------------------------------------------------------------------*
 * @brief Indicates if a pair type can be packed.
 *
 * When the size of the key,value pair being inserted into the hash table is
 * equal in size to a type where atomicCAS is natively supported, it is more
 * efficient to "pack" the pair and insert it with a single atomicCAS.
 *
 * @note Only integral key and value types may be packed because we use
 * bitwise equality comparison, which may not be valid for non-integral
 * types.
 *
 * @tparam pair_type The pair type in question
 * @return true If the pair type can be packed
 * @return false  If the pair type cannot be packed
 *---------------------------------------------------------------------------**/
template <typename pair_type,
          typename key_type = typename pair_type::first_type,
          typename value_type = typename pair_type::second_type>
constexpr bool is_packable() {
  return std::is_integral<key_type>::value and
         std::is_integral<value_type>::value and
         not std::is_void<packed_t<pair_type>>::value;
}

/**---------------------------------------------------------------------------*
 * @brief Allows viewing a pair in a packed representation
 *
 * Used as an optimization for inserting when a pair can be inserted with a
 * single atomicCAS
 *---------------------------------------------------------------------------**/
template <typename pair_type, typename Enable = void>
union pair_packer;

template <typename pair_type>
union pair_packer<pair_type, std::enable_if_t<is_packable<pair_type>()>> {
  using packed_type = packed_t<pair_type>;
  packed_type const packed;
  pair_type const pair;

  __device__ pair_packer(pair_type _pair) : pair{_pair} {}

  __device__ pair_packer(packed_type _packed) : packed{_packed} {}
};
}  // namespace

#endif

/**
 * Supports concurrent insert, but not concurrent insert and find.
 *
 * TODO:
 *  - add constructor that takes pointer to hash_table to avoid allocations
 *  - extend interface to accept streams
 */
template <typename Key, typename Element,
          typename Hasher = MurmurHash3_32<Key>,
          typename Equality = equal_to<Key>,
          typename Allocator = cuda_allocator<thrust::pair<Key, Element>>>
class concurrent_unordered_map_chain {
 public:
  using size_type = size_t;
  using hasher = Hasher;
  using key_equal = Equality;
  using allocator_type = Allocator;
  using key_type = Key;
  using mapped_type = Element;
  using value_type = thrust::pair<Key, Element>;
  using iterator = cycle_iterator_adapter<value_type*>;
  using const_iterator = const cycle_iterator_adapter<value_type*>;

  static constexpr uint32_t m_max_num_submaps = 8;
  static constexpr float m_max_load_factor = 0.5;
  static constexpr uint32_t insertGran = 1;
  static constexpr uint32_t minKernelSize = 10'000;

 public:
  /**---------------------------------------------------------------------------*
   * @brief Factory to construct a new concurrent unordered map.
   *
   * Returns a `std::unique_ptr` to a new concurrent unordered map object. The
   * map is non-owning and trivially copyable and should be passed by value into
   * kernels. The `unique_ptr` contains a custom deleter that will free the
   * map's contents.
   *
   * @note The implementation of this unordered_map uses sentinel values to
   * indicate an entry in the hash table that is empty, i.e., if a hash bucket
   *is empty, the pair residing there will be equal to (unused_key,
   *unused_element). As a result, attempting to insert a key equal to
   *`unused_key` results in undefined behavior.
   *
   * @param capacity The maximum number of pairs the map may hold
   * @param unused_element The sentinel value to use for an empty value
   * @param unused_key The sentinel value to use for an empty key
   * @param hash_function The hash function to use for hashing keys
   * @param equal The equality comparison function for comparing if two keys are
   * equal
   * @param allocator The allocator to use for allocation the hash table's
   * storage
   *---------------------------------------------------------------------------**/
  static auto create(
      size_type capacity,
      const mapped_type unused_element = std::numeric_limits<key_type>::max(),
      const key_type unused_key = std::numeric_limits<key_type>::max(),
      const Hasher& hash_function = hasher(),
      const Equality& equal = key_equal(),
      const allocator_type& allocator = allocator_type()) {
    using Self =
        concurrent_unordered_map_chain<Key, Element, Hasher, Equality, Allocator>;

    auto deleter = [](Self* p) { p->destroy(); };

    return std::unique_ptr<Self, std::function<void(Self*)>>{
        new Self(capacity, unused_element, unused_key, hash_function, equal,
                 allocator),
        deleter};
  }

  __device__ iterator begin() {
    return iterator(m_hashtbl_values, m_hashtbl_values + m_capacity,
                    m_hashtbl_values);
  }
  __device__ const_iterator begin() const {
    return const_iterator(m_hashtbl_values, m_hashtbl_values + m_capacity,
                          m_hashtbl_values);
  }
  __device__ iterator end() {
    return iterator(m_hashtbl_values, m_hashtbl_values + m_capacity,
                    m_hashtbl_values + m_capacity);
  }
  __device__ const_iterator end() const {
    return const_iterator(m_hashtbl_values, m_hashtbl_values + m_capacity,
                          m_hashtbl_values + m_capacity);
  }
  __device__ value_type* data() const { return m_hashtbl_values; }

  __host__ __device__ key_type get_unused_key() const { return m_unused_key; }

  __host__ __device__ mapped_type get_unused_element() const {
    return m_unused_element;
  }

  __host__ __device__ size_type capacity() const { return m_capacity; }

 private:
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



  /**---------------------------------------------------------------------------*
   * @brief Specialization for value types that can be packed.
   *
   * When the size of the key,value pair being inserted is equal in size to
   *a type where atomicCAS is natively supported, this optimization path
   *will insert the pair in a single atomicCAS operation.
   *---------------------------------------------------------------------------**/
  template <typename pair_type = value_type>
  __device__ std::enable_if_t<is_packable<pair_type>(), insert_result>
  attempt_insert(value_type* insert_location, value_type const& insert_pair) {
    pair_packer<pair_type> const unused{
        thrust::make_pair(m_unused_key, m_unused_element)};
    pair_packer<pair_type> const new_pair{insert_pair};
    pair_packer<pair_type> const old{atomicCAS(
        reinterpret_cast<typename pair_packer<pair_type>::packed_type*>(
            insert_location),
        unused.packed, new_pair.packed)};

    if (old.packed == unused.packed) {
      return insert_result::SUCCESS;
    }

    if (m_equal(old.pair.first, insert_pair.first)) {
      return insert_result::DUPLICATE;
    }
    return insert_result::CONTINUE;
  }

  /**---------------------------------------------------------------------------*
   * @brief Atempts to insert a key,value pair at the specified hash bucket.
   *
   * @param[in] insert_location Pointer to hash bucket to attempt insert
   * @param[in] insert_pair The pair to insert
   * @return Enum indicating result of insert attempt.
   *---------------------------------------------------------------------------**/
  template <typename pair_type = value_type>
  __device__ std::enable_if_t<not is_packable<pair_type>(), insert_result>
  attempt_insert(value_type* const __restrict__ insert_location,
                 value_type const& insert_pair) {
    key_type const old_key{
        atomicCAS(&(insert_location->first), m_unused_key, insert_pair.first)};

    // Hash bucket empty
    if (m_equal(m_unused_key, old_key)) {
      insert_location->second = insert_pair.second;
      return insert_result::SUCCESS;
    }

    // Key already exists
    if (m_equal(old_key, insert_pair.first)) {
      return insert_result::DUPLICATE;
    }

    return insert_result::CONTINUE;
  }

 public:
  
  __host__ void addNewSubmap() {
    uint64_t capacity = m_submap_caps[m_num_submaps] * m_capacity;
    value_type* submap = m_allocator.allocate(capacity);
    constexpr int block_size = 128;

    init_hashtbl<<<((capacity - 1) / block_size) + 1, block_size>>>(
        submap, capacity, m_unused_key, m_unused_element);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(0));

    m_submaps[m_num_submaps] = submap;
    m_num_elements[m_num_submaps] = 0;
    m_num_submaps++;
  }
  
  void freeSubmaps() {
    for(auto i = 0; i < m_num_submaps; ++i) {
      m_allocator.deallocate(m_submaps[i], m_submap_caps[i] * m_capacity);
    }
  }
  

  // bulk insert kernel is called from host so that predictive resizing can be implemented
  __host__ float bulkInsert(std::vector<key_type> const& h_keys,
                           std::vector<mapped_type> const& h_values,
                           uint32_t numKeys) {
    thrust::device_vector<key_type> d_keys( h_keys );
    thrust::device_vector<mapped_type> d_values( h_values );
    
    float temp_time = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // determine if it will be necessary to add another map
    uint64_t finalNumElements = m_total_num_elements + numKeys;
    uint64_t numSubmapsNeeded = 0;
    int64_t count = finalNumElements;
    while(count > 0) {
      count -= m_submap_caps[numSubmapsNeeded] * m_capacity * m_max_load_factor;
      numSubmapsNeeded++;
    }
    uint64_t numSubmapsToAllocate = numSubmapsNeeded - m_num_submaps;
    
    for(auto i = 0; i < numSubmapsToAllocate; ++i) {
      addNewSubmap();
    }

    // perform insertions into each submap
    uint64_t numElementsInserted = 0;
    uint64_t numElementsRemaining = numKeys;
    for(auto i = 0; i < m_num_submaps; ++i) {
      uint64_t maxNumElements = m_submap_caps[i] * m_capacity * m_max_load_factor;
      uint64_t numElementsToInsert = std::min(maxNumElements - m_num_elements[i], numElementsRemaining);

      if(numElementsToInsert < minKernelSize) {
        continue;
      }
      
      // allocate space for result for number of keys successfully inserted
      unsigned long long int h_totalNumSuccesses;
      unsigned long long int *d_totalNumSuccesses;
      cudaMalloc((void**)&d_totalNumSuccesses, sizeof(unsigned long long int));
      cudaMemset(d_totalNumSuccesses, 0x00, sizeof(unsigned long long int));
      
      // perform insertions into the current submap
      constexpr uint32_t blockSize = 128;
      constexpr uint32_t tileSize = 1;
      uint32_t numBlocks = (tileSize * numElementsToInsert + blockSize - 1) / blockSize;
      auto view = *this;
      insertKeySet<key_type, mapped_type, value_type>
      <<<numBlocks, blockSize>>>
      (&d_keys[numElementsInserted], &d_values[numElementsInserted], 
       d_totalNumSuccesses, numElementsToInsert, i, view);
      
      // read the number of successful insertions
      cudaMemcpy(&h_totalNumSuccesses, d_totalNumSuccesses, sizeof(unsigned long long int), 
                 cudaMemcpyDeviceToHost);
      
      numElementsRemaining -= numElementsToInsert;
      numElementsInserted += numElementsToInsert;
      m_num_elements[i] += h_totalNumSuccesses;
      m_total_num_elements += h_totalNumSuccesses;
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return temp_time;
  }


  __host__ float bulkSearch(std::vector<key_type> const& h_keys,
                            std::vector<mapped_type> &h_results,
                            uint64_t numKeys) {
    thrust::device_vector<key_type> d_keys( h_keys );
    thrust::device_vector<mapped_type> d_results( h_results );
    
    float temp_time = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    constexpr uint32_t blockSize = 128;
    constexpr uint32_t tileSize = 1;
    uint32_t numBlocks = (tileSize * numKeys + blockSize - 1) / blockSize;
    auto view = *this;
    searchKeySet<>
    <<<numBlocks, blockSize>>>
    (&d_keys[0], &d_results[0], numKeys, view);
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // copy search results back to host
    thrust::copy(d_results.begin(), d_results.end(), h_results.begin());

    return temp_time;
  }
  /**---------------------------------------------------------------------------*
   * @brief Attempts to insert a key, value pair into the map.
   *
   * Returns an iterator, boolean pair.
   *
   * If the new key already present in the map, the iterator points to
   * the location of the existing key and the boolean is `false` indicating
   * that the insert did not succeed.
   *
   * If the new key was not present, the iterator points to the location
   *where the insert occured and the boolean is `true` indicating that the
   *insert succeeded.
   *
   * @param insert_pair The key and value pair to insert
   * @return Iterator, Boolean pair. Iterator is to the location of the
   *newly inserted pair, or the existing pair that prevented the insert.
   *Boolean indicates insert success.
   *---------------------------------------------------------------------------**/
  __device__ thrust::pair<iterator, bool> insert(
      uint32_t submapIdx,
      value_type const& insert_pair) {
    value_type* submap = m_submaps[submapIdx];
    uint32_t submap_cap = m_submap_caps[submapIdx] * m_capacity;
    const size_type key_hash{m_hf(insert_pair.first)};
    size_type index{key_hash % submap_cap};

    insert_result status{insert_result::CONTINUE};

    value_type* current_bucket{nullptr};

    while (status == insert_result::CONTINUE) {
      current_bucket = submap + index;
      status = attempt_insert(current_bucket, insert_pair);
      index = (index + 1) % submap_cap;
    }

    bool const insert_success =
        (status == insert_result::SUCCESS) ? true : false;

    return thrust::make_pair(
        iterator(submap, submap + submap_cap,
                 current_bucket),
        insert_success);
  }

  

  template<typename CG>
  __device__ thrust::pair<iterator, bool> insertCG(CG g, value_type const& insert_pair,
                                                   uint32_t submapIdx) {
    key_type const k = insert_pair.first;
    mapped_type const v = insert_pair.second;
    size_type const keyHash = m_hf(k);
    value_type* submap = m_submaps[submapIdx];
    size_type submap_cap = m_capacity * m_submap_caps[submapIdx];
    uint32_t windowIdx = 0;
    
    while(true) {
      size_type index = (keyHash + windowIdx * g.size() + g.thread_rank()) % submap_cap;
      value_type* current_bucket = &submap[index];
      key_type const existing_key = current_bucket->first;
      uint32_t existing = g.ballot(m_equal(existing_key, k));
      
      // the key we are trying to insert is already in the map, so we return
      // with failure to insert
      if(existing) {
        return thrust::make_pair(iterator(submap, submap + submap_cap, current_bucket), 
                                 false);
      }
      
      uint32_t empty = g.ballot(m_equal(existing_key, m_unused_key));

      // we found an empty slot, but not the key we are inserting, so this must
      // be an empty slot into which we can insert the key
      if(empty) {
        // the first lane in the group with an empty slot will attempt the insert
        insert_result status{insert_result::CONTINUE};
        uint32_t srcLane = __ffs(empty) - 1;

        if(g.thread_rank() == srcLane) {
          key_type const old_key{
            atomicCAS(&(current_bucket->first), m_unused_key, k)};

          // the key was successfully inserted, so insert the value and
          // report success
          if(m_equal(old_key, m_unused_key)) {
            current_bucket->second = v;
            status = insert_result::SUCCESS;
          }          
          // the key was inserted by another group in the time between the 
          // first ballot and the atomicCAS
          else if(m_equal(old_key, k)) {
            status = insert_result::DUPLICATE;
          }
          // another key was inserted in the slot we wanted to try
          // so we need to try the next empty slot in the window
          else {
            status = insert_result::CONTINUE;
          }
        }

        uint32_t res_status = g.shfl(static_cast<uint32_t>(status), srcLane);
        status = static_cast<insert_result>(res_status);

        // successful insert
        if(status == insert_result::SUCCESS) {
          intptr_t res_bucket = g.shfl(reinterpret_cast<intptr_t>(current_bucket), srcLane);
          return thrust::make_pair(iterator(submap, submap + submap_cap, 
                                            reinterpret_cast<value_type*>(res_bucket)), 
                                   true);
        }
        // duplicate present during insert
        if(status == insert_result::DUPLICATE) {
          return thrust::make_pair(iterator(submap, submap + submap_cap, current_bucket),
                                   false);
        }
        // if we've gotten this far, a different key took our spot 
        // before we could insert. We need to retry the insert on the
        // same window
      }
      // if there are no empty slots in the current window,
      // we move onto the next window
      else {
        windowIdx++;
      }
    }
  }



  __device__ bool DupCheck(key_type const& k, uint32_t submapIdx) const {
    size_type const key_hash = m_hf(k);
    size_type index{0};
    value_type* submap{nullptr};
    value_type* current_bucket{nullptr};
    uint32_t submap_cap = 0;

    for(auto i = 0; i < submapIdx; ++i) {
      submap_cap = m_submap_caps[i] * m_capacity;
      index = key_hash % submap_cap;
      submap = m_submaps[i];
      current_bucket = &submap[index];
      
      while (true) {
        key_type const existing_key = current_bucket->first;

        if (m_equal(k, existing_key)) {
          return true;
        }
        if (m_equal(m_unused_key, existing_key)) {
          break;
        }
        index = (index + 1) % submap_cap;
        current_bucket = &submap[index];
      }
    }

    return false;
  }

  template<typename CG>
  __device__ bool dupCheckCG(CG g, key_type const& k, uint32_t maxSubmapIdx) {
    size_type const keyHash = m_hf(k);

    for(auto submapIdx = 0; submapIdx < maxSubmapIdx; ++submapIdx) {
      value_type* submap = m_submaps[submapIdx];
      size_type submap_cap = m_capacity * m_submap_caps[submapIdx];
      uint32_t windowIdx = 0;
      while(true) {
        size_type index = (keyHash + windowIdx * g.size() + g.thread_rank()) % submap_cap;
        value_type* current_bucket = &submap[index];
        key_type const existing_key = current_bucket->first;
        uint32_t existing = g.ballot(m_equal(existing_key, k));
        
        // the key we were searching for was found by one of the threads,
        // so we know the key is a duplicate
        if(existing) {
          return true;
        }
        // we found an empty slot, meaning that the key we're searching 
        // for isn't in this submap, so we should move onto the next one
        uint32_t empty = g.ballot(m_equal(existing_key, m_unused_key));
        if(empty) {
          break;
        }

        // otherwise, all slots in the current window are full with other keys,
        // so we move onto the next window in the current submap
        windowIdx++;
      }
    }

    // the key was not found in any of the submaps
    return false;
  }

  /**---------------------------------------------------------------------------*
   * @brief Searches the map for the specified key.
   *
   * @note `find` is not threadsafe with `insert`. I.e., it is not safe to
   *do concurrent `insert` and `find` operations.
   *
   * @param k The key to search for
   * @return An iterator to the key if it exists, else map.end()
   *---------------------------------------------------------------------------**/
  __device__ const_iterator find(key_type const& k) const {
    
    #if 1
    size_type const key_hash = m_hf(k);
    size_type index{0};
    value_type* submap{nullptr};
    value_type* current_bucket{nullptr}; 

    for(auto i = 0; i < m_num_submaps; ++i) {
      size_type submap_cap = m_capacity * m_submap_caps[i];

      index = key_hash % submap_cap;
      submap = m_submaps[i];
      current_bucket = &submap[index];

      while (true) {
        key_type const existing_key = current_bucket->first;

        if (m_equal(k, existing_key)) {
          return const_iterator(submap, submap + submap_cap,
                                current_bucket);
        }
        if (m_equal(m_unused_key, existing_key)) {
          break;
        }
        index = (index + 1) % submap_cap;
        current_bucket = &submap[index];
      }
    }

    return this->end();
    #endif

    #if 0
    size_type const key_hash = m_hf(k);
    size_type probeIdx = 0;
    bool done0 = false;
    bool done1 = false;
    bool done2 = false;
    uint32_t numDone = 0;
    value_type* submap;
    value_type* current_bucket;
    
    constexpr uint32_t gran = 1;
    
    while (true) {
      for(auto i = 0; i < gran * m_num_submaps; ++i) {
        // change submapIdx every gran probes
        uint32_t submapIdx = i / gran;
        uint32_t submap_cap = m_capacity * m_submap_caps[submapIdx];
        uint32_t localIdx = (key_hash + probeIdx + i % gran) % submap_cap;
        // if we already know it isn't in this map, just continue on to the next one
        
        if((submapIdx == 0 && done0) || (submapIdx == 1 && done1) || (submapIdx == 2 && done2)) {
          i = (submapIdx + 1) * gran - 1;
          continue;
        }
        
        submap = m_submaps[submapIdx];
        current_bucket = &submap[localIdx];
        key_type const existing_key = current_bucket->first;

        if (m_equal(k, existing_key)) {
          return const_iterator(submap, submap + submap_cap,
                                current_bucket);
        }
        if (m_equal(m_unused_key, existing_key)) {
          switch(submapIdx) {
            case 0:
              done0 = true;
              break;
            case 1:
              done1 = true;
              break;
            case 2:
              done2 = true;
              break;
          }
          numDone++;
        }
      }

      if(numDone >= m_num_submaps) {
        return this->end();
      }

      probeIdx += gran;
    }

    return this->end();
    #endif
  }


  template<typename CG>
  __device__ const_iterator findCG(CG g, key_type const& k) {
    size_type const keyHash = m_hf(k);
    
    for(auto submapIdx = 0; submapIdx < m_num_submaps; ++submapIdx) {
      value_type* submap = m_submaps[submapIdx];
      size_type submap_cap = m_capacity * m_submap_caps[submapIdx];
      uint32_t windowIdx = 0;
      while(true) {
        size_type index = (keyHash + windowIdx * g.size() + g.thread_rank()) % submap_cap;
        value_type* current_bucket = &submap[index];
        key_type const existing_key = current_bucket->first;
        uint32_t existing = g.ballot(m_equal(existing_key, k));
        
        // the key we were searching for was found by one of the threads,
        // so we return an iterator to the entry
        if(existing) {
          uint32_t srcLane = __ffs(existing) - 1;
          intptr_t res_bucket = g.shfl(reinterpret_cast<intptr_t>(current_bucket), srcLane);
          return const_iterator(submap, submap + submap_cap,
                                reinterpret_cast<value_type*>(res_bucket));
        }
        
        // we found an empty slot, meaning that the key we're searching 
        // for isn't in this submap, so we should move onto the next one
        uint32_t empty = g.ballot(m_equal(existing_key, m_unused_key));
        if(empty) {
          break;
        }

        // otherwise, all slots in the current window are full with other keys,
        // so we move onto the next window in the current submap
        windowIdx++;
      }
    }

    // the key was not found in any of the submaps
    return this->end();
  }
  
  template<typename CG>
  __device__ const_iterator findCGAlt(uint32_t alt_gran, CG g, key_type const& k) {
    size_type const keyHash = m_hf(k);
    bool done0 = false;
    bool done1 = false;
    bool done2 = false;
    value_type *submap0 = m_submaps[0];
    value_type *submap1 = m_submaps[1];
    value_type *submap2 = m_submaps[2];
    size_type submap_cap0 = m_submap_caps[0];
    size_type submap_cap1 = m_submap_caps[1];
    size_type submap_cap2 = m_submap_caps[2];
    uint32_t numDone = 0;
    value_type* submap;
    size_type submap_cap;
    uint32_t submapIdx = 0;
    uint32_t windowIdx = 0;
    uint32_t probeIdx = 0;

    while(true) {
      switch(submapIdx) {
        case 0: submap = submap0; submap_cap = m_capacity * submap_cap0; break;
        case 1: submap = submap1; submap_cap = m_capacity * submap_cap1; break;
        case 2: submap = submap2; submap_cap = m_capacity * submap_cap2; break;
      }

      size_type index = (keyHash + windowIdx * g.size() + g.thread_rank()) % submap_cap;
      value_type* current_bucket = &submap[index];
      key_type const existing_key = current_bucket->first;
      uint32_t existing = g.ballot(m_equal(existing_key, k));
      
      // the key we were searching for was found by one of the threads,
      // so we return an iterator to the entry
      if(existing) {
        uint32_t srcLane = __ffs(existing) - 1;
        intptr_t res_bucket = g.shfl(reinterpret_cast<intptr_t>(current_bucket), srcLane);
        return const_iterator(submap, submap + submap_cap,
                              reinterpret_cast<value_type*>(res_bucket));
      }
      
      // we found an empty slot, meaning that the key we're searching 
      // for isn't in this submap, so we should move onto the next one
      uint32_t empty = g.ballot(m_equal(existing_key, m_unused_key));
      if(empty) {
        switch(submapIdx) {
          case 0: done0 = true; break;
          case 1: done1 = true; break;
          case 2: done2 = true; break;
        }
      }

      // otherwise, all slots in the current window are full with other keys,
      // so we move onto the next window in the current submap
      bool skipped = false;
      while((submapIdx == 0 && done0) || (submapIdx == 1 && done1) || (submapIdx == 2 && done2)) {
        // ensure that we move onto the next submap given that we know the key cannot be in one
        // of these submaps
        probeIdx += alt_gran - (probeIdx % alt_gran);
        submapIdx = get_submap_idx_alt(probeIdx, alt_gran);
        skipped = true;
      }
      if(!skipped) {
        probeIdx++;
      }
      submapIdx = get_submap_idx_alt(probeIdx, alt_gran);
      windowIdx = get_window_idx_alt(probeIdx, alt_gran);
    }

    // the key was not found in any of the submaps
    return this->end();
  }

  __device__ uint32_t get_submap_idx_alt(uint32_t probeIdx, uint32_t alt_gran) {
    return (probeIdx / alt_gran) % m_num_submaps;
  }

  __device__ uint32_t get_window_idx_alt(uint32_t probeIdx, uint32_t alt_gran) {
    return ((probeIdx / alt_gran) / m_num_submaps) + probeIdx % alt_gran;
  }

  cc_error assign_async(const concurrent_unordered_map_chain& other,
                        cudaStream_t stream = 0) {
    if (other.m_capacity <= m_capacity) {
      m_capacity = other.m_capacity;
    } else {
      m_allocator.deallocate(m_hashtbl_values, m_capacity);
      m_capacity = other.m_capacity;
      m_capacity = other.m_capacity;

      m_hashtbl_values = m_allocator.allocate(m_capacity);
    }
    CUDA_TRY(cudaMemcpyAsync(m_hashtbl_values, other.m_hashtbl_values,
                             m_capacity * sizeof(value_type), cudaMemcpyDefault,
                             stream));
    return CC_SUCCESS;
  }

  void clear_async(cudaStream_t stream = 0) {
    constexpr int block_size = 128;
    init_hashtbl<<<((m_capacity - 1) / block_size) + 1, block_size, 0,
                   stream>>>(m_hashtbl_values, m_capacity, m_unused_key,
                             m_unused_element);
  }

  void print() {
    for (size_type i = 0; i < m_capacity; ++i) {
      std::cout << i << ": " << m_hashtbl_values[i].first << ","
                << m_hashtbl_values[i].second << std::endl;
    }
  }

  /**---------------------------------------------------------------------------*
   * @brief Frees the contents of the map and destroys the map object.
   *
   * This function is invoked as the deleter of the `std::unique_ptr` returned
   * from the `create()` factory function.
   *---------------------------------------------------------------------------**/
  void destroy() {
    /*
    for(auto i = 0; i < m_num_submaps; ++i) {
      m_allocator.deallocate(m_submaps[i], m_capacity);
    }
    */
    delete this;
  }


  concurrent_unordered_map_chain() = delete;
  concurrent_unordered_map_chain(concurrent_unordered_map_chain const&) = default;
  concurrent_unordered_map_chain(concurrent_unordered_map_chain&&) = default;
  concurrent_unordered_map_chain& operator=(concurrent_unordered_map_chain const&) =
      default;
  concurrent_unordered_map_chain& operator=(concurrent_unordered_map_chain&&) = default;
  ~concurrent_unordered_map_chain() = default;

 private:
  hasher m_hf;
  key_equal m_equal;
  mapped_type m_unused_element;
  key_type m_unused_key;
  allocator_type m_allocator;
  size_type m_capacity;
  value_type* m_hashtbl_values;
  value_type* m_submaps[m_max_num_submaps];
  const uint32_t m_submap_caps[m_max_num_submaps] = {1, 1, 2, 4, 5, 16, 32, 64};
  uint32_t m_num_elements[m_max_num_submaps];
  uint32_t m_total_num_elements;
  uint32_t m_num_submaps;

  /**---------------------------------------------------------------------------*
   * @brief Private constructor used by `create` factory function.
   *
   * @param capacity The desired m_capacity of the hash table
   * @param unused_element The sentinel value to use for an empty value
   * @param unused_key The sentinel value to use for an empty key
   * @param hash_function The hash function to use for hashing keys
   * @param equal The equality comparison function for comparing if two keys
   *are equal
   * @param allocator The allocator to use for allocation the hash table's
   * storage
   *---------------------------------------------------------------------------**/
  concurrent_unordered_map_chain(size_type capacity, const mapped_type unused_element,
                           const key_type unused_key,
                           const Hasher& hash_function, const Equality& equal,
                           const allocator_type& allocator)
      : m_hf(hash_function),
        m_equal(equal),
        m_allocator(allocator),
        m_unused_element(unused_element),
        m_unused_key(unused_key),
        m_total_num_elements(0),
        m_num_submaps(1) {
    // round m_capacity to the nearest larger or equal power of 2
    m_capacity = capacity;//std::pow(2, ceil(log(capacity) / log(2)));

    m_hashtbl_values = m_allocator.allocate(m_capacity);
    constexpr int block_size = 128;
    
    init_hashtbl<<<((m_capacity - 1) / block_size) + 1, block_size>>>(
        m_hashtbl_values, m_capacity, m_unused_key, m_unused_element);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(0));

    m_submaps[0] = m_hashtbl_values;
    m_num_elements[0] = 0;
  }
};

