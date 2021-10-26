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

#include <cuco/detail/probe_sequence_base.cuh>

namespace cuco {

/**
 * @brief Cooperative Groups based Linear probing scheme.
 *
 * Linear probing is efficient only when few collisions are present. Performance hints:
 * - Use linear probing only when collisions are rare. e.g. low occupancy or low multiplicity.
 * - `CGSize` = 1 or 2 when hash map is small (10'000'000 or less), 4 or 8 otherwise.
 *
 * `Hash` should be callable object type.
 *
 * @tparam Key Type used for keys
 * @tparam Value Type of the mapped values
 * @tparam CGSize Number of threads in CUDA Cooperative Groups
 * @tparam Hash Unary callable type
 * @tparam Scope The scope in which multimap operations will be performed by
 * individual threads
 */
template <typename Key,
          typename Value,
          uint32_t CGSize          = 8,
          typename Hash            = cuco::detail::MurmurHash3_32<Key>,
          cuda::thread_scope Scope = cuda::thread_scope_device>
class linear_probing : public detail::probe_sequence_base<Key, Value, CGSize, Scope> {
 public:
  using probe_sequence_base_type = detail::probe_sequence_base<Key, Value, CGSize, Scope>;
  using iterator                 = typename probe_sequence_base_type::iterator;
  using probe_sequence_base_type::capacity_;
  using probe_sequence_base_type::cg_size;
  using probe_sequence_base_type::slots_;
  using probe_sequence_base_type::uses_vector_load;
  using probe_sequence_base_type::vector_width;

  /**
   * @brief Constructs a linear probing scheme based on the given hash map features.
   *
   * @param slots Pointer to beginning of the hash map slots
   * @param capacity Capacity of the hash map
   * @param hash Unary function to hash each key
   */
  __host__ __device__ explicit linear_probing(iterator slots,
                                              std::size_t capacity,
                                              Hash hash = Hash{})
    : detail::probe_sequence_base<Key, Value, CGSize, Scope>{slots, capacity}, hash_{hash}
  {
  }

  /**
   * @brief Returns the initial slot for a given key `k`.
   *
   * If vector-load is enabled, the return slot is always even to avoid illegal memory access.
   *
   * @tparam CG CUDA Cooperative Groups type
   * @param g the Cooperative Group for which the initial slot is needed
   * @param k The key to get the slot for
   * @return Pointer to the initial slot for `k`
   */
  template <typename CG>
  __device__ __forceinline__ iterator initial_slot(CG const& g, Key const k) noexcept
  {
    auto const hash_value = [&]() {
      auto const tmp = hash_(k);
      if constexpr (uses_vector_load()) {
        // ensure initial hash value is always even
        return tmp + tmp % 2;
      }
      if constexpr (not uses_vector_load()) { return tmp; }
    }();

    auto const offset = [&]() {
      if constexpr (uses_vector_load()) { return g.thread_rank() * vector_width(); }
      if constexpr (not uses_vector_load()) { return g.thread_rank(); }
    }();

    // Each CG accesses to a window of (`cg_size` * `vector_width`)
    // slots if vector-load is used or `cg_size` slots otherwise
    return &slots_[(hash_value + offset) % capacity_];
  }

  /**
   * @brief Given a slot `s`, returns the next slot.
   *
   * If `s` is the last slot, wraps back around to the first slot.
   *
   * @param s The slot to advance
   * @return The next slot after `s`
   */
  __device__ __forceinline__ iterator next_slot(iterator s) noexcept
  {
    std::size_t index = s - slots_;
    std::size_t offset;
    if constexpr (uses_vector_load()) {
      offset = cg_size() * vector_width();
    } else {
      offset = cg_size();
    }
    return &slots_[(index + offset) % capacity_];
  }

 private:
  Hash hash_;  ///< The unary callable used to hash the key
};             // class linear_probing

/**
 * @brief Cooperative Groups based double hashing scheme.
 *
 * Default probe sequence for `cuco::static_multimap`. Double hashing shows superior
 * performance when dealing with high multiplicty and/or high occupancy use cases. Performance
 * hints:
 * - `CGSize` = 1 or 2 when hash map is small (10'000'000 or less), 4 or 8 otherwise.
 *
 * `Hash1` and `Hash2` should be callable object type.
 *
 * @tparam Key Type used for keys
 * @tparam Value Type of the mapped values
 * @tparam CGSize Number of threads in CUDA Cooperative Groups
 * @tparam Hash1 Unary callable type
 * @tparam Hash2 Unary callable type
 * @tparam Scope The scope in which multimap operations will be performed by
 * individual threads
 */
template <typename Key,
          typename Value,
          uint32_t CGSize          = 8,
          typename Hash1           = cuco::detail::MurmurHash3_32<Key>,
          typename Hash2           = cuco::detail::MurmurHash3_32<Key>,
          cuda::thread_scope Scope = cuda::thread_scope_device>
class double_hashing : public detail::probe_sequence_base<Key, Value, CGSize, Scope> {
 public:
  using probe_sequence_base_type = detail::probe_sequence_base<Key, Value, CGSize, Scope>;
  using iterator                 = typename probe_sequence_base_type::iterator;
  using probe_sequence_base_type::capacity_;
  using probe_sequence_base_type::cg_size;
  using probe_sequence_base_type::slots_;
  using probe_sequence_base_type::uses_vector_load;
  using probe_sequence_base_type::vector_width;

  /**
   * @brief Constructs a double hashing scheme based on the given hash map features.
   *
   * `hash2` takes a different seed to reduce the chance of secondary clustering.
   *
   * @param slots Pointer to beginning of the hash map slots
   * @param capacity Capacity of the hash map
   * @param hash1 First hasher to hash each key
   * @param hash2 Second hasher to determine step size
   */
  __host__ __device__ explicit double_hashing(iterator slots,
                                              std::size_t capacity,
                                              Hash1 hash1 = Hash1{},
                                              Hash2 hash2 = Hash2{1}) noexcept
    : detail::probe_sequence_base<Key, Value, CGSize, Scope>{slots, capacity},
      hash1_{hash1},
      hash2_{hash2}
  {
  }

  /**
   * @brief Returns the initial slot for a given key `k`.
   *
   * If vector-load is enabled, the return slot is always a multiple of (`cg_size` * `vector_width`)
   * to avoid illegal memory access.
   *
   * @tparam CG CUDA Cooperative Groups type
   * @param g the Cooperative Group for which the initial slot is needed
   * @param k The key to get the slot for
   * @return Pointer to the initial slot for `k`
   */
  template <typename CG>
  __device__ __forceinline__ iterator initial_slot(CG const& g, Key const k) noexcept
  {
    std::size_t index;
    auto const hash_value = hash1_(k);
    if constexpr (uses_vector_load()) {
      // step size in range [1, prime - 1] * cg_size * vector_width
      step_size_ = (hash2_(k) % (capacity_ / (cg_size() * vector_width()) - 1) + 1) * cg_size() *
                   vector_width();
      index = hash_value % (capacity_ / (cg_size() * vector_width())) * cg_size() * vector_width() +
              g.thread_rank() * vector_width();
    } else {
      // step size in range [1, prime - 1] * cg_size
      step_size_ = (hash2_(k) % (capacity_ / cg_size() - 1) + 1) * cg_size();
      index      = (hash_value + g.thread_rank()) % capacity_;
    }
    return slots_ + index;
  }

  /**
   * @brief Given a slot `s`, returns the next slot.
   *
   * If `s` is the last slot, wraps back around to the first slot.
   *
   * @param s The slot to advance
   * @return The next slot after `s`
   */
  __device__ __forceinline__ iterator next_slot(iterator s) noexcept
  {
    std::size_t index = s - slots_;
    return &slots_[(index + step_size_) % capacity_];
  }

 private:
  Hash1 hash1_;            ///< The first unary callable used to hash the key
  Hash2 hash2_;            ///< The second unary callable used to determine step size
  std::size_t step_size_;  ///< The step stride when searching for the next slot
};                         // class double_hashing

}  // namespace cuco
