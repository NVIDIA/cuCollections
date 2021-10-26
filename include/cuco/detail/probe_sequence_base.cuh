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

#include <cuda/std/atomic>

#include <cuco/detail/hash_functions.cuh>
#include <cuco/detail/pair.cuh>

namespace cuco {
namespace detail {

/**
 * @brief Base class for a hash map probe sequence. This class should not be used directly.
 *
 * Hash map operations are generally memory-bandwidth bound. A vector-load loads two consecutive
 * slots instead of one to fully utilize the 16B memory load supported by SASS/hardware thus
 * improve memory throughput. This method (flagged by `uses_vector_load` logic) is implicitly
 * applied to all hash map operations (e.g. `insert`, `count`, and `retrieve`, etc.) when pairs
 * are packable (see `cuco::detail::is_packable` logic).
 *
 * @tparam Key Type used for keys
 * @tparam Value Type of the mapped values
 * @tparam CGSize Number of threads in CUDA Cooperative Groups
 * @tparam Scope The scope in which multimap operations will be performed by
 * individual threads
 */
template <typename Key, typename Value, uint32_t CGSize, cuda::thread_scope Scope>
class probe_sequence_base {
 protected:
  using value_type         = cuco::pair_type<Key, Value>;
  using key_type           = Key;
  using mapped_type        = Value;
  using atomic_key_type    = cuda::atomic<key_type, Scope>;
  using atomic_mapped_type = cuda::atomic<mapped_type, Scope>;
  using pair_atomic_type   = cuco::pair_type<atomic_key_type, atomic_mapped_type>;
  using iterator           = pair_atomic_type*;
  using const_iterator     = pair_atomic_type const*;

 public:
  /**
   * @brief Returns the size of the CUDA cooperative thread group.
   */
  __host__ __device__ static constexpr uint32_t cg_size() noexcept { return CGSize; }

  /**
   * @brief Returns the number of elements loaded with each vector-load.
   */
  __host__ __device__ static constexpr uint32_t vector_width() noexcept { return 2u; }

  /**
   * @brief Indicates if vector-load is used.
   *
   * Users have no explicit control on whether vector-load is used.
   *
   * @return Boolean indicating if vector-load is used.
   */
  __host__ __device__ static constexpr bool uses_vector_load() noexcept
  {
    return cuco::detail::is_packable<value_type>();
  }

  /**
   * @brief Constructs a probe sequence based on the given hash map features.
   *
   * @param slots Pointer to beginning of the hash map slots
   * @param capacity Capacity of the hash map
   */
  __host__ __device__ explicit probe_sequence_base(iterator slots, std::size_t capacity)
    : slots_{slots}, capacity_{capacity}
  {
  }

  /**
   * @brief Returns the capacity of the hash map.
   */
  __host__ __device__ __forceinline__ std::size_t get_capacity() const noexcept
  {
    return capacity_;
  }

  /**
   * @brief Returns slots array.
   */
  __device__ __forceinline__ iterator get_slots() noexcept { return slots_; }

  /**
   * @brief Returns slots array.
   */
  __device__ __forceinline__ const_iterator get_slots() const noexcept { return slots_; }

 protected:
  iterator slots_;              ///< Pointer to beginning of the hash map slots
  const std::size_t capacity_;  ///< Total number of slots
};                              // class probe_sequence_base

}  // namespace detail
}  // namespace cuco
