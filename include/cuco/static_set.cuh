/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuco/allocator.hpp>
#include <cuco/detail/hash_functions.cuh>
#include <cuco/detail/prime.hpp>
#include <cuco/detail/storage.cuh>
#include <cuco/extent.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/sentinel.cuh>
#include <cuco/traits.hpp>

#include <thrust/functional.h>

#include <cuda/atomic>
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11000) && defined(__CUDA_ARCH__) && \
  (__CUDA_ARCH__ >= 700)
#define CUCO_HAS_CUDA_BARRIER
#endif

// cg::memcpy_aysnc is supported for CUDA 11.1 and up
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11100)
#define CUCO_HAS_CG_MEMCPY_ASYNC
#endif

#if defined(CUCO_HAS_CUDA_BARRIER)
#include <cuda/barrier>
#endif

#include <cstddef>
#include <type_traits>

namespace cuco {
namespace experimental {
/**
 * @brief A GPU-accelerated, unordered, associative container of unique keys.
 *
 * Allows constant time concurrent inserts or concurrent find operations from threads in device
 * code. Concurrent insert/find is allowed only when
 * <tt>static_set<Key>::supports_concurrent_insert_find()</tt> is true.
 *
 * @tparam Key Type used for keys. Requires `cuco::is_bitwise_comparable_v<Key>`
 * @tparam Scope The scope in which set operations will be performed by individual threads
 * @tparam KeyEqual Binary callable type used to compare two keys for equality
 * @tparam ProbingScheme Probing scheme chosen between `cuco::linear_probing`
 * and `cuco::double_hashing`. (see `detail/probe_sequences.cuh`)
 * @tparam Allocator Type of allocator used for device storage
 * @tparam Storage Slot storage type
 */
// template<
// class Key,
// class Hash = std::hash<Key>,
// class KeyEqual = std::equal_to<Key>,
// class Allocator = std::allocator<Key>
// > class unordered_set;
template <class Key,
          class Extent             = cuco::experimental::extent<std::size_t>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class KeyEqual           = thrust::equal_to<Key>,
          class ProbingScheme =
            experimental::double_hashing<1,                           // CG size
                                         2,                           // Window size (vector length)
                                         enable_window_probing::YES,  // uses window probing
                                         cuco::detail::MurmurHash3_32<Key>,  // Hash1
                                         cuco::detail::MurmurHash3_32<Key>   // Hash2
                                         >,
          class Allocator = cuco::cuda_allocator<char>,
          class Storage   = cuco::experimental::detail::aos_storage<Key, Extent, Allocator>>
class static_set {
  static_assert(
    cuco::is_bitwise_comparable_v<Key>,
    "Key type must have unique object representations or have been explicitly declared as safe for "
    "bitwise comparison via specialization of cuco::is_bitwise_comparable_v<Key>.");

  static_assert(
    std::is_base_of_v<
      cuco::experimental::detail::probing_scheme_base<ProbingScheme::cg_size,
                                                      ProbingScheme::window_size,
                                                      ProbingScheme::uses_window_probing>,
      ProbingScheme>,
    "ProbingScheme must be a specialization of either cuco::double_hashing or "
    "cuco::linear_probing.");

 public:
  using key_type            = Key;                               ///< Key type
  using value_type          = Key;                               ///< Key type
  using extent_type         = Extent;                            ///< Extent type
  using size_type           = typename extent_type::value_type;  ///< Size type
  using key_equal           = KeyEqual;                          ///< Key equality comparator type
  using allocator_type      = Allocator;                         ///< Allocator type
  using slot_storage_type   = Storage;                           ///< Slot storage type
  using slot_view_type      = typename slot_storage_type::view_type;  ///< Slot view type
  using probing_scheme_type = ProbingScheme;                          ///< Probe scheme type
  using counter_storage_type =
    detail::counter_storage<size_type, Scope, Allocator>;  ///< Counter storage type

  static constexpr int cg_size = ProbingScheme::cg_size;  ///< CG size used to for probing

  static_set(static_set const&) = delete;
  static_set& operator=(static_set const&) = delete;

  static_set(static_set&&) = default;  ///< Move constructor

  /**
   * @brief Replaces the contents of the map with another map.
   *
   * @return Reference of the current map object
   */
  static_set& operator=(static_set&&) = default;
  ~static_set()                       = default;

  /**
   * @brief Construct a statically-sized set with the specified initial capacity,
   * sentinel values and CUDA stream.
   *
   * The capacity of the set is fixed. Insert operations will not automatically
   * grow the set. Attempting to insert more unique keys than the capacity of
   * the map results in undefined behavior.
   *
   * The `empty_key_sentinel` is reserved and behavior is undefined when attempting to insert
   * this sentinel value.
   *
   * @param capacity The total number of slots in the set
   * @param empty_key_sentinel The reserved key value for empty slots
   * @param pred Key equality binary predicate
   * @param probing_scheme Probing scheme
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the map
   */
  static_set(Extent capacity,
             sentinel::empty_key<Key> empty_key_sentinel,
             KeyEqual pred                = KeyEqual{},
             ProbingScheme probing_scheme = ProbingScheme{cuco::detail::MurmurHash3_32<Key>{},
                                                          cuco::detail::MurmurHash3_32<Key>{}},
             Allocator const& alloc       = Allocator{},
             cudaStream_t stream          = 0);

  /**
   * @brief Inserts all keys in the range `[first, last)`.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * static_set<K>::value_type></tt> is `true`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stream CUDA stream used for insert
   */
  template <typename InputIt>
  void insert(InputIt first, InputIt last, cudaStream_t stream = 0);

  /**
   * @brief Indicates whether the keys in the range `[first, last)` are contained in the set.
   *
   * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
   *
   * @tparam InputIt Device accessible input iterator
   * @tparam OutputIt Device accessible output iterator assignable from `bool`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param stream Stream used for executing the kernels
   */
  template <typename InputIt, typename OutputIt>
  void contains(InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream = 0) const;

  /**
   * @brief Gets the maximum number of elements the hash map can hold.
   *
   * @return The maximum number of elements the hash map can hold
   */
  size_type capacity() const noexcept { return slot_storage_.capacity(); }

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  key_type empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

  /**
   * @brief Get device reference.
   *
   * @return Device reference of the current static_set
   */
  auto reference() const noexcept;

 private:
  size_type size_;                  ///< Number of entries
  key_type empty_key_sentinel_;     ///< Key value that represents an empty slot
  key_equal predicate_;             ///< Key equality binary predicate
  ProbingScheme probing_scheme_;    ///< Probing scheme
  allocator_type allocator_;        ///< Allocator used to (de)allocate temporary storage
  counter_storage_type counter_;    ///< Device counter storage
  slot_storage_type slot_storage_;  ///< Flat slot storage
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/static_set/static_set.inl>
