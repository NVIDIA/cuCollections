/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cuco/detail/__config>
#include <cuco/detail/prime.hpp>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/sentinel.cuh>
#include <cuco/static_set_ref.cuh>
#include <cuco/storage.cuh>
#include <cuco/utility/allocator.hpp>
#include <cuco/utility/traits.hpp>

#include <thrust/functional.h>

#include <cuda/atomic>

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
 * @tparam Key Type used for keys. Requires `cuco::is_bitwise_comparable_v<Key>`
 * @tparam Scope The scope in which set operations will be performed by individual threads
 * @tparam KeyEqual Binary callable type used to compare two keys for equality
 * @tparam ProbingScheme Probing scheme chosen between `cuco::linear_probing`
 * and `cuco::double_hashing`. (see `probing_scheme.cuh`)
 * @tparam Allocator Type of allocator used for device storage
 * @tparam Storage Slot window storage type
 */

template <class Key,
          class Extent             = cuco::experimental::extent<std::size_t>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class KeyEqual           = thrust::equal_to<Key>,
          class ProbingScheme      = experimental::double_hashing<1,  // CG size
                                                             cuco::murmurhash3_32<Key>,
                                                             cuco::murmurhash3_32<Key>>,
          class Allocator          = cuco::cuda_allocator<std::byte>,
          class Storage            = cuco::experimental::aow_storage<2  // Window size
                                                          >>
class static_set {
  static_assert(sizeof(Key) <= 8, "Container does not support key types larger than 8 bytes.");

  static_assert(
    cuco::is_bitwise_comparable_v<Key>,
    "Key type must have unique object representations or have been explicitly declared as safe for "
    "bitwise comparison via specialization of cuco::is_bitwise_comparable_v<Key>.");

  static_assert(
    std::is_base_of_v<cuco::experimental::detail::probing_scheme_base<ProbingScheme::cg_size>,
                      ProbingScheme>,
    "ProbingScheme must be a specialization of either cuco::double_hashing or "
    "cuco::linear_probing.");

 public:
  static constexpr auto cg_size      = ProbingScheme::cg_size;  ///< CG size used to for probing
  static constexpr auto window_size  = Storage::window_size;    ///< Window size used to for probing
  static constexpr auto thread_scope = Scope;                   ///< CUDA thread scope

  using key_type   = Key;  ///< Key type
  using value_type = Key;  ///< Key type
  /// Extent type
  using extent_type    = decltype(make_valid_extent<cg_size, window_size>(std::declval<Extent>()));
  using size_type      = typename extent_type::value_type;  ///< Size type
  using key_equal      = KeyEqual;                          ///< Key equality comparator type
  using allocator_type = Allocator;                         ///< Allocator type
  using storage_type =
    detail::storage<Storage, value_type, extent_type, allocator_type>;  ///< Storage type

  using storage_ref_type    = typename storage_type::ref_type;  ///< Window storage reference type
  using probing_scheme_type = ProbingScheme;                    ///< Probe scheme type
  template <typename... Operators>
  using ref_type = cuco::experimental::static_set_ref<key_type,
                                                      thread_scope,
                                                      key_equal,
                                                      probing_scheme_type,
                                                      storage_ref_type,
                                                      Operators...>;  ///< Container reference type

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
   * @brief Constructs a statically-sized set with the specified initial capacity, sentinel values
   * and CUDA stream.
   *
   * The actual set capacity depends on the given `capacity`, the probing scheme, CG size, and the
   * window size and it's computed via `make_valid_extent` factory. Insert operations will not
   * automatically grow the set. Attempting to insert more unique keys than the capacity of the map
   * results in undefined behavior.
   *
   * The `empty_key_sentinel` is reserved and behavior is undefined when attempting to insert
   * this sentinel value.
   *
   * @param capacity The lower-bound set size requested by the user
   * @param empty_key_sentinel The reserved key value for empty slots
   * @param pred Key equality binary predicate
   * @param probing_scheme Probing scheme
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the map
   */
  constexpr static_set(Extent capacity,
                       empty_key<Key> empty_key_sentinel,
                       KeyEqual pred                       = {},
                       ProbingScheme const& probing_scheme = {},
                       Allocator const& alloc              = {},
                       cudaStream_t stream                 = nullptr);

  /**
   * @brief Inserts all keys in the range `[first, last)` and returns the number of successful
   * insertions.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `insert_async`.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * static_set<K>::value_type></tt> is `true`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stream CUDA stream used for insert
   *
   * @return Number of successfully inserted keys
   */
  template <typename InputIt>
  size_type insert(InputIt first, InputIt last, cudaStream_t stream = nullptr);

  /**
   * @brief Inserts all keys in the range `[first, last)` asynchronously.
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
  void insert_async(InputIt first, InputIt last, cudaStream_t stream = nullptr);

  /**
   * @brief Indicates whether the keys in the range `[first, last)` are contained in the set.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `contains_async`.
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
  void contains(InputIt first,
                InputIt last,
                OutputIt output_begin,
                cudaStream_t stream = nullptr) const;

  /**
   * @brief Indicates whether the keys in the range `[first, last)` are contained in the set.
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
  void contains_async(InputIt first,
                      InputIt last,
                      OutputIt output_begin,
                      cudaStream_t stream = nullptr) const;

  /**
   * @brief Gets the number of elements in the container.
   *
   * @param stream CUDA stream used to get the number of inserted elements
   * @return The number of elements in the container
   */
  [[nodiscard]] size_type size(cudaStream_t stream = nullptr) const;

  /**
   * @brief Gets the maximum number of elements the hash map can hold.
   *
   * @return The maximum number of elements the hash map can hold
   */
  [[nodiscard]] constexpr auto capacity() const noexcept;

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] constexpr key_type empty_key_sentinel() const noexcept;

  /**
   * @brief Get device ref with operators.
   *
   * @tparam Operators Set of `cuco::op` to be provided by the ref
   *
   * @param ops List of operators, e.g., `cuco::insert`
   *
   * @return Device ref of the current `static_set` object
   */
  template <typename... Operators>
  [[nodiscard]] auto ref(Operators... ops) const noexcept;

 private:
  key_type empty_key_sentinel_;         ///< Key value that represents an empty slot
  key_equal predicate_;                 ///< Key equality binary predicate
  probing_scheme_type probing_scheme_;  ///< Probing scheme
  allocator_type allocator_;            ///< Allocator used to (de)allocate temporary storage
  storage_type storage_;                ///< Slot window storage
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/static_set/static_set.inl>
