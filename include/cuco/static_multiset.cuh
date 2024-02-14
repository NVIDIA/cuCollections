/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuco/cuda_stream_ref.hpp>
#include <cuco/detail/open_addressing/open_addressing_impl.cuh>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/sentinel.cuh>
#include <cuco/static_multiset_ref.cuh>
#include <cuco/storage.cuh>
#include <cuco/utility/allocator.hpp>
#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuco/utility/traits.hpp>

#include <thrust/functional.h>

#include <cuda/atomic>

#include <cstddef>
#include <memory>

namespace cuco {
/**
 * @brief A GPU-accelerated, unordered, associative container of possibly non-unique objects
 *
 * The `static_multiset` supports two types of operations:
 * - Host-side "bulk" operations
 * - Device-side "singular" operations
 *
 * The host-side bulk operations include `insert`, `contains`, etc. These APIs should be used when
 * there are a large number of keys to modify or lookup. For example, given a range of keys
 * specified by device-accessible iterators, the bulk `insert` function will insert all keys into
 * the set.
 *
 * The singular device-side operations allow individual threads (or cooperative groups) to perform
 * independent modify or lookup operations from device code. These operations are accessed through
 * non-owning, trivially copyable reference types (or "ref"). User can combine any arbitrary
 * operators (see options in `include/cuco/operator.hpp`) when creating the ref. Concurrent modify
 * and lookup will be supported if both kinds of operators are specified during the ref
 * construction.
 *
 * @note Allows constant time concurrent modify or lookup operations from threads in device code.
 * @note cuCollections data structures always place the slot keys on the left-hand side when
 * invoking the key comparison predicate, i.e., `pred(slot_key, query_key)`. Order-sensitive
 * `KeyEqual` should be used with caution.
 * @note `ProbingScheme::cg_size` indicates how many threads are used to handle one independent
 * device operation. `cg_size == 1` uses the scalar (or non-CG) code paths.
 *
 * @throw If the size of the given key type is larger than 8 bytes
 * @throw If the given key type doesn't have unique object representations, i.e.,
 * `cuco::bitwise_comparable_v<Key> == false`
 * @throw If the probing scheme type is not inherited from `cuco::detail::probing_scheme_base`
 *
 * @tparam Key Type used for keys. Requires `cuco::is_bitwise_comparable_v<Key>`
 * @tparam Extent Data structure size type
 * @tparam Scope The scope in which operations will be performed by individual threads.
 * @tparam KeyEqual Binary callable type used to compare two keys for equality
 * @tparam ProbingScheme Probing scheme (see `include/cuco/probing_scheme.cuh` for choices)
 * @tparam Allocator Type of allocator used for device storage
 * @tparam Storage Slot window storage type
 */
template <class Key,
          class Extent             = cuco::extent<std::size_t>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class KeyEqual           = thrust::equal_to<Key>,
          class ProbingScheme      = cuco::double_hashing<4,  // CG size
                                                     cuco::default_hash_function<Key>>,
          class Allocator          = cuco::cuda_allocator<Key>,
          class Storage            = cuco::storage<1>>
class static_multiset {
  using impl_type = detail::
    open_addressing_impl<Key, Key, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>;

 public:
  static constexpr auto cg_size      = impl_type::cg_size;       ///< CG size used for probing
  static constexpr auto window_size  = impl_type::window_size;   ///< Window size used for probing
  static constexpr auto thread_scope = impl_type::thread_scope;  ///< CUDA thread scope

  using key_type       = typename impl_type::key_type;        ///< Key type
  using value_type     = typename impl_type::value_type;      ///< Key type
  using extent_type    = typename impl_type::extent_type;     ///< Extent type
  using size_type      = typename impl_type::size_type;       ///< Size type
  using key_equal      = typename impl_type::key_equal;       ///< Key equality comparator type
  using allocator_type = typename impl_type::allocator_type;  ///< Allocator type
  /// Non-owning window storage ref type
  using storage_ref_type    = typename impl_type::storage_ref_type;
  using probing_scheme_type = typename impl_type::probing_scheme_type;  ///< Probing scheme type

  template <typename... Operators>
  using ref_type = cuco::static_multiset_ref<key_type,
                                             thread_scope,
                                             key_equal,
                                             probing_scheme_type,
                                             storage_ref_type,
                                             Operators...>;  ///< Non-owning container ref type

  static_multiset(static_multiset const&)            = delete;
  static_multiset& operator=(static_multiset const&) = delete;

  static_multiset(static_multiset&&) = default;  ///< Move constructor

  /**
   * @brief Replaces the contents of the container with another container.
   *
   * @return Reference of the current multiset object
   */
  static_multiset& operator=(static_multiset&&) = default;
  ~static_multiset()                            = default;

  /**
   * @brief Constructs a statically-sized multiset with the specified initial capacity, sentinel
   * values and CUDA stream
   *
   * The actual multiset capacity depends on the given `capacity`, the probing scheme, CG size, and
   * the window size and it is computed via the `make_window_extent` factory. Insert operations will
   * not automatically grow the set. Attempting to insert more unique keys than the capacity of the
   * multiset results in undefined behavior.
   *
   * @note Any `*_sentinel`s are reserved and behavior is undefined when attempting to insert
   * this sentinel value.
   * @note This constructor doesn't synchronize the given stream.
   *
   * @param capacity The requested lower-bound multiset size
   * @param empty_key_sentinel The reserved key value for empty slots
   * @param pred Key equality binary predicate
   * @param probing_scheme Probing scheme
   * @param scope The scope in which operations will be performed
   * @param storage Kind of storage to use
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the set
   */
  constexpr static_multiset(Extent capacity,
                            empty_key<Key> empty_key_sentinel,
                            KeyEqual const& pred                = {},
                            ProbingScheme const& probing_scheme = {},
                            cuda_thread_scope<Scope> scope      = {},
                            Storage storage                     = {},
                            Allocator const& alloc              = {},
                            cuda_stream_ref stream              = {});

  /**
   * @brief Constructs a statically-sized multiset with the number of elements to insert `n`, the
   * desired load factor, etc
   *
   * @note This constructor helps users create a set based on the number of elements to insert and
   * the desired load factor without manually computing the desired capacity. The actual set
   * capacity will be a size no smaller than `ceil(n / desired_load_factor)`. It's determined by
   * multiple factors including the given `n`, the desired load factor, the probing scheme, the CG
   * size, and the window size and is computed via the `make_window_extent` factory.
   * @note Insert operations will not automatically grow the container.
   * @note Attempting to insert more unique keys than the capacity of the container results in
   * undefined behavior.
   * @note Any `*_sentinel`s are reserved and behavior is undefined when attempting to insert
   * this sentinel value.
   * @note This constructor doesn't synchronize the given stream.
   * @note This overload will convert compile-time extents to runtime constants which might lead to
   * performance regressions.
   *
   * @throw If the desired occupancy is no bigger than zero
   * @throw If the desired occupancy is no smaller than one
   *
   * @param n The number of elements to insert
   * @param desired_load_factor The desired load factor of the container, e.g., 0.5 implies a 50%
   * load factor
   * @param empty_key_sentinel The reserved key value for empty slots
   * @param pred Key equality binary predicate
   * @param probing_scheme Probing scheme
   * @param scope The scope in which operations will be performed
   * @param storage Kind of storage to use
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the set
   */
  constexpr static_multiset(Extent n,
                            double desired_load_factor,
                            empty_key<Key> empty_key_sentinel,
                            KeyEqual const& pred                = {},
                            ProbingScheme const& probing_scheme = {},
                            cuda_thread_scope<Scope> scope      = {},
                            Storage storage                     = {},
                            Allocator const& alloc              = {},
                            cuda_stream_ref stream              = {});

  /**
   * @brief Constructs a statically-sized set with the specified initial capacity, sentinel values
   * and CUDA stream.
   *
   * The actual set capacity depends on the given `capacity`, the probing scheme, CG size, and the
   * window size and it is computed via the `make_window_extent` factory. Insert operations will not
   * automatically grow the set. Attempting to insert more unique keys than the capacity of the
   * multiset results in undefined behavior.
   *
   * @note Any `*_sentinel`s are reserved and behavior is undefined when attempting to insert
   * this sentinel value.
   * @note If a non-default CUDA stream is provided, the caller is responsible for synchronizing the
   * stream before the object is first used.
   *
   * @param capacity The requested lower-bound set size
   * @param empty_key_sentinel The reserved key value for empty slots
   * @param erased_key_sentinel The reserved key to denote erased slots
   * @param pred Key equality binary predicate
   * @param probing_scheme Probing scheme
   * @param scope The scope in which operations will be performed
   * @param storage Kind of storage to use
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the set
   */
  constexpr static_multiset(Extent capacity,
                            empty_key<Key> empty_key_sentinel,
                            erased_key<Key> erased_key_sentinel,
                            KeyEqual const& pred                = {},
                            ProbingScheme const& probing_scheme = {},
                            cuda_thread_scope<Scope> scope      = {},
                            Storage storage                     = {},
                            Allocator const& alloc              = {},
                            cuda_stream_ref stream              = {});

  /**
   * @brief Erases all elements from the container. After this call, `size()` returns zero.
   * Invalidates any references, pointers, or iterators referring to contained elements.
   *
   * @param stream CUDA stream this operation is executed in
   */
  void clear(cuda_stream_ref stream = {}) noexcept;

  /**
   * @brief Asynchronously erases all elements from the container. After this call, `size()` returns
   * zero. Invalidates any references, pointers, or iterators referring to contained elements.
   *
   * @param stream CUDA stream this operation is executed in
   */
  void clear_async(cuda_stream_ref stream = {}) noexcept;

  /**
   * @brief Inserts all keys in the range `[first, last)`
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `insert_async`.
   *
   * // TODO: to be revised due to heterogeneous lookup
   * @tparam InputIt Device accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * static_multiset<K>::value_type></tt> is `true`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stream CUDA stream used for insert
   */
  template <typename InputIt>
  void insert(InputIt first, InputIt last, cuda_stream_ref stream = {});

  /**
   * @brief Asynchronously inserts all keys in the range `[first, last)`.
   *
   * // TODO: to be revised due to heterogeneous lookup
   * @tparam InputIt Device accessible random access input iterator where
   * <tt>std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * static_multiset<K>::value_type></tt> is `true`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stream CUDA stream used for insert
   */
  template <typename InputIt>
  void insert_async(InputIt first, InputIt last, cuda_stream_ref stream = {}) noexcept;

  /**
   * @brief Inserts keys in the range `[first, last)` if `pred` of the corresponding stencil returns
   * true.
   *
   * @note The key `*(first + i)` is inserted if `pred( *(stencil + i) )` returns true.
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `insert_if_async`.
   *
   * @tparam InputIt Device accessible random access iterator whose `value_type` is
   * convertible to the container's `value_type`
   * @tparam StencilIt Device accessible random access iterator whose value_type is
   * convertible to Predicate's argument type
   * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool` and
   * argument type is convertible from <tt>std::iterator_traits<StencilIt>::value_type</tt>
   *
   * @param first Beginning of the sequence of key/value pairs
   * @param last End of the sequence of key/value pairs
   * @param stencil Beginning of the stencil sequence
   * @param pred Predicate to test on every element in the range `[stencil, stencil +
   * std::distance(first, last))`
   * @param stream CUDA stream used for the operation
   */
  template <typename InputIt, typename StencilIt, typename Predicate>
  void insert_if(
    InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda_stream_ref stream = {});

  /**
   * @brief Asynchronously inserts keys in the range `[first, last)` if `pred` of the corresponding
   * stencil returns true.
   *
   * @note The key `*(first + i)` is inserted if `pred( *(stencil + i) )` returns true.
   *
   * @tparam InputIt Device accessible random access iterator whose `value_type` is
   * convertible to the container's `value_type`
   * @tparam StencilIt Device accessible random access iterator whose value_type is
   * convertible to Predicate's argument type
   * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool` and
   * argument type is convertible from <tt>std::iterator_traits<StencilIt>::value_type</tt>
   *
   * @param first Beginning of the sequence of key/value pairs
   * @param last End of the sequence of key/value pairs
   * @param stencil Beginning of the stencil sequence
   * @param pred Predicate to test on every element in the range `[stencil, stencil +
   * std::distance(first, last))`
   * @param stream CUDA stream used for the operation
   */
  template <typename InputIt, typename StencilIt, typename Predicate>
  void insert_if_async(InputIt first,
                       InputIt last,
                       StencilIt stencil,
                       Predicate pred,
                       cuda_stream_ref stream = {}) noexcept;

  /**
   * @brief Gets the number of elements in the container.
   *
   * @note This function synchronizes the given stream.
   *
   * @param stream CUDA stream used to get the number of inserted elements
   * @return The number of elements in the container
   */
  [[nodiscard]] size_type size(cuda_stream_ref stream = {}) const noexcept;

  /**
   * @brief Gets the maximum number of elements the multiset can hold.
   *
   * @return The maximum number of elements the multiset can hold
   */
  [[nodiscard]] constexpr auto capacity() const noexcept;

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] constexpr key_type empty_key_sentinel() const noexcept;

  /**
   * @brief Gets the sentinel value used to represent an erased key slot.
   *
   * @return The sentinel value used to represent an erased key slot
   */
  [[nodiscard]] constexpr key_type erased_key_sentinel() const noexcept;

  /**
   * @brief Get device ref with operators.
   *
   * @tparam Operators Set of `cuco::op` to be provided by the ref
   *
   * @param ops List of operators, e.g., `cuco::insert`
   *
   * @return Device ref of the current `static_multiset` object
   */
  template <typename... Operators>
  [[nodiscard]] auto ref(Operators... ops) const noexcept;

 private:
  std::unique_ptr<impl_type> impl_;
};
}  // namespace cuco

#include <cuco/detail/static_multiset/static_multiset.inl>
