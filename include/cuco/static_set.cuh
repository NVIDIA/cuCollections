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

#include <cuco/cuda_stream_ref.hpp>
#include <cuco/detail/open_addressing_impl.cuh>
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
 * The `static_set` supports two types of operations:
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
          class Extent             = cuco::experimental::extent<std::size_t>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class KeyEqual           = thrust::equal_to<Key>,
          class ProbingScheme      = experimental::double_hashing<4,  // CG size
                                                             cuco::default_hash_function<Key>>,
          class Allocator          = cuco::cuda_allocator<Key>,
          class Storage            = cuco::experimental::storage<1>>
class static_set {
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
  using ref_type =
    cuco::experimental::static_set_ref<key_type,
                                       thread_scope,
                                       key_equal,
                                       probing_scheme_type,
                                       storage_ref_type,
                                       Operators...>;  ///< Non-owning container ref type

  static_set(static_set const&) = delete;
  static_set& operator=(static_set const&) = delete;

  static_set(static_set&&) = default;  ///< Move constructor

  /**
   * @brief Replaces the contents of the container with another container.
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
   * window size and it is computed via the `make_window_extent` factory. Insert operations will not
   * automatically grow the set. Attempting to insert more unique keys than the capacity of the map
   * results in undefined behavior.
   *
   * @note Any `*_sentinel`s are reserved and behavior is undefined when attempting to insert
   * this sentinel value.
   * @note If a non-default CUDA stream is provided, the caller is responsible for synchronizing the
   * stream before the object is first used.
   *
   * @param capacity The requested lower-bound set size
   * @param empty_key_sentinel The reserved key value for empty slots
   * @param pred Key equality binary predicate
   * @param probing_scheme Probing scheme
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the set
   */
  constexpr static_set(Extent capacity,
                       empty_key<Key> empty_key_sentinel,
                       KeyEqual const& pred                = {},
                       ProbingScheme const& probing_scheme = {},
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
  size_type insert(InputIt first, InputIt last, cuda_stream_ref stream = {});

  /**
   * @brief Asynchronously inserts all keys in the range `[first, last)`.
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
  void insert_async(InputIt first, InputIt last, cuda_stream_ref stream = {}) noexcept;

  /**
   * @brief Inserts keys in the range `[first, last)` if `pred` of the corresponding stencil returns
   * true.
   *
   * @note The key `*(first + i)` is inserted if `pred( *(stencil + i) )` returns true.
   * @note This function synchronizes the given stream and returns the number of successful
   * insertions. For asynchronous execution use `insert_if_async`.
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
   *
   * @return Number of successfully inserted keys
   */
  template <typename InputIt, typename StencilIt, typename Predicate>
  size_type insert_if(
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
                cuda_stream_ref stream = {}) const;

  /**
   * @brief Asynchronously indicates whether the keys in the range `[first, last)` are contained in
   * the set.
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
                      cuda_stream_ref stream = {}) const noexcept;

  /**
   * @brief Indicates whether the keys in the range `[first, last)` are contained in the set if
   * `pred` of the corresponding stencil returns true.
   *
   * @note If `pred( *(stencil + i) )` is true, stores `true` or `false` to `(output_begin + i)`
   * indicating if the key `*(first + i)` is present in the set. If `pred( *(stencil + i) )` is
   * false, stores false to `(output_begin + i)`.
   * @note This function synchronizes the given stream. For asynchronous execution use
   * `contains_if_async`.
   *
   * @tparam InputIt Device accessible input iterator
   * @tparam StencilIt Device accessible random access iterator whose value_type is
   * convertible to Predicate's argument type
   * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool` and
   * argument type is convertible from <tt>std::iterator_traits<StencilIt>::value_type</tt>
   * @tparam OutputIt Device accessible output iterator assignable from `bool`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stencil Beginning of the stencil sequence
   * @param pred Predicate to test on every element in the range `[stencil, stencil +
   * std::distance(first, last))`
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param stream Stream used for executing the kernels
   */
  template <typename InputIt, typename StencilIt, typename Predicate, typename OutputIt>
  void contains_if(InputIt first,
                   InputIt last,
                   StencilIt stencil,
                   Predicate pred,
                   OutputIt output_begin,
                   cuda_stream_ref stream = {}) const;

  /**
   * @brief Asynchronously indicates whether the keys in the range `[first, last)` are contained in
   * the set if `pred` of the corresponding stencil returns true.
   *
   * @note If `pred( *(stencil + i) )` is true, stores `true` or `false` to `(output_begin + i)`
   * indicating if the key `*(first + i)` is present in the set. If `pred( *(stencil + i) )` is
   * false, stores false to `(output_begin + i)`.
   *
   * @tparam InputIt Device accessible input iterator
   * @tparam StencilIt Device accessible random access iterator whose value_type is
   * convertible to Predicate's argument type
   * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool` and
   * argument type is convertible from <tt>std::iterator_traits<StencilIt>::value_type</tt>
   * @tparam OutputIt Device accessible output iterator assignable from `bool`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stencil Beginning of the stencil sequence
   * @param pred Predicate to test on every element in the range `[stencil, stencil +
   * std::distance(first, last))`
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param stream Stream used for executing the kernels
   */
  template <typename InputIt, typename StencilIt, typename Predicate, typename OutputIt>
  void contains_if_async(InputIt first,
                         InputIt last,
                         StencilIt stencil,
                         Predicate pred,
                         OutputIt output_begin,
                         cuda_stream_ref stream = {}) const noexcept;

  /**
   * @brief For all keys in the range `[first, last)`, finds an element with key equivalent to the
   * query key.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use `find_async`.
   * @note If the key `*(first + i)` has a matched `element` in the set, copies `element` to
   * `(output_begin + i)`. Else, copies the empty key sentinel.
   *
   * @tparam InputIt Device accessible input iterator
   * @tparam OutputIt Device accessible output iterator assignable from the set's `key_type`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of elements retrieved for each key
   * @param stream Stream used for executing the kernels
   */
  template <typename InputIt, typename OutputIt>
  void find(InputIt first, InputIt last, OutputIt output_begin, cuda_stream_ref stream = {}) const;

  /**
   * @brief For all keys in the range `[first, last)`, asynchronously finds an element with key
   * equivalent to the query key.
   *
   * @note If the key `*(first + i)` has a matched `element` in the set, copies `element` to
   * `(output_begin + i)`. Else, copies the empty key sentinel.
   *
   * @tparam InputIt Device accessible input iterator
   * @tparam OutputIt Device accessible output iterator assignable from the set's `key_type`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of elements retrieved for each key
   * @param stream Stream used for executing the kernels
   */
  template <typename InputIt, typename OutputIt>
  void find_async(InputIt first,
                  InputIt last,
                  OutputIt output_begin,
                  cuda_stream_ref stream = {}) const;

  /**
   * @brief Retrieves all keys contained in the set.
   *
   * @note This API synchronizes the given stream.
   * @note The order in which keys are returned is implementation defined and not guaranteed to be
   * consistent between subsequent calls to `retrieve_all`.
   * @note Behavior is undefined if the range beginning at `output_begin` is smaller than the return
   * value of `size()`.
   *
   * @tparam OutputIt Device accessible random access output iterator whose `value_type` is
   * convertible from the container's `key_type`.
   *
   * @param output_begin Beginning output iterator for keys
   * @param stream CUDA stream used for this operation
   *
   * @return Iterator indicating the end of the output
   */
  template <typename OutputIt>
  OutputIt retrieve_all(OutputIt output_begin, cuda_stream_ref stream = {}) const;

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
  std::unique_ptr<impl_type> impl_;
};
}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/static_set/static_set.inl>
