/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cuco/detail/error.hpp>
#include <cuco/detail/prime.hpp>
#include <cuco/detail/static_multimap/kernels.cuh>
#include <cuco/probe_sequences.cuh>
#include <cuco/sentinel.cuh>
#include <cuco/traits.hpp>

#include <thrust/functional.h>

#include <cuda/std/atomic>
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

#include <cooperative_groups.h>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace cuco {

/**
 * @brief A GPU-accelerated, unordered, associative container of key-value
 * pairs that supports equivalent keys.
 *
 * Allows constant time concurrent inserts or concurrent find operations from threads in device
 * code. Concurrent insert/find is allowed only when
 * `static_multimap<Key, Value>::supports_concurrent_insert_find()` is true.
 *
 * Current limitations:
 * - Requires keys and values where `cuco::is_bitwise_comparable_v<T>` is true
 * - Comparisons against the "sentinel" values will always be done with bitwise comparisons
 * Therefore, the objects must have unique, bitwise object representations (e.g., no padding bits).
 * - Does not support erasing keys
 * - Capacity is fixed and will not grow automatically
 * - Requires the user to specify sentinel values for both key and mapped value
 * to indicate empty slots
 * - Concurrent insert/find is only supported when `static_multimap<Key,
 * Value>::supports_concurrent_insert_find()` is true`
 *
 * The `static_multimap` supports two types of operations:
 * - Host-side "bulk" operations
 * - Device-side "singular" operations
 *
 * The host-side bulk operations include `insert`, `contains`, `count`, `retrieve` and their
 * variants. These APIs should be used when there are a large number of keys to insert or lookup in
 * the map. For example, given a range of keys specified by device-accessible iterators, the bulk
 * `insert` function will insert all keys into the map.
 *
 * The singular device-side operations allow individual threads to perform
 * independent operations (e.g. `insert`, etc.) from device code. These
 * operations are accessed through non-owning, trivially copyable "view" types:
 * `device_view` and `device_mutable_view`. The `device_view` class is an
 * immutable view that allows only non-modifying operations such as `count` or
 * `contains`. The `device_mutable_view` class only allows `insert` operations.
 * The two types are separate to prevent erroneous concurrent insert/find
 * operations.
 *
 * By default, when querying for a Key `k` in operations like `count` or `retrieve`, if `k` is not
 * present in the map, it will not contribute to the output. Query APIs with the `_outer` suffix
 * will include non-matching keys in the output. See the relevant API documentation for more
 * information.
 *
 * Typical associative container query APIs like `retrieve` look up values by solely by key, e.g.,
 * `count` for a Key `k` will count all values whose associated key `k'` matches `k` as determined
 * by `key_equal(k, k')`. In some cases, one may want to consider both key _and_ value when
 * determining if a key-value pair should contribute to the output. `static_multimap` supports this
 * use case with APIs prefixed with `pair_`, e.g., `pair_count` is given a key-value pair
 * `{k,v}` and only counts key-value pairs, `{k', v'}`, in the map where `pair_equal({k,v}, {k',
 * v'})` is true. See the relevant API documentation for more information.
 *
 * Example:
 * \code{.cpp}
 * int empty_key_sentinel = -1;
 * int empty_value_sentinel = -1;
 *
 * // Constructs a multimap with 100,000 slots using -1 and -1 as the empty key/value
 * // sentinels. Note the capacity is chosen knowing we will insert 50,000 keys,
 * // for an load factor of 50%.
 * static_multimap<int, int> m{100'000, empty_key_sentinel, empty_value_sentinel};
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
 * // `contains/count/retrieve` lookups
 * kernel<<<...>>>(m.get_device_view());
 * \endcode
 *
 *
 * @tparam Key Type used for keys. Requires `cuco::is_bitwise_comparable_v<Key>`
 * @tparam Value Type of the mapped values. Requires `cuco::is_bitwise_comparable_v<Value>`
 * @tparam Scope The scope in which multimap operations will be performed by
 * individual threads
 * @tparam ProbeSequence Probe sequence chosen between `cuco::detail::linear_probing`
 * and `cuco::detail::double_hashing`. (see `detail/probe_sequences.cuh`)
 * @tparam Allocator Type of allocator used for device storage
 */
template <typename Key,
          typename Value,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          typename Allocator       = cuco::cuda_allocator<char>,
          class ProbeSequence =
            cuco::double_hashing<8, detail::MurmurHash3_32<Key>, detail::MurmurHash3_32<Key>>>
class static_multimap {
  static_assert(
    cuco::is_bitwise_comparable_v<Key>,
    "Key type must have unique object representations or have been explicitly declared as safe for "
    "bitwise comparison via specialization of cuco::is_bitwise_comparable_v<Key>.");

  static_assert(
    cuco::is_bitwise_comparable_v<Value>,
    "Value type must have unique object representations or have been explicitly declared as safe "
    "for bitwise comparison via specialization of cuco::is_bitwise_comparable_v<Value>.");

  static_assert(
    std::is_base_of_v<cuco::detail::probe_sequence_base<ProbeSequence::cg_size>, ProbeSequence>,
    "ProbeSequence must be a specialization of either cuco::double_hashing or "
    "cuco::linear_probing");

 public:
  using value_type         = cuco::pair_type<Key, Value>;
  using key_type           = Key;
  using mapped_type        = Value;
  using atomic_key_type    = cuda::atomic<key_type, Scope>;
  using atomic_mapped_type = cuda::atomic<mapped_type, Scope>;
  using pair_atomic_type   = cuco::pair_type<atomic_key_type, atomic_mapped_type>;
  using atomic_ctr_type    = cuda::atomic<std::size_t, Scope>;
  using allocator_type     = Allocator;
  using slot_allocator_type =
    typename std::allocator_traits<Allocator>::rebind_alloc<pair_atomic_type>;
  using counter_allocator_type =
    typename std::allocator_traits<Allocator>::rebind_alloc<atomic_ctr_type>;
  using probe_sequence_type = detail::probe_sequence<ProbeSequence, Key, Value, Scope>;

  static_multimap(static_multimap const&) = delete;
  static_multimap& operator=(static_multimap const&) = delete;

  static_multimap(static_multimap&&) = default;
  static_multimap& operator=(static_multimap&&) = default;
  ~static_multimap()                            = default;

  /**
   * @brief Indicate if concurrent insert/find is supported for the key/value types.
   *
   * @return Boolean indicating if concurrent insert/find is supported.
   */
  __host__ __device__ __forceinline__ static constexpr bool
  supports_concurrent_insert_find() noexcept
  {
    return cuco::detail::is_packable<value_type>();
  }

  /**
   * @brief The size of the CUDA cooperative thread group.
   *
   * @return The CG size.
   */
  __host__ __device__ __forceinline__ static constexpr uint32_t cg_size() noexcept
  {
    return ProbeSequence::cg_size;
  }

  /**
   * @brief Construct a statically-sized map with the specified initial capacity,
   * sentinel values and CUDA stream.
   *
   * The capacity of the map is fixed. Insert operations will not automatically
   * grow the map. Attempting to insert more unique keys than the capacity of
   * the map results in undefined behavior.
   *
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
   * @param stream CUDA stream used to initialize the map
   * @param alloc Allocator used for allocating device storage
   */
  static_multimap(std::size_t capacity,
                  sentinel::empty_key<Key> empty_key_sentinel,
                  sentinel::empty_value<Value> empty_value_sentinel,
                  cudaStream_t stream    = 0,
                  Allocator const& alloc = Allocator{});

  /**
   * @brief Inserts all key/value pairs in the range `[first, last)`.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * `std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * static_multimap<K, V>::value_type>` is `true`
   * @param first Beginning of the sequence of key/value pairs
   * @param last End of the sequence of key/value pairs
   * @param stream CUDA stream used for insert
   */
  template <typename InputIt>
  void insert(InputIt first, InputIt last, cudaStream_t stream = 0);

  /**
   * @brief Inserts key/value pairs in the range `[first, first + n)` if `pred`
   * of the corresponding stencil returns true.
   *
   * The key/value pair `*(first + i)` is inserted if `pred( *(stencil + i) )` returns true.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * `std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * static_multimap<K, V>::value_type>` is `true`
   * @tparam StencilIt Device accessible random access iterator whose value_type is
   * convertible to Predicate's argument type
   * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool` and
   * argument type is convertible from `std::iterator_traits<StencilIt>::value_type`.
   * @param first Beginning of the sequence of key/value pairs
   * @param last End of the sequence of key/value pairs
   * @param stencil Beginning of the stencil sequence
   * @param pred Predicate to test on every element in the range `[stencil, stencil +
   * std::distance(first, last))`
   * @param stream CUDA stream used for insert
   */
  template <typename InputIt, typename StencilIt, typename Predicate>
  void insert_if(
    InputIt first, InputIt last, StencilIt stencil, Predicate pred, cudaStream_t stream = 0);

  /**
   * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
   *
   * Stores `true` or `false` to `(output + i)` indicating if the key `*(first + i)` exists in the
   * map.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is convertible from
   * `bool`
   * @tparam KeyEqual Binary callable type used to compare two keys for equality
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the output sequence indicating whether each key is present
   * @param stream CUDA stream used for contains
   * @param key_equal The binary function to compare two keys for equality
   */
  template <typename InputIt, typename OutputIt, typename KeyEqual = thrust::equal_to<key_type>>
  void contains(InputIt first,
                InputIt last,
                OutputIt output_begin,
                cudaStream_t stream = 0,
                KeyEqual key_equal  = KeyEqual{}) const;

  /**
   * @brief Indicates whether the pairs in the range `[first, last)` are contained in the map.
   *
   * Stores `true` or `false` to `(output + i)` indicating if the pair `*(first + i)` exists in
   * the map.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * `std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * static_multimap<K, V>::value_type>` is `true`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is convertible from
   * `bool`
   * @tparam PairEqual Binary callable type used to compare input pair and slot content for equality
   *
   * @param first Beginning of the sequence of pairs
   * @param last End of the sequence of pairs
   * @param output_begin Beginning of the output sequence indicating whether each pair is present
   * @param pair_equal The binary function to compare input pair and slot content for equality
   * @param stream CUDA stream used for contains
   */
  template <typename InputIt, typename OutputIt, typename PairEqual = thrust::equal_to<value_type>>
  void pair_contains(InputIt first,
                     InputIt last,
                     OutputIt output_begin,
                     PairEqual pair_equal = PairEqual{},
                     cudaStream_t stream  = 0) const;

  /**
   * @brief Counts the occurrences of keys in `[first, last)` contained in the multimap.
   *
   * For each key, `k = *(first + i)`, counts all matching keys, `k'`, as determined by
   * `key_equal(k, k')` and returns the sum of all matches for all keys.
   *
   * @tparam Input Device accesible input iterator whose `value_type` is convertible to `key_type`
   * @tparam KeyEqual Binary callable
   * @param first Beginning of the sequence of keys to count
   * @param last End of the sequence of keys to count
   * @param stream CUDA stream used for count
   * @param key_equal Binary function to compare two keys for equality
   * @return The sum of total occurrences of all keys in `[first, last)`
   */
  template <typename InputIt, typename KeyEqual = thrust::equal_to<key_type>>
  std::size_t count(InputIt first,
                    InputIt last,
                    cudaStream_t stream = 0,
                    KeyEqual key_equal  = KeyEqual{}) const;

  /**
   * @brief Counts the occurrences of keys in `[first, last)` contained in the multimap.
   *
   * For each key, `k = *(first + i)`, counts all matching keys, `k'`, as determined by
   * `key_equal(k, k')` and returns the sum of all matches for all keys. If `k` does not have any
   * matches, it contributes 1 to the final sum.
   *
   * @tparam Input Device accesible input iterator whose `value_type` is convertible to `key_type`
   * @tparam KeyEqual Binary callable
   * @param first Beginning of the sequence of keys to count
   * @param last End of the sequence of keys to count
   * @param stream CUDA stream used for count_outer
   * @param key_equal Binary function to compare two keys for equality
   * @return The sum of total occurrences of all keys in `[first, last)` where keys without matches
   * are considered to have a single occurrence.
   */
  template <typename InputIt, typename KeyEqual = thrust::equal_to<key_type>>
  std::size_t count_outer(InputIt first,
                          InputIt last,
                          cudaStream_t stream = 0,
                          KeyEqual key_equal  = KeyEqual{}) const;

  /**
   * @brief Counts the occurrences of key/value pairs in `[first, last)` contained in the multimap.
   *
   * For key-value pair, `kv = *(first + i)`, counts all matching key-value pairs, `kv'`, as
   * determined by `pair_equal(kv, kv')` and returns the sum of all matches for all key-value pairs.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * `std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * static_multimap<K, V>::value_type>` is `true`
   * @tparam PairEqual Binary callable
   * @param first Beginning of the sequence of pairs to count
   * @param last End of the sequence of pairs to count
   * @param pair_equal Binary function to compare two pairs for equality
   * @param stream CUDA stream used for pair_count
   * @return The sum of total occurrences of all pairs in `[first, last)`
   */
  template <typename InputIt, typename PairEqual>
  std::size_t pair_count(InputIt first,
                         InputIt last,
                         PairEqual pair_equal,
                         cudaStream_t stream = 0) const;

  /**
   * @brief Counts the occurrences of key/value pairs in `[first, last)` contained in the multimap.
   *
   * For key-value pair, `kv = *(first + i)`, counts all matching key-value pairs, `kv'`, as
   * determined by `pair_equal(kv, kv')` and returns the sum of all matches for all key-value pairs.
   * if `kv` does not have any matches, it contributes 1 to the final sum.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * `std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * static_multimap<K, V>::value_type>` is `true`
   * @tparam PairEqual Binary callable
   * @param first Beginning of the sequence of pairs to count
   * @param last End of the sequence of pairs to count
   * @param pair_equal Binary function to compare two pairs for equality
   * @param stream CUDA stream used for pair_count_outer
   * @return The sum of total occurrences of all pairs in `[first, last)` where a key-value pair
   * without a match is considered to have a single occurrence
   */
  template <typename InputIt, typename PairEqual>
  std::size_t pair_count_outer(InputIt first,
                               InputIt last,
                               PairEqual pair_equal,
                               cudaStream_t stream = 0) const;

  /**
   * @brief Retrieves all the values corresponding to all keys in the range `[first, last)`.
   *
   * If key `k = *(first + i)` exists in the map, copies `k` and all associated values to
   * unspecified locations in `[output_begin, output_end)`. Else, does nothing.
   *
   * Behavior is undefined if the size of the output range exceeds `std::distance(output_begin,
   * output_end)`. Use `count()` to determine the size of the output range.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is
   * constructible from the map's `value_type`
   * @tparam KeyEqual Binary callable type
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of key/value pairs retrieved for each key
   * @param stream CUDA stream used for retrieve
   * @param key_equal The binary function to compare two keys for equality
   * @return The iterator indicating the last valid key/value pairs in the output
   */
  template <typename InputIt, typename OutputIt, typename KeyEqual = thrust::equal_to<key_type>>
  OutputIt retrieve(InputIt first,
                    InputIt last,
                    OutputIt output_begin,
                    cudaStream_t stream = 0,
                    KeyEqual key_equal  = KeyEqual{}) const;

  /**
   * @brief Retrieves all the matches corresponding to all keys in the range `[first, last)`.
   *
   * If key `k = *(first + i)` exists in the map, copies `k` and all associated values to
   * unspecified locations in `[output_begin, output_end)`. Else, copies `k` and
   * `empty_value_sentinel`.
   *
   * Behavior is undefined if the size of the output range exceeds `std::distance(output_begin,
   * output_end)`. Use `count_outer()` to determine the size of the output range.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is
   * constructible from the map's `value_type`
   * @tparam KeyEqual Binary callable type
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of key/value pairs retrieved for each key
   * @param stream CUDA stream used for retrieve_outer
   * @param key_equal The binary function to compare two keys for equality
   * @return The iterator indicating the last valid key/value pairs in the output
   */
  template <typename InputIt, typename OutputIt, typename KeyEqual = thrust::equal_to<key_type>>
  OutputIt retrieve_outer(InputIt first,
                          InputIt last,
                          OutputIt output_begin,
                          cudaStream_t stream = 0,
                          KeyEqual key_equal  = KeyEqual{}) const;

  /**
   * @brief Retrieves all pairs matching the input probe pair in the range `[first, last)`.
   *
   * The `pair_` prefix indicates that the input data type is convertible to the map's
   * `value_type`. If pair_equal(*(first + i), slot[j]) returns true, then *(first+i) is
   * stored to `probe_output_begin`, and slot[j] is stored to `contained_output_begin`.
   *
   * Behavior is undefined if the size of the output range exceeds
   * `std::distance(probe_output_begin, probe_output_end)` (or
   * `std::distance(contained_output_begin, contained_output_end)`). Use
   * `pair_count()` to determine the size of the output range.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * `std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * static_multimap<K, V>::value_type>` is `true`
   * @tparam OutputIt1 Device accessible output iterator whose `value_type` is constructible from
   * `InputIt`s `value_type`.
   * @tparam OutputIt2 Device accessible output iterator whose `value_type` is constructible from
   * the map's `value_type`.
   * @tparam PairEqual Binary callable type
   * @param first Beginning of the sequence of pairs
   * @param last End of the sequence of pairs
   * @param probe_output_begin Beginning of the sequence of the matched probe pairs
   * @param contained_output_begin Beginning of the sequence of the matched contained pairs
   * @param pair_equal The binary function to compare two pairs for equality
   * @param stream CUDA stream used for pair_retrieve
   * @return Pair of iterators pointing to the last elements in the output
   */
  template <typename InputIt, typename OutputIt1, typename OutputIt2, typename PairEqual>
  std::pair<OutputIt1, OutputIt2> pair_retrieve(InputIt first,
                                                InputIt last,
                                                OutputIt1 probe_output_begin,
                                                OutputIt2 contained_output_begin,
                                                PairEqual pair_equal,
                                                cudaStream_t stream = 0) const;

  /**
   * @brief Retrieves all pairs matching the input probe pair in the range `[first, last)`.
   *
   * The `pair_` prefix indicates that the input data type is convertible to the map's `value_type`.
   * If pair_equal(*(first + i), slot[j]) returns true, then *(first+i) is stored to
   * `probe_output_begin`, and slot[j] is stored to `contained_output_begin`. If *(first+i) doesn't
   * have matches in the map, copies *(first + i) in `probe_output_begin` and a pair of
   * `empty_key_sentinel` and `empty_value_sentinel` in `contained_output_begin`.
   *
   * Behavior is undefined if the size of the output range exceeds
   * `std::distance(probe_output_begin, probe_output_end)` (or
   * `std::distance(contained_output_begin, contained_output_end)`). Use
   * `pair_count()` to determine the size of the output range.
   *
   * @tparam InputIt Device accessible random access input iterator where
   * `std::is_convertible<std::iterator_traits<InputIt>::value_type,
   * static_multimap<K, V>::value_type>` is `true`
   * @tparam OutputIt1 Device accessible output iterator whose `value_type` is constructible from
   * `InputIt`s `value_type`.
   * @tparam OutputIt2 Device accessible output iterator whose `value_type` is constructible from
   * the map's `value_type`.
   * @tparam PairEqual Binary callable type
   * @param first Beginning of the sequence of pairs
   * @param last End of the sequence of pairs
   * @param probe_output_begin Beginning of the sequence of the matched probe pairs
   * @param contained_output_begin Beginning of the sequence of the matched contained pairs
   * @param pair_equal The binary function to compare two pairs for equality
   * @param stream CUDA stream used for pair_retrieve_outer
   * @return Pair of iterators pointing to the last elements in the output
   */
  template <typename InputIt, typename OutputIt1, typename OutputIt2, typename PairEqual>
  std::pair<OutputIt1, OutputIt2> pair_retrieve_outer(InputIt first,
                                                      InputIt last,
                                                      OutputIt1 probe_output_begin,
                                                      OutputIt2 contained_output_begin,
                                                      PairEqual pair_equal,
                                                      cudaStream_t stream = 0) const;

 private:
  /**
   * @brief Indicates if vector-load is used.
   *
   * Users have no explicit control on whether vector-load is used.
   *
   * @return Boolean indicating if vector-load is used.
   */
  static constexpr bool uses_vector_load() noexcept
  {
    return cuco::detail::is_packable<value_type>();
  }

  /**
   * @brief Returns the number of pairs loaded with each vector-load
   */
  static constexpr uint32_t vector_width() noexcept { return ProbeSequence::vector_width(); }

  /**
   * @brief Returns the warp size.
   */
  static constexpr uint32_t warp_size() noexcept { return 32u; }

  /**
   * @brief Custom deleter for unique pointer of device counter.
   */
  struct counter_deleter {
    counter_deleter(counter_allocator_type& a) : allocator{a} {}

    counter_deleter(counter_deleter const&) = default;

    void operator()(atomic_ctr_type* ptr) { allocator.deallocate(ptr, 1); }

    counter_allocator_type& allocator;
  };

  /**
   * @brief Custom deleter for unique pointer of slots.
   */
  struct slot_deleter {
    slot_deleter(slot_allocator_type& a, size_t& c) : allocator{a}, capacity{c} {}

    slot_deleter(slot_deleter const&) = default;

    void operator()(pair_atomic_type* ptr) { allocator.deallocate(ptr, capacity); }

    slot_allocator_type& allocator;
    size_t& capacity;
  };

  class device_view_impl_base;
  class device_mutable_view_impl;
  class device_view_impl;

  template <typename ViewImpl>
  class device_view_base {
   protected:
    // Import member type definitions from `static_multimap`
    using value_type          = value_type;
    using key_type            = Key;
    using mapped_type         = Value;
    using pair_atomic_type    = pair_atomic_type;
    using iterator            = pair_atomic_type*;
    using const_iterator      = pair_atomic_type const*;
    using probe_sequence_type = probe_sequence_type;

    __host__ __device__ device_view_base(pair_atomic_type* slots,
                                         std::size_t capacity,
                                         sentinel::empty_key<Key> empty_key_sentinel,
                                         sentinel::empty_value<Value> empty_value_sentinel) noexcept
      : impl_{slots, capacity, empty_key_sentinel.value, empty_value_sentinel.value}
    {
    }

   public:
    /**
     * @brief Gets slots array.
     *
     * @return Slots array
     */
    __device__ __forceinline__ pair_atomic_type* get_slots() noexcept { return impl_.get_slots(); }

    /**
     * @brief Gets slots array.
     *
     * @return Slots array
     */
    __device__ __forceinline__ pair_atomic_type const* get_slots() const noexcept
    {
      return impl_.get_slots();
    }

    /**
     * @brief Gets the maximum number of elements the hash map can hold.
     *
     * @return The maximum number of elements the hash map can hold
     */
    __host__ __device__ __forceinline__ std::size_t get_capacity() const noexcept
    {
      return impl_.get_capacity();
    }

    /**
     * @brief Gets the sentinel value used to represent an empty key slot.
     *
     * @return The sentinel value used to represent an empty key slot
     */
    __host__ __device__ __forceinline__ Key get_empty_key_sentinel() const noexcept
    {
      return impl_.get_empty_key_sentinel();
    }

    /**
     * @brief Gets the sentinel value used to represent an empty value slot.
     *
     * @return The sentinel value used to represent an empty value slot
     */
    __host__ __device__ __forceinline__ Value get_empty_value_sentinel() const noexcept
    {
      return impl_.get_empty_value_sentinel();
    }

   protected:
    ViewImpl impl_;
  };  // class device_view_base

 public:
  /**
   * @brief Mutable, non-owning view-type that may be used in device code to
   * perform singular inserts into the map.
   *
   * `device_mutable_view` is trivially-copyable and is intended to be passed by
   * value.
   *
   * Example:
   * \code{.cpp}
   * cuco::static_multimap<int,int> m{100'000, -1, -1};
   *
   * // Inserts a sequence of pairs {{0,0}, {1,1}, ... {i,i}}
   * thrust::for_each(thrust::make_counting_iterator(0),
   *                  thrust::make_counting_iterator(50'000),
   *                  [map = m.get_device_mutable_view()]
   *                  __device__ (auto i) mutable {
   *                     map.insert(thrust::make_pair(i,i));
   *                  });
   * \endcode
   */
  class device_mutable_view : public device_view_base<device_mutable_view_impl> {
   public:
    using view_base_type = device_view_base<device_mutable_view_impl>;
    using value_type     = typename view_base_type::value_type;
    using key_type       = typename view_base_type::key_type;
    using mapped_type    = typename view_base_type::mapped_type;
    using iterator       = typename view_base_type::iterator;
    using const_iterator = typename view_base_type::const_iterator;

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
    __host__ __device__
    device_mutable_view(pair_atomic_type* slots,
                        std::size_t capacity,
                        sentinel::empty_key<Key> empty_key_sentinel,
                        sentinel::empty_value<Value> empty_value_sentinel) noexcept
      : view_base_type{slots, capacity, empty_key_sentinel, empty_value_sentinel}
    {
    }

    /**
     * @brief Inserts the specified key/value pair into the map.
     *
     * @param g The Cooperative Group that performs the insert
     * @param insert_pair The pair to insert
     * @return void.
     */
    __device__ __forceinline__ void insert(
      cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
      value_type const& insert_pair) noexcept;

   private:
    using device_view_base<device_mutable_view_impl>::impl_;
  };  // class device mutable view

  /**
   * @brief Non-owning view-type that may be used in device code to
   * perform singular find and contains operations for the map.
   *
   * `device_view` is trivially-copyable and is intended to be passed by
   * value.
   *
   */
  class device_view : public device_view_base<device_view_impl> {
   public:
    using view_base_type = device_view_base<device_view_impl>;
    using value_type     = typename view_base_type::value_type;
    using key_type       = typename view_base_type::key_type;
    using mapped_type    = typename view_base_type::mapped_type;
    using iterator       = typename view_base_type::iterator;
    using const_iterator = typename view_base_type::const_iterator;

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
    __host__ __device__ device_view(pair_atomic_type* slots,
                                    std::size_t capacity,
                                    sentinel::empty_key<Key> empty_key_sentinel,
                                    sentinel::empty_value<Value> empty_value_sentinel) noexcept
      : view_base_type{slots, capacity, empty_key_sentinel, empty_value_sentinel}
    {
    }

    /**
     * @brief Makes a copy of given `device_view` using non-owned memory.
     *
     * This function is intended to be used to create shared memory copies of small static maps,
     * although global memory can be used as well.
     *
     * @tparam CG The type of the cooperative thread group
     * @param g The cooperative thread group used to copy the slots
     * @param source_device_view `device_view` to copy from
     * @param memory_to_use Array large enough to support `capacity` elements. Object does not
     * take the ownership of the memory
     * @return Copy of passed `device_view`
     */
    template <typename CG>
    __device__ __forceinline__ static device_view make_copy(
      CG g, pair_atomic_type* const memory_to_use, device_view source_device_view) noexcept;

    /**
     * @brief Flushes per-CG buffer into the output sequence.
     *
     * A given CUDA Cooperative Group, `g`, loads `num_outputs` key-value pairs from `output_buffer`
     * and writes them into global memory in a coalesced fashion. CG-wide `memcpy_sync` is used if
     * `CUCO_HAS_CG_MEMCPY_ASYNC` is defined and `thrust::is_contiguous_iterator_v<OutputIt>`
     * returns true. All threads of `g` must be active due to implicit CG-wide synchronization
     * during flushing.
     *
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputIt Device accessible output iterator whose `value_type` is
     * constructible from the map's `value_type`
     * @param g The Cooperative Group used to flush output buffer
     * @param num_outputs Number of valid output in the buffer
     * @param output_buffer Buffer of the key/value pair sequence
     * @param num_matches Size of the output sequence
     * @param output_begin Beginning of the output sequence of key/value pairs
     */
    template <typename CG, typename atomicT, typename OutputIt>
    __device__ __forceinline__ void flush_output_buffer(CG const& g,
                                                        uint32_t const num_outputs,
                                                        value_type* output_buffer,
                                                        atomicT* num_matches,
                                                        OutputIt output_begin) noexcept;

    /**
     * @brief Flushes per-CG buffer into the output sequences.
     *
     * A given CUDA Cooperative Group, `g`, loads `num_outputs` elements from `probe_output_buffer`
     * and `num_outputs` elements from `contained_output_buffer`, then writes them into global
     * memory started from `probe_output_begin` and `contained_output_begin` respectively. All
     * threads of `g` must be active due to implicit CG-wide synchronization during flushing.
     *
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputIt1 Device accessible output iterator whose `value_type` is constructible from
     * `InputIt`s `value_type`.
     * @tparam OutputIt2 Device accessible output iterator whose `value_type` is constructible from
     * the map's `value_type`.
     * @param g The Cooperative Group used to flush output buffer
     * @param num_outputs Number of valid output in the buffer
     * @param probe_output_buffer Buffer of the matched probe pair sequence
     * @param contained_output_buffer Buffer of the matched contained pair sequence
     * @param num_matches Size of the output sequence
     * @param probe_output_begin Beginning of the output sequence of the matched probe pairs
     * @param contained_output_begin Beginning of the output sequence of the matched contained
     * pairs
     */
    template <typename CG, typename atomicT, typename OutputIt1, typename OutputIt2>
    __device__ __forceinline__ void flush_output_buffer(CG const& g,
                                                        uint32_t const num_outputs,
                                                        value_type* probe_output_buffer,
                                                        value_type* contained_output_buffer,
                                                        atomicT* num_matches,
                                                        OutputIt1 probe_output_begin,
                                                        OutputIt2 contained_output_begin) noexcept;

    /**
     * @brief Indicates whether the key `k` exists in the map.
     *
     * If the key `k` was inserted into the map, `contains` returns
     * true. Otherwise, it returns false. Uses the CUDA Cooperative Groups API to
     * to leverage multiple threads to perform a single `contains` operation. This provides a
     * significant boost in throughput compared to the non Cooperative Group
     * `contains` at moderate to high load factors.
     *
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the contains operation
     * @param k The key to search for
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return A boolean indicating whether the key/value pair
     * containing `k` was inserted
     */
    template <typename KeyEqual = thrust::equal_to<key_type>>
    __device__ __forceinline__ bool contains(
      cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
      Key const& k,
      KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Indicates whether the pair `p` exists in the map.
     *
     * If the pair `p` was inserted into the map, `contains` returns
     * true. Otherwise, it returns false. Uses the CUDA Cooperative Groups API to
     * to leverage multiple threads to perform a single `contains` operation. This provides a
     * significant boost in throughput compared to the non Cooperative Group
     * `contains` at moderate to high load factors.
     *
     * @tparam PairEqual Binary callable type
     * @param g The Cooperative Group used to perform the contains operation
     * @param k The pair to search for
     * @param pair_equal The binary callable used to compare input pair and slot content
     * for equality
     * @return A boolean indicating whether the input pair was inserted in the map
     */
    template <typename PairEqual = thrust::equal_to<value_type>>
    __device__ __forceinline__ bool pair_contains(
      cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
      value_type const& p,
      PairEqual pair_equal = PairEqual{}) noexcept;

    /**
     * @brief Counts the occurrence of a given key contained in multimap.
     *
     * For a given key, `k`, counts all matching keys, `k'`, as determined by `key_equal(k, k')` and
     * returns the sum of all matches for `k`.
     *
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the count operation
     * @param k The key to search for
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return Number of matches found by the current thread
     */
    template <typename KeyEqual = thrust::equal_to<key_type>>
    __device__ __forceinline__ std::size_t count(
      cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
      Key const& k,
      KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Counts the occurrence of a given key contained in multimap. If no
     * matches can be found for a given key, the corresponding occurrence is 1.
     *
     * For a given key, `k`, counts all matching keys, `k'`, as determined by `key_equal(k, k')` and
     * returns the sum of all matches for `k`. If `k` does not have any matches, returns 1.
     *
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the count operation
     * @param k The key to search for
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return Number of matches found by the current thread
     */
    template <typename KeyEqual = thrust::equal_to<key_type>>
    __device__ __forceinline__ std::size_t count_outer(
      cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
      Key const& k,
      KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Counts the occurrence of a given key/value pair contained in multimap.
     *
     * For a given pair, `p`, counts all matching pairs, `p'`, as determined by `pair_equal(p, p')`
     * and returns the sum of all matches for `p`.
     *
     * @tparam PairEqual Binary callable type
     * @param g The Cooperative Group used to perform the pair_count operation
     * @param pair The pair to search for
     * @param pair_equal The binary callable used to compare two pairs
     * for equality
     * @return Number of matches found by the current thread
     */
    template <typename PairEqual>
    __device__ __forceinline__ std::size_t pair_count(
      cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
      value_type const& pair,
      PairEqual pair_equal) noexcept;

    /**
     * @brief Counts the occurrence of a given key/value pair contained in multimap.
     * If no matches can be found for a given key, the corresponding occurrence is 1.
     *
     * For a given pair, `p`, counts all matching pairs, `p'`, as determined by `pair_equal(p, p')`
     * and returns the sum of all matches for `p`. If `p` does not have any matches, returns 1.
     *
     * @tparam PairEqual Binary callable type
     * @param g The Cooperative Group used to perform the pair_count operation
     * @param pair The pair to search for
     * @param pair_equal The binary callable used to compare two pairs
     * for equality
     * @return Number of matches found by the current thread
     */
    template <typename PairEqual>
    __device__ __forceinline__ std::size_t pair_count_outer(
      cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
      value_type const& pair,
      PairEqual pair_equal) noexcept;

    /**
     * @brief Retrieves all the matches of a given key contained in multimap with per-flushing-CG
     * shared memory buffer.
     *
     * For key `k` existing in the map, copies `k` and all associated values to unspecified
     * locations in `[output_begin, output_end)`.
     *
     * @tparam buffer_size Size of the output buffer
     * @tparam FlushingCG Type of Cooperative Group used to flush output buffer
     * @tparam atomicT Type of atomic storage
     * @tparam OutputIt Device accessible output iterator whose `value_type` is
     * constructible from the map's `value_type`
     * @tparam KeyEqual Binary callable type
     * @param flushing_cg The Cooperative Group used to flush output buffer
     * @param probing_cg The Cooperative Group used to retrieve
     * @param k The key to search for
     * @param flushing_cg_counter Pointer to flushing_cg counter
     * @param output_buffer Shared memory buffer of the key/value pair sequence
     * @param num_matches Size of the output sequence
     * @param output_begin Beginning of the output sequence of key/value pairs
     * @param key_equal The binary callable used to compare two keys
     * for equality
     */
    template <uint32_t buffer_size,
              typename FlushingCG,
              typename atomicT,
              typename OutputIt,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ __forceinline__ void retrieve(
      FlushingCG const& flushing_cg,
      cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& probing_cg,
      Key const& k,
      uint32_t* flushing_cg_counter,
      value_type* output_buffer,
      atomicT* num_matches,
      OutputIt output_begin,
      KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Retrieves all the matches of a given key contained in multimap with per-flushing-CG
     * shared memory buffer.
     *
     * For key `k` existing in the map, copies `k` and all associated values to unspecified
     * locations in `[output_begin, output_end)`. If `k` does not have any matches, copies `k` and
     * `empty_value_sentinel()` into the output.
     *
     * @tparam buffer_size Size of the output buffer
     * @tparam FlushingCG Type of Cooperative Group used to flush output buffer
     * @tparam atomicT Type of atomic storage
     * @tparam OutputIt Device accessible output iterator whose `value_type` is
     * constructible from the map's `value_type`
     * @tparam KeyEqual Binary callable type
     * @param flushing_cg The Cooperative Group used to flush output buffer
     * @param probing_cg The Cooperative Group used to retrieve
     * @param k The key to search for
     * @param flushing_cg_counter Pointer to flushing_cg counter
     * @param output_buffer Shared memory buffer of the key/value pair sequence
     * @param num_matches Size of the output sequence
     * @param output_begin Beginning of the output sequence of key/value pairs
     * @param key_equal The binary callable used to compare two keys
     * for equality
     */
    template <uint32_t buffer_size,
              typename FlushingCG,
              typename atomicT,
              typename OutputIt,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ __forceinline__ void retrieve_outer(
      FlushingCG const& flushing_cg,
      cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& probing_cg,
      Key const& k,
      uint32_t* flushing_cg_counter,
      value_type* output_buffer,
      atomicT* num_matches,
      OutputIt output_begin,
      KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Retrieves all the matches of a given pair
     *
     * For pair `p` with `n = pair_count(cg, p, pair_equal)` matching pairs, if `pair_equal(p,
     * slot)` returns true, stores `probe_key_begin[j] = p.first`, `probe_val_begin[j] = p.second`,
     * `contained_key_begin[j] = slot.first`, and `contained_val_begin[j] = slot.second` for an
     * unspecified value of `j` where `0 <= j < n`.
     *
     * Concurrent reads or writes to any of the output ranges results in undefined behavior.
     *
     * Behavior is undefined if the extent of any of the output ranges is less than `n`.
     *
     * @tparam OutputIt1 Device accessible output iterator whose `value_type` is constructible from
     * `pair`'s `Key` type.
     * @tparam OutputIt2 Device accessible output iterator whose `value_type` is constructible from
     * `pair`'s `Value` type.
     * @tparam OutputIt3 Device accessible output iterator whose `value_type` is constructible from
     * the map's `key_type`.
     * @tparam OutputIt4 Device accessible output iterator whose `value_type` is constructible from
     * the map's `mapped_type`.
     * @tparam PairEqual Binary callable type
     * @param probing_cg The Cooperative Group used to retrieve
     * @param pair The pair to search for
     * @param probe_key_begin Beginning of the output sequence of the matched probe keys
     * @param probe_val_begin Beginning of the output sequence of the matched probe values
     * @param contained_key_begin Beginning of the output sequence of the matched contained keys
     * @param contained_val_begin Beginning of the output sequence of the matched contained values
     * @param pair_equal The binary callable used to compare two pairs for equality
     */
    template <typename OutputIt1,
              typename OutputIt2,
              typename OutputIt3,
              typename OutputIt4,
              typename PairEqual>
    __device__ __forceinline__ void pair_retrieve(
      cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& probing_cg,
      value_type const& pair,
      OutputIt1 probe_key_begin,
      OutputIt2 probe_val_begin,
      OutputIt3 contained_key_begin,
      OutputIt4 contained_val_begin,
      PairEqual pair_equal) noexcept;

    /**
     * @brief Retrieves all the matches of a given pair contained in multimap with per-flushing-CG
     * shared memory buffer.
     *
     * For pair `p`, if pair_equal(p, slot[j]) returns true, copies `p` to unspecified locations
     * in `[probe_output_begin, probe_output_end)` and copies slot[j] to unspecified locations in
     * `[contained_output_begin, contained_output_end)`.
     *
     * @tparam buffer_size Size of the output buffer
     * @tparam FlushingCG Type of Cooperative Group used to flush output buffer
     * @tparam atomicT Type of atomic storage
     * @tparam OutputIt1 Device accessible output iterator whose `value_type` is constructible from
     * `InputIt`s `value_type`.
     * @tparam OutputIt2 Device accessible output iterator whose `value_type` is constructible from
     * the map's `value_type`.
     * @tparam PairEqual Binary callable type
     * @param flushing_cg The Cooperative Group used to flush output buffer
     * @param probing_cg The Cooperative Group used to retrieve
     * @param pair The pair to search for
     * @param flushing_cg_counter Pointer to the flushing CG counter
     * @param probe_output_buffer Buffer of the matched probe pair sequence
     * @param contained_output_buffer Buffer of the matched contained pair sequence
     * @param num_matches Size of the output sequence
     * @param probe_output_begin Beginning of the output sequence of the matched probe pairs
     * @param contained_output_begin Beginning of the output sequence of the matched contained
     * pairs
     * @param pair_equal The binary callable used to compare two pairs for equality
     */
    template <uint32_t buffer_size,
              typename FlushingCG,
              typename atomicT,
              typename OutputIt1,
              typename OutputIt2,
              typename PairEqual>
    __device__ __forceinline__ void pair_retrieve(
      FlushingCG const& flushing_cg,
      cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& probing_cg,
      value_type const& pair,
      uint32_t* warp_counter,
      value_type* probe_output_buffer,
      value_type* contained_output_buffer,
      atomicT* num_matches,
      OutputIt1 probe_output_begin,
      OutputIt2 contained_output_begin,
      PairEqual pair_equal) noexcept;

    /**
     * @brief Retrieves all the matches of a given pair
     *
     * For pair `p` with `n = pair_count_outer(cg, p, pair_equal)` matching pairs, if `pair_equal(p,
     * slot)` returns true, stores `probe_key_begin[j] = p.first`, `probe_val_begin[j] = p.second`,
     * `contained_key_begin[j] = slot.first`, and `contained_val_begin[j] = slot.second` for an
     * unspecified value of `j` where `0 <= j < n`. If `p` does not have any matches, stores
     * `probe_key_begin[0] = p.first`, `probe_val_begin[0] = p.second`, `contained_key_begin[0] =
     * empty_key_sentinel`, and `contained_val_begin[0] = empty_value_sentinel`.
     *
     * Concurrent reads or writes to any of the output ranges results in undefined behavior.
     *
     * Behavior is undefined if the extent of any of the output ranges is less than `n`.
     *
     * @tparam OutputIt1 Device accessible output iterator whose `value_type` is constructible from
     * `pair`'s `Key` type.
     * @tparam OutputIt2 Device accessible output iterator whose `value_type` is constructible from
     * `pair`'s `Value` type.
     * @tparam OutputIt3 Device accessible output iterator whose `value_type` is constructible from
     * the map's `key_type`.
     * @tparam OutputIt4 Device accessible output iterator whose `value_type` is constructible from
     * the map's `mapped_type`.
     * @tparam PairEqual Binary callable type
     * @param probing_cg The Cooperative Group used to retrieve
     * @param pair The pair to search for
     * @param probe_key_begin Beginning of the output sequence of the matched probe keys
     * @param probe_val_begin Beginning of the output sequence of the matched probe values
     * @param contained_key_begin Beginning of the output sequence of the matched contained keys
     * @param contained_val_begin Beginning of the output sequence of the matched contained values
     * @param pair_equal The binary callable used to compare two pairs for equality
     */
    template <typename OutputIt1,
              typename OutputIt2,
              typename OutputIt3,
              typename OutputIt4,
              typename PairEqual>
    __device__ __forceinline__ void pair_retrieve_outer(
      cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& probing_cg,
      value_type const& pair,
      OutputIt1 probe_key_begin,
      OutputIt2 probe_val_begin,
      OutputIt3 contained_key_begin,
      OutputIt4 contained_val_begin,
      PairEqual pair_equal) noexcept;

    /**
     * @brief Retrieves all the matches of a given pair contained in multimap with per-flushing-CG
     * shared memory buffer.
     *
     * For pair `p`, if pair_equal(p, slot[j]) returns true, copies `p` to unspecified locations
     * in `[probe_output_begin, probe_output_end)` and copies slot[j] to unspecified locations in
     * `[contained_output_begin, contained_output_end)`. If `p` does not have any matches, copies
     * `p` and a pair of `empty_key_sentinel` and `empty_value_sentinel` into the output.
     *
     * @tparam buffer_size Size of the output buffer
     * @tparam FlushingCG Type of Cooperative Group used to flush output buffer
     * @tparam atomicT Type of atomic storage
     * @tparam OutputIt1 Device accessible output iterator whose `value_type` is constructible from
     * `InputIt`s `value_type`.
     * @tparam OutputIt2 Device accessible output iterator whose `value_type` is constructible from
     * the map's `value_type`.
     * @tparam PairEqual Binary callable type
     * @param flushing_cg The Cooperative Group used to flush output buffer
     * @param probing_cg The Cooperative Group used to retrieve
     * @param pair The pair to search for
     * @param flushing_cg_counter Pointer to the flushing CG counter
     * @param probe_output_buffer Buffer of the matched probe pair sequence
     * @param contained_output_buffer Buffer of the matched contained pair sequence
     * @param num_matches Size of the output sequence
     * @param probe_output_begin Beginning of the output sequence of the matched probe pairs
     * @param contained_output_begin Beginning of the output sequence of the matched contained
     * pairs
     * @param pair_equal The binary callable used to compare two pairs for equality
     */
    template <uint32_t buffer_size,
              typename FlushingCG,
              typename atomicT,
              typename OutputIt1,
              typename OutputIt2,
              typename PairEqual>
    __device__ __forceinline__ void pair_retrieve_outer(
      FlushingCG const& flushing_cg,
      cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& probing_cg,
      value_type const& pair,
      uint32_t* flushing_cg_counter,
      value_type* probe_output_buffer,
      value_type* contained_output_buffer,
      atomicT* num_matches,
      OutputIt1 probe_output_begin,
      OutputIt2 contained_output_begin,
      PairEqual pair_equal) noexcept;

   private:
    using device_view_base<device_view_impl>::impl_;
  };  // class device_view

  /**
   * @brief Return the raw pointer of the hash map slots.
   */
  value_type* raw_slots() noexcept
  {
    // Unsafe access to the slots stripping away their atomic-ness to allow non-atomic access.
    // TODO: to be replace by atomic_ref when it's ready
    return reinterpret_cast<value_type*>(slots_.get());
  }

  /**
   * @brief Return the raw pointer of the hash map slots.
   */
  value_type const* raw_slots() const noexcept
  {
    // Unsafe access to the slots stripping away their atomic-ness to allow non-atomic access.
    // TODO: to be replace by atomic_ref when it's ready
    return reinterpret_cast<value_type const*>(slots_.get());
  }

  /**
   * @brief Gets the maximum number of elements the hash map can hold.
   *
   * @return The maximum number of elements the hash map can hold
   */
  std::size_t get_capacity() const noexcept { return capacity_; }

  /**
   * @brief Gets the number of elements in the hash map.
   *
   * @param stream CUDA stream used to get the number of inserted elements
   * @return The number of elements in the map
   */
  std::size_t get_size(cudaStream_t stream = 0) const noexcept;

  /**
   * @brief Gets the load factor of the hash map.
   *
   * @param stream CUDA stream used to get the load factor
   * @return The load factor of the hash map
   */
  float get_load_factor(cudaStream_t stream = 0) const noexcept;

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
   * @brief Constructs a device_view object based on the members of the `static_multimap`
   * object.
   *
   * @return A device_view object based on the members of the `static_multimap` object
   */
  device_view get_device_view() const noexcept
  {
    return device_view(slots_.get(),
                       capacity_,
                       sentinel::empty_key<Key>{empty_key_sentinel_},
                       sentinel::empty_value<Value>{empty_value_sentinel_});
  }

  /**
   * @brief Constructs a device_mutable_view object based on the members of the
   * `static_multimap` object
   *
   * @return A device_mutable_view object based on the members of the `static_multimap` object
   */
  device_mutable_view get_device_mutable_view() const noexcept
  {
    return device_mutable_view(slots_.get(),
                               capacity_,
                               sentinel::empty_key<Key>{empty_key_sentinel_},
                               sentinel::empty_value<Value>{empty_value_sentinel_});
  }

 private:
  std::size_t capacity_{};                      ///< Total number of slots
  Key empty_key_sentinel_{};                    ///< Key value that represents an empty slot
  Value empty_value_sentinel_{};                ///< Initial value of empty slot
  slot_allocator_type slot_allocator_{};        ///< Allocator used to allocate slots
  counter_allocator_type counter_allocator_{};  ///< Allocator used to allocate counters
  counter_deleter delete_counter_;              ///< Custom counter deleter
  slot_deleter delete_slots_;                   ///< Custom slots deleter
  std::unique_ptr<atomic_ctr_type, counter_deleter> d_counter_{};  ///< Preallocated device counter
  std::unique_ptr<pair_atomic_type, slot_deleter> slots_{};  ///< Pointer to flat slots storage
};                                                           // class static_multimap

}  // namespace cuco

#include <cuco/detail/static_multimap/device_view_impl.inl>
#include <cuco/detail/static_multimap/static_multimap.inl>
