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

#include <cooperative_groups.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/optional.h>
#include <cub/cub.cuh>
#include <memory>

#include <cuco/allocator.hpp>

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11000) && defined(__CUDA_ARCH__) && \
  (__CUDA_ARCH__ >= 700)
#define CUCO_HAS_CUDA_BARRIER
#endif

// cg::memcpy_aysnc is supported for CUDA 11.1 and up
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11100)
#define SUPPORTS_CG_MEMCPY_ASYNC 1
#else
#define SUPPORTS_CG_MEMCPY_ASYNC 0
#endif

#if defined(CUCO_HAS_CUDA_BARRIER)
#include <cuda/barrier>
#endif

#include <cuco/detail/bitwise_compare.cuh>
#include <cuco/detail/error.hpp>
#include <cuco/detail/prime.hpp>
#include <cuco/detail/probe_sequences.cuh>
#include <cuco/detail/static_multimap_kernels.cuh>

namespace cuco {

/**
 * @brief A GPU-accelerated, unordered, associative container of key-value
 * pairs that supports equivalent keys.
 *
 * Allows constant time concurrent inserts or concurrent find operations (not
 * concurrent insert and find) from threads in device code.
 *
 * Current limitations:
 * - Requires keys and values that where `cuco::detail::is_bitwise_comparable<T>::value` is true
 * - Comparisons against the "sentinel" values will always be done with bitwise comparisons
 * Therefore, the objects must have unique, bitwise object representations (e.g., no padding bits).
 * - Does not support erasing keys
 * - Capacity is fixed and will not grow automatically
 * - Requires the user to specify sentinel values for both key and mapped value
 * to indicate empty slots
 * - Does not support concurrent insert and find operations
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
 * `device_view` and `mutable_device_view`. The `device_view` class is an
 * immutable view that allows only non-modifying operations such as `count` or
 * `contains`. The `mutable_device_view` class only allows `insert` operations.
 * The two types are separate to prevent erroneous concurrent insert/find
 * operations.
 *
 * Loading two consecutive slots instead of one for small pairs can improve cache utilization since
 * 16B memory loads are natively supported at the SASS/hardware level. This `vector load` method is
 * implicitly applied to all involved operations (e.g. `insert`, `count`, and `retrieve`, etc.) if
 * the pair is packable (see `multimap::uses_vector_load` logic).
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
 * @tparam Key Arithmetic type used for key
 * @tparam Value Type of the mapped values
 * @tparam ProbeSequence Probe sequence used in multimap
 * @tparam Scope The scope in which insert/find operations will be performed by
 * individual threads.
 * @tparam Allocator Type of allocator used for device storage
 */
template <typename Key,
          typename Value,
          class ProbeSequence      = cuco::double_hashing<Key, Value>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          typename Allocator       = cuco::cuda_allocator<char>>
class static_multimap {
  static_assert(
    detail::is_bitwise_comparable<Key>::value,
    "Key type must have unique object representations or have been explicitly declared as safe for "
    "bitwise comparison via specialization of cuco::detail::is_bitwise_comparable<Key>.");

  static_assert(
    detail::is_bitwise_comparable<Value>::value,
    "Value type must have unique object representations or have been explicitly declared as safe "
    "for bitwise comparison via specialization of cuco::detail::is_bitwise_comparable<Value>.");

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
  using optional_hash_type = thrust::optional<typename detail::MurmurHash3_32<Key>::result_type>;

  static_multimap(static_multimap const&) = delete;
  static_multimap(static_multimap&&)      = delete;
  static_multimap& operator=(static_multimap const&) = delete;
  static_multimap& operator=(static_multimap&&) = delete;

  /**
   * @brief Indicate the warp size.
   *
   * @return The warp size.
   */
  static constexpr uint32_t warp_size() noexcept { return 32u; }

  /**
   * @brief The size of the CUDA cooperative thread group.
   *
   * @return Boolean indicating if concurrent insert/find is supported.
   */
  static constexpr uint32_t cg_size() noexcept { return ProbeSequence::cg_size(); }

  /**
   * @brief Construct a fixed-size map with the specified capacity and sentinel values.
   * @brief Construct a statically sized map with the specified number of slots
   * and sentinel values.
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
                  Key empty_key_sentinel,
                  Value empty_value_sentinel,
                  cudaStream_t stream    = 0,
                  Allocator const& alloc = Allocator{});

  /**
   * @brief Destroys the map and frees its contents.
   *
   */
  ~static_multimap();

  /**
   * @brief Inserts all key/value pairs in the range `[first, last)`.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `value_type`
   * @tparam KeyEqual Binary callable type
   * @param first Beginning of the sequence of key/value pairs
   * @param last End of the sequence of key/value pairs
   * @param stream CUDA stream used for insert
   * @param key_equal The binary function to compare two keys for equality
   */
  template <typename InputIt, typename KeyEqual = thrust::equal_to<key_type>>
  void insert(InputIt first,
              InputIt last,
              cudaStream_t stream = 0,
              KeyEqual key_equal  = KeyEqual{});

  /**
   * @brief Inserts key/value pairs in the range `[first, last)` if `pred` returns true.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `value_type`
   * @tparam Predicate Unary predicate function type
   * @tparam KeyEqual Binary callable type
   * @param first Beginning of the sequence of key/value pairs
   * @param last End of the sequence of key/value pairs
   * @param stream CUDA stream used for insert
   * @param key_equal The binary function to compare two keys for equality
   */
  template <typename InputIt, typename Predicate, typename KeyEqual = thrust::equal_to<key_type>>
  void insert_if(InputIt first,
                 InputIt last,
                 Predicate pred,
                 cudaStream_t stream = 0,
                 KeyEqual key_equal  = KeyEqual{});

  /**
   * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
   *
   * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is
   * convertible to the map's `mapped_type`
   * @tparam KeyEqual Binary callable type
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
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
   * @brief Counts the occurrences of keys in `[first, last)` contained in the multimap.
   *
   * @tparam Input Device accesible input iterator whose `value_type` is convertible to `key_type`
   * @tparam KeyEqual Binary callable
   * @param first Beginning of the sequence of keys to count
   * @param last End of the sequence of keys to count
   * @param stream CUDA stream used for count
   * @param key_equal Binary function to compare two keys for equality
   * @return The sum of total occurrences of all keys in `[first,last)`
   */
  template <typename InputIt, typename KeyEqual = thrust::equal_to<key_type>>
  std::size_t count(InputIt first,
                    InputIt last,
                    cudaStream_t stream = 0,
                    KeyEqual key_equal  = KeyEqual{}) const;

  /**
   * @brief Counts the occurrences of keys in `[first, last)` contained in the multimap. If no
   * matches can be found for a given key, the corresponding occurrence is 1.
   *
   * @tparam Input Device accesible input iterator whose `value_type` is convertible to `key_type`
   * @tparam KeyEqual Binary callable
   * @param first Beginning of the sequence of keys to count
   * @param last End of the sequence of keys to count
   * @param stream CUDA stream used for count_outer
   * @param key_equal Binary function to compare two keys for equality
   * @return The sum of total occurrences of all keys in `[first,last)`
   */
  template <typename InputIt, typename KeyEqual = thrust::equal_to<key_type>>
  std::size_t count_outer(InputIt first,
                          InputIt last,
                          cudaStream_t stream = 0,
                          KeyEqual key_equal  = KeyEqual{}) const;

  /**
   * @brief Counts the occurrences of key/value pairs in `[first, last)` contained in the multimap.
   *
   * @tparam Input Device accesible input iterator of key/value pairs
   * @tparam PairEqual Binary callable
   * @param first Beginning of the sequence of pairs to count
   * @param last End of the sequence of pairs to count
   * @param pair_equal Binary function to compare two pairs for equality
   * @param stream CUDA stream used for pair_count
   * @return The sum of total occurrences of all pairs in `[first,last)`
   */
  template <typename InputIt, typename PairEqual>
  std::size_t pair_count(InputIt first,
                         InputIt last,
                         PairEqual pair_equal,
                         cudaStream_t stream = 0) const;

  /**
   * @brief Counts the occurrences of key/value pairs in `[first, last)` contained in the multimap.
   * If no matches can be found for a given key/value pair, the corresponding occurrence is 1.
   *
   * @tparam Input Device accesible input iterator of key/value pairs
   * @tparam PairEqual Binary callable
   * @param first Beginning of the sequence of pairs to count
   * @param last End of the sequence of pairs to count
   * @param pair_equal Binary function to compare two pairs for equality
   * @param stream CUDA stream used for pair_count_outer
   * @return The sum of total occurrences of all pairs in `[first,last)`
   */
  template <typename InputIt, typename PairEqual>
  std::size_t pair_count_outer(InputIt first,
                               InputIt last,
                               PairEqual pair_equal,
                               cudaStream_t stream = 0) const;

  /**
   * @brief Finds all the values corresponding to all keys in the range `[first, last)`.
   *
   * If the key `k = *(first + i)` exists in the map, copies `k` and all associated values to
   * unspecified locations in `[output_begin, output_end)`. Else, does nothing.
   *
   * Behavior is undefined if the total number of matching keys exceeds `std::distance(output_begin,
   * output_end)`. Use `count()` to determine the number of matching keys.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is
   * convertible to the map's `value_type`
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
   * @brief Finds all the matches corresponding to all keys in the range `[first, last)`.
   *
   * If the key `k = *(first + i)` exists in the map, copies `k` and all associated values to
   * unspecified locations in `[output_begin, output_end)`. Else, copies `k` and the empty value
   * sentinel.
   *
   * Behavior is undefined if the total number of matching keys exceeds `std::distance(output_begin,
   * output_end)`. Use `count_outer()` to determine the number of matching keys.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is
   * convertible to the map's `value_type`
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
   * @brief Finds all pairs matching the input probe pair in the range `[first, last)`.
   *
   * if pair_equal(*(first + i), slot[j]) returns true, then *(first+i) is stored to
   * `probe_output_begin`, and slot[j] is stored to `contained_output_begin`.
   *
   * Behavior is undefined if the total number of matching pairs exceeds
   * `std::distance(probe_output_begin, probe_output_end)` (or
   * `std::distance(contained_output_begin, contained_output_end)`). Use
   * `pair_count()` to determine the number of matching pairs.
   *
   * @tparam InputIt Device accessible input iterator for probe pairs
   * @tparam OutputZipIt1 Device accessible output zip iterator for probe matches
   * @tparam OutputZipIt2 Device accessible output zip iterator for contained matches
   * @tparam PairEqual Binary callable type
   * @param first Beginning of the sequence of pairs
   * @param last End of the sequence of pairs
   * @param probe_output_begin Beginning of the sequence of the matched probe pairs
   * @param contained_output_begin Beginning of the sequence of the matched contained pairs
   * @param pair_equal The binary function to compare two pairs for equality
   * @param stream CUDA stream used for retrieve_outer
   * @return The total number of matches
   */
  template <typename InputIt, typename OutputZipIt1, typename OutputZipIt2, typename PairEqual>
  std::size_t pair_retrieve(InputIt first,
                            InputIt last,
                            OutputZipIt1 probe_output_begin,
                            OutputZipIt2 contained_output_begin,
                            PairEqual pair_equal,
                            cudaStream_t stream = 0) const;

  /**
   * @brief Finds all pairs matching the input probe pair in the range `[first, last)`.
   *
   * if pair_equal(*(first + i), slot[j]) returns true, then *(first+i) is stored to
   * `probe_output_begin`, and slot[j] is stored to `contained_output_begin`. If *(first+i) doesn't
   * have matches in the map, copies *(first + i) in `probe_output_begin` and a pair of
   * `empty_key_sentinel` and `empty_value_sentinel` in `contained_output_begin`.
   *
   * Behavior is undefined if the total number of matching pairs exceeds
   * `std::distance(probe_output_begin, probe_output_end)` (or
   * `std::distance(contained_output_begin, contained_output_end)`). Use
   * `pair_count()` to determine the number of matching pairs.
   *
   * @tparam InputIt Device accessible input iterator for probe pairs
   * @tparam OutputZipIt1 Device accessible output zip iterator for probe matches
   * @tparam OutputZipIt2 Device accessible output zip iterator for contained matches
   * @tparam PairEqual Binary callable type
   * @param first Beginning of the sequence of pairs
   * @param last End of the sequence of pairs
   * @param probe_output_begin Beginning of the sequence of the matched probe pairs
   * @param contained_output_begin Beginning of the sequence of the matched contained pairs
   * @param pair_equal The binary function to compare two pairs for equality
   * @param stream CUDA stream used for retrieve_outer
   * @return The total number of matches
   */
  template <typename InputIt, typename OutputZipIt1, typename OutputZipIt2, typename PairEqual>
  std::size_t pair_retrieve_outer(InputIt first,
                                  InputIt last,
                                  OutputZipIt1 probe_output_begin,
                                  OutputZipIt2 contained_output_begin,
                                  PairEqual pair_equal,
                                  cudaStream_t stream = 0) const;

 private:
  /**
   * @brief Indicate if vector load is used.
   *
   * @return Boolean indicating if vector load is used.
   */
  static constexpr bool uses_vector_load() noexcept { return ProbeSequence::uses_vector_load(); }

  /**
   * @brief The number of elements for each vector load
   *
   * @return The number of elements for each vector load.
   */
  static constexpr uint32_t vector_width() noexcept { return ProbeSequence::vector_width(); }

  class device_view_base {
   protected:
    // Import member type definitions from `static_multimap`
    using value_type     = value_type;
    using key_type       = Key;
    using mapped_type    = Value;
    using iterator       = pair_atomic_type*;
    using const_iterator = pair_atomic_type const*;

    /**
     * @brief Indicate if vector load is used.
     *
     * @return Boolean indicating if vector load is used.
     */
    static constexpr bool uses_vector_load() noexcept { return ProbeSequence::uses_vector_load(); }

    /**
     * @brief The number of elements for each vector load
     *
     * @return The number of elements for each vector load.
     */
    static constexpr uint32_t vector_width() noexcept { return ProbeSequence::vector_width(); }

    /**
     * @brief Load two key/value pairs from the given slot to the target pair array.
     *
     * @param arr The pair array to be loaded
     * @param current_slot The given slot to load from
     */
    __device__ void load_pair_array(value_type* arr, const_iterator current_slot) noexcept;

   private:
    ProbeSequence probe_sequence_;
    Key empty_key_sentinel_{};      ///< Key value that represents an empty slot
    Value empty_value_sentinel_{};  ///< Initial Value of empty slot

   public:
    __host__ __device__ device_view_base(pair_atomic_type* slots,
                                         std::size_t capacity,
                                         Key empty_key_sentinel,
                                         Value empty_value_sentinel) noexcept
      : probe_sequence_{slots, capacity},
        empty_key_sentinel_{empty_key_sentinel},
        empty_value_sentinel_{empty_value_sentinel}
    {
    }

    /**
     * @brief Gets slots array.
     *
     * @return Slots array
     */
    __device__ pair_atomic_type* get_slots() noexcept { return probe_sequence_.get_slots(); }

    /**
     * @brief Gets slots array.
     *
     * @return Slots array
     */
    __device__ pair_atomic_type const* get_slots() const noexcept
    {
      return probe_sequence_.get_slots();
    }

    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * To be used for Cooperative Group based probing.
     *
     * @tparam CG Cooperative Group type
     * @param g the Cooperative Group for which the initial slot is needed
     * @param k The key to get the slot for
     * @return Pointer to the initial slot for `k`
     */
    template <typename CG>
    __device__ iterator initial_slot(CG const& g,
                                     Key const& k,
                                     optional_hash_type precomputed_hash) noexcept
    {
      return probe_sequence_.initial_slot(g, k, precomputed_hash);
    }

    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * To be used for Cooperative Group based probing.
     *
     * @tparam CG Cooperative Group type
     * @param g the Cooperative Group for which the initial slot is needed
     * @param k The key to get the slot for
     * @return Pointer to the initial slot for `k`
     */
    template <typename CG>
    __device__ const_iterator initial_slot(CG g,
                                           Key const& k,
                                           optional_hash_type precomputed_hash) const noexcept
    {
      return probe_sequence_.initial_slot(g, k, precomputed_hash);
    }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot. To
     * be used for Cooperative Group based probing.
     *
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    __device__ iterator next_slot(iterator s) noexcept { return probe_sequence_.next_slot(s); }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot. To
     * be used for Cooperative Group based probing.
     *
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    __device__ const_iterator next_slot(const_iterator s) const noexcept
    {
      return probe_sequence_.next_slot(s);
    }

   public:
    /**
     * @brief Gets the maximum number of elements the hash map can hold.
     *
     * @return The maximum number of elements the hash map can hold
     */
    __host__ __device__ std::size_t get_capacity() const noexcept
    {
      return probe_sequence_.get_capacity();
    }

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
    __host__ __device__ Value get_empty_value_sentinel() const noexcept
    {
      return empty_value_sentinel_;
    }
  };

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
   *                  [map = m.get_mutable_device_view()]
   *                  __device__ (auto i) mutable {
   *                     map.insert(thrust::make_pair(i,i));
   *                  });
   * \endcode
   */
  class device_mutable_view : public device_view_base {
   public:
    using value_type     = typename device_view_base::value_type;
    using key_type       = typename device_view_base::key_type;
    using mapped_type    = typename device_view_base::mapped_type;
    using iterator       = typename device_view_base::iterator;
    using const_iterator = typename device_view_base::const_iterator;

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
    __host__ __device__ device_mutable_view(pair_atomic_type* slots,
                                            std::size_t capacity,
                                            Key empty_key_sentinel,
                                            Value empty_value_sentinel) noexcept
      : device_view_base{slots, capacity, empty_key_sentinel, empty_value_sentinel}
    {
    }

   private:
    /**
     * @brief Enumeration of the possible results of attempting to insert into a hash bucket.
     */
    enum class insert_result {
      CONTINUE,  ///< Insert did not succeed, continue trying to insert
      SUCCESS,   ///< New pair inserted successfully
      DUPLICATE  ///< Insert did not succeed, key is already present
    };

    /**
     * @brief Inserts the specified key/value pair with one single CAS operation.
     *
     * @tparam KeyEqual Binary callable type
     * @param current_slot The slot to insert
     * @param insert_pair The pair to insert
     * @param key_equal The binary callable used to compare two keys for
     * equality
     * @return An insert result from the `insert_resullt` enumeration.
     */
    template <typename KeyEqual>
    __device__ insert_result packed_cas(iterator current_slot,
                                        value_type const& insert_pair,
                                        KeyEqual key_equal) noexcept;

    /**
     * @brief Inserts the specified key/value pair with two back-to-back CAS operations.
     *
     * @tparam KeyEqual Binary callable type
     * @param current_slot The slot to insert
     * @param insert_pair The pair to insert
     * @param key_equal The binary callable used to compare two keys for
     * equality
     * @return An insert result from the `insert_resullt` enumeration.
     */
    template <typename KeyEqual>
    __device__ insert_result back_to_back_cas(iterator current_slot,
                                              value_type const& insert_pair,
                                              KeyEqual key_equal) noexcept;

    /**
     * @brief Inserts the specified key/value pair with a CAS of the key and a dependent write of
     * the value.
     *
     * @tparam KeyEqual Binary callable type
     * @param current_slot The slot to insert
     * @param insert_pair The pair to insert
     * @param key_equal The binary callable used to compare two keys for
     * equality
     * @return An insert result from the `insert_resullt` enumeration.
     */
    template <typename KeyEqual>
    __device__ insert_result cas_dependent_write(iterator current_slot,
                                                 value_type const& insert_pair,
                                                 KeyEqual key_equal) noexcept;

    /**
     * @brief Inserts the specified key/value pair into the map using vector loads.
     *
     * @tparam uses_vector_load Boolean flag indicating whether vector loads are used or not
     * @tparam CG Cooperative Group type
     * @tparam KeyEqual Binary callable type
     *
     * @param g The Cooperative Group that performs the insert
     * @param insert_pair The pair to insert
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param key_equal The binary callable used to compare two keys for
     * equality
     * @return void.
     */
    template <bool uses_vector_load, typename CG, typename KeyEqual = thrust::equal_to<key_type>>
    __device__ std::enable_if_t<uses_vector_load, void> insert_impl(
      CG g,
      value_type const& insert_pair,
      optional_hash_type precomputed_hash,
      KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Inserts the specified key/value pair into the map using scalar loads.
     *
     * @tparam uses_vector_load Boolean flag indicating whether vector loads are used or not
     * @tparam CG Cooperative Group type
     * @tparam KeyEqual Binary callable type
     *
     * @param g The Cooperative Group that performs the insert
     * @param insert_pair The pair to insert
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param key_equal The binary callable used to compare two keys for
     * equality
     * @return void.
     */
    template <bool uses_vector_load, typename CG, typename KeyEqual = thrust::equal_to<key_type>>
    __device__ std::enable_if_t<not uses_vector_load, void> insert_impl(
      CG g,
      value_type const& insert_pair,
      optional_hash_type precomputed_hash,
      KeyEqual key_equal = KeyEqual{}) noexcept;

   public:
    /**
     * @brief Inserts the specified key/value pair into the map.
     *
     * @tparam CG Cooperative Group type
     * @tparam KeyEqual Binary callable type
     *
     * @param g The Cooperative Group that performs the insert
     * @param insert_pair The pair to insert
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param key_equal The binary callable used to compare two keys for
     * equality
     * @return void.
     */
    template <typename CG, typename KeyEqual = thrust::equal_to<key_type>>
    __device__ void insert(CG g,
                           value_type const& insert_pair,
                           optional_hash_type precomputed_hash,
                           KeyEqual key_equal = KeyEqual{}) noexcept;
  };  // class device mutable view

  /**
   * @brief Non-owning view-type that may be used in device code to
   * perform singular find and contains operations for the map.
   *
   * `device_view` is trivially-copyable and is intended to be passed by
   * value.
   *
   */
  class device_view : public device_view_base {
   public:
    using value_type     = typename device_view_base::value_type;
    using key_type       = typename device_view_base::key_type;
    using mapped_type    = typename device_view_base::mapped_type;
    using iterator       = typename device_view_base::iterator;
    using const_iterator = typename device_view_base::const_iterator;

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
                                    Key empty_key_sentinel,
                                    Value empty_value_sentinel) noexcept
      : device_view_base{slots, capacity, empty_key_sentinel, empty_value_sentinel}
    {
    }

    /**
     * @brief Makes a copy of given `device_view` using non-owned memory.
     *
     * This function is intended to be used to create shared memory copies of small static maps,
     * although global memory can be used as well.
     *
     * Example:
     * @code{.cpp}
     * template <typename MapType, int CAPACITY>
     * __global__ void use_device_view(const typename MapType::device_view device_view,
     *                                 map_key_t const* const keys_to_search,
     *                                 map_value_t* const values_found,
     *                                 const size_t number_of_elements)
     * {
     *     const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
     *
     *     __shared__ typename MapType::pair_atomic_type sm_buffer[CAPACITY];
     *
     *     auto g = cg::this_thread_block();
     *
     *     const map_t::device_view sm_static_multimap = device_view.make_copy(g,
     *                                                                    sm_buffer);
     *
     *     for (size_t i = g.thread_rank(); i < number_of_elements; i += g.size())
     *     {
     *         values_found[i] = sm_static_multimap.find(keys_to_search[i])->second;
     *     }
     * }
     * @endcode
     *
     * @tparam CG The type of the cooperative thread group
     * @param g The cooperative thread group used to copy the slots
     * @param source_device_view `device_view` to copy from
     * @param memory_to_use Array large enough to support `capacity` elements. Object does not take
     * the ownership of the memory
     * @return Copy of passed `device_view`
     */
    template <typename CG>
    __device__ static device_view make_copy(CG g,
                                            pair_atomic_type* const memory_to_use,
                                            device_view source_device_view) noexcept
    {
#if defined(CUCO_HAS_CUDA_BARRIER)
      __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
      if (g.thread_rank() == 0) { init(&barrier, g.size()); }
      g.sync();

      cuda::memcpy_async(g,
                         memory_to_use,
                         source_device_view.get_slots(),
                         sizeof(pair_atomic_type) * source_device_view.get_capacity(),
                         barrier);

      barrier.arrive_and_wait();
#else
      pair_atomic_type const* const slots_ptr = source_device_view.get_slots();
      for (std::size_t i = g.thread_rank(); i < source_device_view.get_capacity(); i += g.size()) {
        new (&memory_to_use[i].first)
          atomic_key_type{slots_ptr[i].first.load(cuda::memory_order_relaxed)};
        new (&memory_to_use[i].second)
          atomic_mapped_type{slots_ptr[i].second.load(cuda::memory_order_relaxed)};
      }
      g.sync();
#endif

      return device_view(memory_to_use,
                         source_device_view.get_capacity(),
                         source_device_view.get_empty_key_sentinel(),
                         source_device_view.get_empty_value_sentinel());
    }

   private:
    /**
     * @brief Indicates whether the key `k` was inserted into the map using vector loads.
     *
     * If the key `k` was inserted into the map, find returns
     * true. Otherwise, it returns false. Uses the CUDA Cooperative Groups API to
     * to leverage multiple threads to perform a single contains operation. This provides a
     * significant boost in throughput compared to the non Cooperative Group
     * `contains` at moderate to high load factors.
     *
     * @tparam uses_vector_load Boolean flag indicating whether vector loads are used or not
     * @tparam CG Cooperative Group type
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the contains operation
     * @param k The key to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return A boolean indicating whether the key/value pair
     * containing `k` was inserted
     */
    template <bool uses_vector_load, typename CG, typename KeyEqual = thrust::equal_to<key_type>>
    __device__ std::enable_if_t<uses_vector_load, bool> contains_impl(
      CG g,
      Key const& k,
      optional_hash_type precomputed_hash,
      KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Indicates whether the key `k` was inserted into the map using scalar loads.
     *
     * If the key `k` was inserted into the map, find returns
     * true. Otherwise, it returns false. Uses the CUDA Cooperative Groups API to
     * to leverage multiple threads to perform a single contains operation. This provides a
     * significant boost in throughput compared to the non Cooperative Group
     * `contains` at moderate to high load factors.
     *
     * @tparam uses_vector_load Boolean flag indicating whether vector loads are used or not
     * @tparam CG Cooperative Group type
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the contains operation
     * @param k The key to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return A boolean indicating whether the key/value pair
     * containing `k` was inserted
     */
    template <bool uses_vector_load, typename CG, typename KeyEqual = thrust::equal_to<key_type>>
    __device__ std::enable_if_t<not uses_vector_load, bool> contains_impl(
      CG g,
      Key const& k,
      optional_hash_type precomputed_hash,
      KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Counts the occurrence of a given key contained in multimap using vector loads.
     *
     * @tparam uses_vector_load Boolean flag indicating whether vector loads are used or not
     * @tparam is_outer Boolean flag indicating whether outer join is peformed or not
     * @tparam CG Cooperative Group type
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the count operation
     * @param k The key to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param thread_num_matches Number of matches found by the current thread
     * @param key_equal The binary callable used to compare two keys
     * for equality
     */
    template <bool uses_vector_load,
              bool is_outer,
              typename CG,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ std::enable_if_t<uses_vector_load, void> count_impl(
      CG const& g,
      Key const& k,
      optional_hash_type precomputed_hash,
      std::size_t& thread_num_matches,
      KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Counts the occurrence of a given key contained in multimap using scalar loads.
     *
     * @tparam uses_vector_load Boolean flag indicating whether vector loads are used or not
     * @tparam is_outer Boolean flag indicating whether outer join is peformed or not
     * @tparam CG Cooperative Group type
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the count operation
     * @param k The key to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param thread_num_matches Number of matches found by the current thread
     * @param key_equal The binary callable used to compare two keys
     * for equality
     */
    template <bool uses_vector_load,
              bool is_outer,
              typename CG,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ std::enable_if_t<not uses_vector_load, void> count_impl(
      CG const& g,
      Key const& k,
      optional_hash_type precomputed_hash,
      std::size_t& thread_num_matches,
      KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Counts the occurrence of a given key/value pair contained in multimap using vector
     * loads.
     *
     * @tparam uses_vector_load Boolean flag indicating whether vector loads are used or not
     * @tparam is_outer Boolean flag indicating whether outer join is peformed or not
     * @tparam CG Cooperative Group type
     * @tparam PairEqual Binary callable type
     * @param g The Cooperative Group used to perform the pair_count operation
     * @param pair The pair to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param thread_num_matches Number of matches found by the current thread
     * @param pair_equal The binary callable used to compare two pairs
     * for equality
     */
    template <bool uses_vector_load, bool is_outer, typename CG, typename PairEqual>
    __device__ std::enable_if_t<uses_vector_load, void> pair_count_impl(
      CG const& g,
      value_type const& pair,
      optional_hash_type precomputed_hash,
      std::size_t& thread_num_matches,
      PairEqual pair_equal) noexcept;

    /**
     * @brief Counts the occurrence of a given key/value pair contained in multimap using scalar
     * loads.
     *
     * @tparam uses_vector_load Boolean flag indicating whether vector loads are used or not
     * @tparam is_outer Boolean flag indicating whether outer join is peformed or not
     * @tparam CG Cooperative Group type
     * @tparam PairEqual Binary callable type
     * @param g The Cooperative Group used to perform the pair_count operation
     * @param pair The pair to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param thread_num_matches Number of matches found by the current thread
     * @param pair_equal The binary callable used to compare two pairs
     * for equality
     */
    template <bool uses_vector_load, bool is_outer, typename CG, typename PairEqual>
    __device__ std::enable_if_t<not uses_vector_load, void> pair_count_impl(
      CG const& g,
      value_type const& pair,
      optional_hash_type precomputed_hash,
      std::size_t& thread_num_matches,
      PairEqual pair_equal) noexcept;

    /**
     * @brief Find all the matches of a given key contained in multimap using vector
     * loads with per-warp shared memory buffer.
     *
     * For keys `k = *(first + i)` existing in the map, copies `k` and all associated values to
     * unspecified locations in `[output_begin, output_end)`. In case of non-matches, copies `k` and
     * the empty value sentinel into the output only if `is_outer` is true.
     *
     * @tparam buffer_size Size of the output buffer
     * @tparam is_outer Boolean flag indicating whether outer join is peformed or not
     * @tparam warpT Warp (Cooperative Group) type
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputIt Device accessible output iterator whose `value_type` is
     * convertible to the map's `mapped_type`
     * @tparam KeyEqual Binary callable type
     * @param warp The Cooperative Group (or warp) used to flush output buffer
     * @param g The Cooperative Group used to retrieve
     * @param k The key to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param warp_counter Pointer to the warp counter
     * @param output_buffer Shared memory buffer of the key/value pair sequence
     * @param num_matches Size of the output sequence
     * @param output_begin Beginning of the output sequence of key/value pairs
     * @param key_equal The binary callable used to compare two keys
     * for equality
     */
    template <uint32_t buffer_size,
              bool is_outer,
              typename warpT,
              typename CG,
              typename atomicT,
              typename OutputIt,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ void retrieve_impl(warpT const& warp,
                                  CG const& g,
                                  Key const& k,
                                  optional_hash_type precomputed_hash,
                                  uint32_t* warp_counter,
                                  value_type* output_buffer,
                                  atomicT* num_matches,
                                  OutputIt output_begin,
                                  KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Find all the matches of a given key contained in multimap using scalar
     * loads with per-cg shared memory buffer.
     *
     * For keys `k = *(first + i)` existing in the map, copies `k` and all associated values to
     * unspecified locations in `[output_begin, output_end)`. In case of non-matches, copies `k` and
     * the empty value sentinel into the output only if `is_outer` is true.
     *
     * @tparam cg_size The number of threads in CUDA Cooperative Groups
     * @tparam buffer_size Size of the output buffer
     * @tparam is_outer Boolean flag indicating whether outer join is peformed or not
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputIt Device accessible output iterator whose `value_type` is
     * convertible to the map's `mapped_type`
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to retrieve
     * @param k The key to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param cg_counter Pointer to the CG counter
     * @param output_buffer Shared memory buffer of the key/value pair sequence
     * @param num_matches Size of the output sequence
     * @param output_begin Beginning of the output sequence of key/value pairs
     * @param key_equal The binary callable used to compare two keys
     * for equality
     */
    template <uint32_t cg_size,
              uint32_t buffer_size,
              bool is_outer,
              typename CG,
              typename atomicT,
              typename OutputIt,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ void retrieve_impl(CG const& g,
                                  Key const& k,
                                  optional_hash_type precomputed_hash,
                                  uint32_t* cg_counter,
                                  value_type* output_buffer,
                                  atomicT* num_matches,
                                  OutputIt output_begin,
                                  KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Find all the matches of a given pair contained in multimap using vector
     * loads with per-warp shared memory buffer.
     *
     * For pair `p`, if pair_equal(p, slot[j]) returns true, copies `p` to unspecified locations in
     * `[probe_output_begin, probe_output_end)` and copies slot[j] to unspecified locations in
     * `[contained_output_begin, contained_output_end)`. In case of non-matches, copies `p` and the
     * empty value sentinels into the output only if `is_outer` is true.
     *
     * @tparam buffer_size Size of the output buffer
     * @tparam is_outer Boolean flag indicating whether outer join is peformed or not
     * @tparam warpT Warp (Cooperative Group) type
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputZipIt1 Device accessible output zip iterator for probe matches
     * @tparam OutputZipIt2 Device accessible output zip iterator for contained matches
     * @tparam PairEqual Binary callable type
     * @param warp The Cooperative Group (or warp) used to flush output buffer
     * @param g The Cooperative Group used to retrieve
     * @param pair The pair to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param warp_counter Pointer to the warp counter
     * @param probe_output_buffer Buffer of the matched probe pair sequence
     * @param contained_output_buffer Buffer of the matched contained pair sequence
     * @param num_matches Size of the output sequence
     * @param probe_output_begin Beginning of the output sequence of the matched probe pairs
     * @param contained_output_begin Beginning of the output sequence of the matched contained pairs
     * @param pair_equal The binary callable used to compare two pairs for equality
     */
    template <uint32_t buffer_size,
              bool is_outer,
              typename warpT,
              typename CG,
              typename atomicT,
              typename OutputZipIt1,
              typename OutputZipIt2,
              typename PairEqual>
    __device__ void pair_retrieve_impl(warpT const& warp,
                                       CG const& g,
                                       value_type const& pair,
                                       optional_hash_type precomputed_hash,
                                       uint32_t* warp_counter,
                                       value_type* probe_output_buffer,
                                       value_type* contained_output_buffer,
                                       atomicT* num_matches,
                                       OutputZipIt1 probe_output_begin,
                                       OutputZipIt2 contained_output_begin,
                                       PairEqual pair_equal) noexcept;

    /**
     * @brief Find all the matches of a given pair contained in multimap using scalar
     * loads with per-cg shared memory buffer.
     *
     * For pair `p`, if pair_equal(p, slot[j]) returns true, copies `p` to unspecified locations in
     * `[probe_output_begin, probe_output_end)` and copies slot[j] to unspecified locations in
     * `[contained_output_begin, contained_output_end)`. In case of non-matches, copies `p` and the
     * empty value sentinels into the output only if `is_outer` is true.
     *
     * @tparam cg_size The number of threads in CUDA Cooperative Groups
     * @tparam buffer_size Size of the output buffer
     * @tparam is_outer Boolean flag indicating whether outer join is peformed or not
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputZipIt1 Device accessible output zip iterator for probe matches
     * @tparam OutputZipIt2 Device accessible output zip iterator for contained matches
     * @tparam PairEqual Binary callable type
     * @param g The Cooperative Group used to retrieve
     * @param pair The pair to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param cg_counter Pointer to the CG counter
     * @param probe_output_buffer Buffer of the matched probe pair sequence
     * @param contained_output_buffer Buffer of the matched contained pair sequence
     * @param num_matches Size of the output sequence
     * @param probe_output_begin Beginning of the output sequence of the matched probe pairs
     * @param contained_output_begin Beginning of the output sequence of the matched contained pairs
     * @param pair_equal The binary callable used to compare two pairs for equality
     */
    template <uint32_t cg_size,
              uint32_t buffer_size,
              bool is_outer,
              typename CG,
              typename atomicT,
              typename OutputZipIt1,
              typename OutputZipIt2,
              typename PairEqual>
    __device__ void pair_retrieve_impl(CG const& g,
                                       value_type const& pair,
                                       optional_hash_type precomputed_hash,
                                       uint32_t* cg_counter,
                                       value_type* probe_output_buffer,
                                       value_type* contained_output_buffer,
                                       atomicT* num_matches,
                                       OutputZipIt1 probe_output_begin,
                                       OutputZipIt2 contained_output_begin,
                                       PairEqual pair_equal) noexcept;

   public:
    /**
     * @brief Flushes per-CG buffer into the output sequence using CG memcpy_async.
     *
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputIt Device accessible output iterator whose `value_type` is
     * convertible to the map's `mapped_type`
     * @param g The Cooperative Group used to flush output buffer
     * @param num_outputs Number of valid output in the buffer
     * @param output_buffer Buffer of the key/value pair sequence
     * @param num_matches Size of the output sequence
     * @param output_begin Beginning of the output sequence of key/value pairs
     */
    template <typename CG, typename atomicT, typename OutputIt>
    __inline__ __device__
      std::enable_if_t<thrust::is_contiguous_iterator<OutputIt>::value and SUPPORTS_CG_MEMCPY_ASYNC,
                       void>
      flush_output_buffer(CG const& g,
                          uint32_t const num_outputs,
                          value_type* output_buffer,
                          atomicT* num_matches,
                          OutputIt output_begin) noexcept;

    /**
     * @brief Flushes per-CG buffer into the output sequence.
     *
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputIt Device accessible output iterator whose `value_type` is
     * convertible to the map's `mapped_type`
     * @param g The Cooperative Group used to flush output buffer
     * @param num_outputs Number of valid output in the buffer
     * @param output_buffer Buffer of the key/value pair sequence
     * @param num_matches Size of the output sequence
     * @param output_begin Beginning of the output sequence of key/value pairs
     */
    template <typename CG, typename atomicT, typename OutputIt>
    __inline__ __device__ std::enable_if_t<not(thrust::is_contiguous_iterator<OutputIt>::value and
                                               SUPPORTS_CG_MEMCPY_ASYNC),
                                           void>
    flush_output_buffer(CG const& g,
                        uint32_t const num_outputs,
                        value_type* output_buffer,
                        atomicT* num_matches,
                        OutputIt output_begin) noexcept;

    /**
     * @brief Flushes per-CG buffer into the output sequences.
     *
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputZipIt1 Device accessible output zip iterator for probe pairs
     * @tparam OutputZipIt2 Device accessible output zip iterator for contained pairs
     * @param g The Cooperative Group used to flush output buffer
     * @param num_outputs Number of valid output in the buffer
     * @param probe_output_buffer Buffer of the matched probe pair sequence
     * @param contained_output_buffer Buffer of the matched contained pair sequence
     * @param num_matches Size of the output sequence
     * @param probe_output_begin Beginning of the output sequence of the matched probe pairs
     * @param contained_output_begin Beginning of the output sequence of the matched contained pairs
     */
    template <typename CG, typename atomicT, typename OutputZipIt1, typename OutputZipIt2>
    __inline__ __device__ void flush_output_buffer(CG const& g,
                                                   uint32_t const num_outputs,
                                                   value_type* probe_output_buffer,
                                                   value_type* contained_output_buffer,
                                                   atomicT* num_matches,
                                                   OutputZipIt1 probe_output_begin,
                                                   OutputZipIt2 contained_output_begin) noexcept;

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
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the contains operation
     * @param k The key to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return A boolean indicating whether the key/value pair
     * containing `k` was inserted
     */
    template <typename CG, typename KeyEqual = thrust::equal_to<key_type>>
    __device__ bool contains(CG g,
                             Key const& k,
                             optional_hash_type precomputed_hash,
                             KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Counts the occurrence of a given key contained in multimap.
     *
     * @tparam CG Cooperative Group type
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the count operation
     * @param k The key to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param thread_num_matches Number of matches found by the current thread
     * @param key_equal The binary callable used to compare two keys
     * for equality
     */
    template <typename CG, typename KeyEqual = thrust::equal_to<key_type>>
    __device__ void count(CG const& g,
                          Key const& k,
                          optional_hash_type precomputed_hash,
                          std::size_t& thread_num_matches,
                          KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Counts the occurrence of a given key contained in multimap. If no
     * matches can be found for a given key, the corresponding occurrence is 1.
     *
     * @tparam CG Cooperative Group type
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the count operation
     * @param k The key to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param thread_num_matches Number of matches found by the current thread
     * @param key_equal The binary callable used to compare two keys
     * for equality
     */
    template <typename CG, typename KeyEqual = thrust::equal_to<key_type>>
    __device__ void count_outer(CG const& g,
                                Key const& k,
                                optional_hash_type precomputed_hash,
                                std::size_t& thread_num_matches,
                                KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Counts the occurrence of a given key/value pair contained in multimap.
     *
     * @tparam CG Cooperative Group type
     * @tparam PairEqual Binary callable type
     * @param g The Cooperative Group used to perform the pair_count operation
     * @param pair The pair to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param thread_num_matches Number of matches found by the current thread
     * @param pair_equal The binary callable used to compare two pairs
     * for equality
     */
    template <typename CG, typename PairEqual>
    __device__ void pair_count(CG const& g,
                               value_type const& pair,
                               optional_hash_type precomputed_hash,
                               std::size_t& thread_num_matches,
                               PairEqual pair_equal) noexcept;

    /**
     * @brief Counts the occurrence of a given key/value pair contained in multimap.
     * If no matches can be found for a given key, the corresponding occurrence is 1.
     *
     * @tparam CG Cooperative Group type
     * @tparam PairEqual Binary callable type
     * @param g The Cooperative Group used to perform the pair_count operation
     * @param pair The pair to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param thread_num_matches Number of matches found by the current thread
     * @param pair_equal The binary callable used to compare two pairs
     * for equality
     */
    template <typename CG, typename PairEqual>
    __device__ void pair_count_outer(CG const& g,
                                     value_type const& pair,
                                     optional_hash_type precomputed_hash,
                                     std::size_t& thread_num_matches,
                                     PairEqual pair_equal) noexcept;

    /**
     * @brief Find all the matches of a given key contained in multimap using vector
     * loads with per-warp shared memory buffer.
     *
     * For keys `k = *(first + i)` existing in the map, copies `k` and all associated values to
     * unspecified locations in `[output_begin, output_end)`. In case of non-matches, copies `k` and
     * the empty value sentinel into the output only if `is_outer` is true.
     *
     * @tparam buffer_size Size of the output buffer
     * @tparam warpT Warp (Cooperative Group) type
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputIt Device accessible output iterator whose `value_type` is
     * convertible to the map's `mapped_type`
     * @tparam KeyEqual Binary callable type
     * @param warp The Cooperative Group (or warp) used to flush output buffer
     * @param g The Cooperative Group used to retrieve
     * @param k The key to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param warp_counter Pointer to the warp counter
     * @param output_buffer Shared memory buffer of the key/value pair sequence
     * @param num_matches Size of the output sequence
     * @param output_begin Beginning of the output sequence of key/value pairs
     * @param key_equal The binary callable used to compare two keys
     * for equality
     */
    template <uint32_t buffer_size,
              typename warpT,
              typename CG,
              typename atomicT,
              typename OutputIt,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ void retrieve(warpT const& warp,
                             CG const& g,
                             Key const& k,
                             optional_hash_type precomputed_hash,
                             uint32_t* warp_counter,
                             value_type* output_buffer,
                             atomicT* num_matches,
                             OutputIt output_begin,
                             KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Find all the matches of a given key contained in multimap using vector
     * loads with per-warp shared memory buffer.
     *
     * For keys `k = *(first + i)` existing in the map, copies `k` and all associated values to
     * unspecified locations in `[output_begin, output_end)`. In case of non-matches, copies `k` and
     * the empty value sentinel into the output only if `is_outer` is true.
     *
     * @tparam buffer_size Size of the output buffer
     * @tparam warpT Warp (Cooperative Group) type
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputIt Device accessible output iterator whose `value_type` is
     * convertible to the map's `mapped_type`
     * @tparam KeyEqual Binary callable type
     * @param warp The Cooperative Group (or warp) used to flush output buffer
     * @param g The Cooperative Group used to retrieve
     * @param k The key to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param warp_counter Pointer to the warp counter
     * @param output_buffer Shared memory buffer of the key/value pair sequence
     * @param num_matches Size of the output sequence
     * @param output_begin Beginning of the output sequence of key/value pairs
     * @param key_equal The binary callable used to compare two keys
     * for equality
     */
    template <uint32_t buffer_size,
              typename warpT,
              typename CG,
              typename atomicT,
              typename OutputIt,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ void retrieve_outer(warpT const& warp,
                                   CG const& g,
                                   Key const& k,
                                   optional_hash_type precomputed_hash,
                                   uint32_t* warp_counter,
                                   value_type* output_buffer,
                                   atomicT* num_matches,
                                   OutputIt output_begin,
                                   KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Find all the matches of a given key contained in multimap using scalar
     * loads with per-cg shared memory buffer.
     *
     * For keys `k = *(first + i)` existing in the map, copies `k` and all associated values to
     * unspecified locations in `[output_begin, output_end)`.
     *
     * @tparam cg_size The number of threads in CUDA Cooperative Groups
     * @tparam buffer_size Size of the output buffer
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputIt Device accessible output iterator whose `value_type` is
     * convertible to the map's `mapped_type`
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to retrieve
     * @param k The key to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param cg_counter Pointer to the CG counter
     * @param output_buffer Shared memory buffer of the key/value pair sequence
     * @param num_matches Size of the output sequence
     * @param output_begin Beginning of the output sequence of key/value pairs
     * @param key_equal The binary callable used to compare two keys
     * for equality
     */
    template <uint32_t cg_size,
              uint32_t buffer_size,
              typename CG,
              typename atomicT,
              typename OutputIt,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ void retrieve(CG const& g,
                             Key const& k,
                             optional_hash_type precomputed_hash,
                             uint32_t* cg_counter,
                             value_type* output_buffer,
                             atomicT* num_matches,
                             OutputIt output_begin,
                             KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Find all the matches of a given key contained in multimap using scalar
     * loads with per-cg shared memory buffer.
     *
     * For keys `k = *(first + i)` existing in the map, copies `k` and all associated values to
     * unspecified locations in `[output_begin, output_end)`. In case of non-matches, copies `k` and
     * the empty value sentinel into the output.
     *
     * @tparam cg_size The number of threads in CUDA Cooperative Groups
     * @tparam buffer_size Size of the output buffer
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputIt Device accessible output iterator whose `value_type` is
     * convertible to the map's `mapped_type`
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to retrieve
     * @param k The key to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param cg_counter Pointer to the CG counter
     * @param output_buffer Shared memory buffer of the key/value pair sequence
     * @param num_matches Size of the output sequence
     * @param output_begin Beginning of the output sequence of key/value pairs
     * @param key_equal The binary callable used to compare two keys
     * for equality
     */
    template <uint32_t cg_size,
              uint32_t buffer_size,
              typename CG,
              typename atomicT,
              typename OutputIt,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ void retrieve_outer(CG const& g,
                                   Key const& k,
                                   optional_hash_type precomputed_hash,
                                   uint32_t* cg_counter,
                                   value_type* output_buffer,
                                   atomicT* num_matches,
                                   OutputIt output_begin,
                                   KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Find all the matches of a given pair contained in multimap using vector
     * loads with per-warp shared memory buffer.
     *
     * For pair `p`, if pair_equal(p, slot[j]) returns true, copies `p` to unspecified locations in
     * `[probe_output_begin, probe_output_end)` and copies slot[j] to unspecified locations in
     * `[contained_output_begin, contained_output_end)`.
     *
     * @tparam buffer_size Size of the output buffer
     * @tparam warpT Warp (Cooperative Group) type
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputZipIt1 Device accessible output zip iterator for probe matches
     * @tparam OutputZipIt2 Device accessible output zip iterator for contained matches
     * @tparam PairEqual Binary callable type
     * @param warp The Cooperative Group (or warp) used to flush output buffer
     * @param g The Cooperative Group used to retrieve
     * @param pair The pair to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param warp_counter Pointer to the warp counter
     * @param probe_output_buffer Buffer of the matched probe pair sequence
     * @param contained_output_buffer Buffer of the matched contained pair sequence
     * @param num_matches Size of the output sequence
     * @param probe_output_begin Beginning of the output sequence of the matched probe pairs
     * @param contained_output_begin Beginning of the output sequence of the matched contained pairs
     * @param pair_equal The binary callable used to compare two pairs for equality
     */
    template <uint32_t buffer_size,
              typename warpT,
              typename CG,
              typename atomicT,
              typename OutputZipIt1,
              typename OutputZipIt2,
              typename PairEqual>
    __device__ void pair_retrieve(warpT const& warp,
                                  CG const& g,
                                  value_type const& pair,
                                  optional_hash_type precomputed_hash,
                                  uint32_t* warp_counter,
                                  value_type* probe_output_buffer,
                                  value_type* contained_output_buffer,
                                  atomicT* num_matches,
                                  OutputZipIt1 probe_output_begin,
                                  OutputZipIt2 contained_output_begin,
                                  PairEqual pair_equal) noexcept;

    /**
     * @brief Find all the matches of a given pair contained in multimap using vector
     * loads with per-warp shared memory buffer.
     *
     * For pair `p`, if pair_equal(p, slot[j]) returns true, copies `p` to unspecified locations in
     * `[probe_output_begin, probe_output_end)` and copies slot[j] to unspecified locations in
     * `[contained_output_begin, contained_output_end)`. In case of non-matches, copies `p` and the
     * empty value sentinels into the output.
     *
     * @tparam buffer_size Size of the output buffer
     * @tparam warpT Warp (Cooperative Group) type
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputZipIt1 Device accessible output zip iterator for probe matches
     * @tparam OutputZipIt2 Device accessible output zip iterator for contained matches
     * @tparam PairEqual Binary callable type
     * @param warp The Cooperative Group (or warp) used to flush output buffer
     * @param g The Cooperative Group used to retrieve
     * @param pair The pair to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param warp_counter Pointer to the warp counter
     * @param probe_output_buffer Buffer of the matched probe pair sequence
     * @param contained_output_buffer Buffer of the matched contained pair sequence
     * @param num_matches Size of the output sequence
     * @param probe_output_begin Beginning of the output sequence of the matched probe pairs
     * @param contained_output_begin Beginning of the output sequence of the matched contained pairs
     * @param pair_equal The binary callable used to compare two pairs for equality
     */
    template <uint32_t buffer_size,
              typename warpT,
              typename CG,
              typename atomicT,
              typename OutputZipIt1,
              typename OutputZipIt2,
              typename PairEqual>
    __device__ void pair_retrieve_outer(warpT const& warp,
                                        CG const& g,
                                        value_type const& pair,
                                        optional_hash_type precomputed_hash,
                                        uint32_t* warp_counter,
                                        value_type* probe_output_buffer,
                                        value_type* contained_output_buffer,
                                        atomicT* num_matches,
                                        OutputZipIt1 probe_output_begin,
                                        OutputZipIt2 contained_output_begin,
                                        PairEqual pair_equal) noexcept;

    /**
     * @brief Find all the matches of a given pair contained in multimap using scalar
     * loads with per-cg shared memory buffer.
     *
     * For pair `p`, if pair_equal(p, slot[j]) returns true, copies `p` to unspecified locations in
     * `[probe_output_begin, probe_output_end)` and copies slot[j] to unspecified locations in
     * `[contained_output_begin, contained_output_end)`.
     *
     * @tparam cg_size The number of threads in CUDA Cooperative Groups
     * @tparam buffer_size Size of the output buffer
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputZipIt1 Device accessible output zip iterator for probe matches
     * @tparam OutputZipIt2 Device accessible output zip iterator for contained matches
     * @tparam PairEqual Binary callable type
     * @param g The Cooperative Group used to retrieve
     * @param pair The pair to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param cg_counter Pointer to the CG counter
     * @param probe_output_buffer Buffer of the matched probe pair sequence
     * @param contained_output_buffer Buffer of the matched contained pair sequence
     * @param num_matches Size of the output sequence
     * @param probe_output_begin Beginning of the output sequence of the matched probe pairs
     * @param contained_output_begin Beginning of the output sequence of the matched contained pairs
     * @param pair_equal The binary callable used to compare two pairs for equality
     */
    template <uint32_t cg_size,
              uint32_t buffer_size,
              typename CG,
              typename atomicT,
              typename OutputZipIt1,
              typename OutputZipIt2,
              typename PairEqual>
    __device__ void pair_retrieve(CG const& g,
                                  value_type const& pair,
                                  optional_hash_type precomputed_hash,
                                  uint32_t* cg_counter,
                                  value_type* probe_output_buffer,
                                  value_type* contained_output_buffer,
                                  atomicT* num_matches,
                                  OutputZipIt1 probe_output_begin,
                                  OutputZipIt2 contained_output_begin,
                                  PairEqual pair_equal) noexcept;

    /**
     * @brief Find all the matches of a given pair contained in multimap using scalar
     * loads with per-cg shared memory buffer.
     *
     * For pair `p`, if pair_equal(p, slot[j]) returns true, copies `p` to unspecified locations in
     * `[probe_output_begin, probe_output_end)` and copies slot[j] to unspecified locations in
     * `[contained_output_begin, contained_output_end)`. In case of non-matches, copies `p` and the
     * empty value sentinels into the output.
     *
     * @tparam cg_size The number of threads in CUDA Cooperative Groups
     * @tparam buffer_size Size of the output buffer
     * @tparam CG Cooperative Group type
     * @tparam atomicT Type of atomic storage
     * @tparam OutputZipIt1 Device accessible output zip iterator for probe matches
     * @tparam OutputZipIt2 Device accessible output zip iterator for contained matches
     * @tparam PairEqual Binary callable type
     * @param g The Cooperative Group used to retrieve
     * @param pair The pair to search for
     * @param precomputed_hash Optional value which allows users to specify the precomputed hash
     * value
     * @param cg_counter Pointer to the CG counter
     * @param probe_output_buffer Buffer of the matched probe pair sequence
     * @param contained_output_buffer Buffer of the matched contained pair sequence
     * @param num_matches Size of the output sequence
     * @param probe_output_begin Beginning of the output sequence of the matched probe pairs
     * @param contained_output_begin Beginning of the output sequence of the matched contained pairs
     * @param pair_equal The binary callable used to compare two pairs for equality
     */
    template <uint32_t cg_size,
              uint32_t buffer_size,
              typename CG,
              typename atomicT,
              typename OutputZipIt1,
              typename OutputZipIt2,
              typename PairEqual>
    __device__ void pair_retrieve_outer(CG const& g,
                                        value_type const& pair,
                                        optional_hash_type precomputed_hash,
                                        uint32_t* cg_counter,
                                        value_type* probe_output_buffer,
                                        value_type* contained_output_buffer,
                                        atomicT* num_matches,
                                        OutputZipIt1 probe_output_begin,
                                        OutputZipIt2 contained_output_begin,
                                        PairEqual pair_equal) noexcept;

  };  // class device_view

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
   * @brief Constructs a device_view object based on the members of the `static_multimap` object.
   *
   * @return A device_view object based on the members of the `static_multimap` object
   */
  device_view get_device_view() const noexcept
  {
    return device_view(slots_, capacity_, empty_key_sentinel_, empty_value_sentinel_);
  }

  /**
   * @brief Constructs a device_mutable_view object based on the members of the `static_multimap`
   * object
   *
   * @return A device_mutable_view object based on the members of the `static_multimap` object
   */
  device_mutable_view get_device_mutable_view() const noexcept
  {
    return device_mutable_view(slots_, capacity_, empty_key_sentinel_, empty_value_sentinel_);
  }

 private:
  pair_atomic_type* slots_{nullptr};            ///< Pointer to flat slots storage
  std::size_t capacity_{};                      ///< Total number of slots
  std::size_t size_{};                          ///< Number of keys in map
  Key empty_key_sentinel_{};                    ///< Key value that represents an empty slot
  Value empty_value_sentinel_{};                ///< Initial value of empty slot
  slot_allocator_type slot_allocator_{};        ///< Allocator used to allocate slots
  counter_allocator_type counter_allocator_{};  ///< Allocator used to allocate counters
  atomic_ctr_type* d_counter_;
  cudaStream_t const& stream_;
};  // class static_multimap
}  // namespace cuco

#include <cuco/detail/static_multimap.inl>
