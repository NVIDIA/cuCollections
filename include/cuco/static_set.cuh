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
#include <cuco/detail/error.hpp>
#include <cuco/detail/prime.hpp>
#include <cuco/detail/storage.cuh>
#include <cuco/probe_sequences.cuh>
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

#include <cooperative_groups.h>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace cuco {

/**
 * @brief A GPU-accelerated, unordered, associative container of unique keys.
 *
 * Allows constant time concurrent inserts or concurrent find operations from threads in device
 * code. Concurrent insert/find is allowed only when
 * <tt>static_set<Key>::supports_concurrent_insert_find()</tt> is true.
 *
 * @tparam Key Type used for keys. Requires `cuco::is_bitwise_comparable_v<Key>`
 * @tparam Scope The scope in which multimap operations will be performed by
 * individual threads
 * @tparam KeyEqual Binary callable type used to compare two keys for equality
 * @tparam ProbeSequence Probe sequence chosen between `cuco::detail::linear_probing`
 * and `cuco::detail::double_hashing`. (see `detail/probe_sequences.cuh`)
 * @tparam Allocator Type of allocator used for device storage
 */
template <class Key,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class KeyEqual           = thrust::equal_to<Key>,
          class ProbeSequence      = cuco::linear_probing<2, detail::MurmurHash3_32<Key>>,
          class Allocator          = cuco::cuda_allocator<char>,
          class Storage            = cuco::detail::aos_storage<Key, void, Allocator>>
class static_set {
  static_assert(
    cuco::is_bitwise_comparable_v<Key>,
    "Key type must have unique object representations or have been explicitly declared as safe for "
    "bitwise comparison via specialization of cuco::is_bitwise_comparable_v<Key>.");

  static_assert(
    std::is_base_of_v<cuco::detail::probe_sequence_base<ProbeSequence::cg_size>, ProbeSequence>,
    "ProbeSequence must be a specialization of either cuco::double_hashing or "
    "cuco::linear_probing.");

 public:
  using key_type       = Key;        ///< Key type
  using value_type     = Key;        ///< Key type
  using allocator_type = Allocator;  ///< Allocator type
  using probe_sequence_type =
    detail::probe_sequence<ProbeSequence, Key, Key, Scope>;  ///< Probe scheme type

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
   */
  __host__ __device__ static constexpr uint32_t cg_size = ProbeSequence::cg_size;

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
   * @param alloc Allocator used for allocating device storage
   * @param stream CUDA stream used to initialize the map
   */
  static_set(std::size_t capacity,
             sentinel::empty_key<Key> empty_key_sentinel,
             Allocator const& alloc = Allocator{},
             cudaStream_t stream    = 0);

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
   * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
   *
   * Stores `true` or `false` to `(output + i)` indicating if the key `*(first + i)` exists in the
   * map.
   *
   * ProbeSequence hashers should be callable with both
   * <tt>std::iterator_traits<InputIt>::value_type</tt> and Key type.
   * <tt>std::invoke_result<KeyEqual, std::iterator_traits<InputIt>::value_type, Key></tt> must be
   * well-formed.
   *
   * @tparam InputIt Device accessible input iterator
   * @tparam OutputIt Device accessible output iterator assignable from `bool`
   *
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the output sequence indicating whether each key is present
   * @param stream CUDA stream used for contains
   */
  template <typename InputIt, typename OutputIt>
  void contains(InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream = 0) const;

 private:
  /**
   * @brief Custom deleter for unique pointer of device counter.
   */
  struct counter_deleter {
    counter_deleter(counter_allocator_type& a) : allocator{a} {}

    counter_deleter(counter_deleter const&) = default;

    void operator()(atomic_ctr_type* ptr) { allocator.deallocate(ptr, 1); }

    counter_allocator_type& allocator;
  };

  template <typename ViewImpl>
  class device_view_base {
   protected:
    // Import member type definitions from `static_set`
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
   * cuco::static_set<int,int> m{100'000, -1, -1};
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
    using view_base_type =
      device_view_base<device_mutable_view_impl>;              ///< Base view implementation type
    using value_type  = typename view_base_type::value_type;   ///< Type of key/value pairs
    using key_type    = typename view_base_type::key_type;     ///< Key type
    using mapped_type = typename view_base_type::mapped_type;  ///< Type of the mapped values
    using iterator =
      typename view_base_type::iterator;  ///< Type of the forward iterator to `value_type`
    using const_iterator =
      typename view_base_type::const_iterator;  ///< Type of the forward iterator to `const
                                                ///< value_type`

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
    using view_base_type = device_view_base<device_view_impl>;    ///< Base view implementation type
    using value_type     = typename view_base_type::value_type;   ///< Type of key/value pairs
    using key_type       = typename view_base_type::key_type;     ///< Key type
    using mapped_type    = typename view_base_type::mapped_type;  ///< Type of the mapped values
    using iterator =
      typename view_base_type::iterator;  ///< Type of the forward iterator to `value_type`
    using const_iterator =
      typename view_base_type::const_iterator;  ///< Type of the forward iterator to `const
                                                ///< value_type`

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
     * @brief Indicates whether the key `k` exists in the map.
     *
     * If the key `k` was inserted into the map, `contains` returns
     * true. Otherwise, it returns false. Uses the CUDA Cooperative Groups API to
     * to leverage multiple threads to perform a single `contains` operation. This provides a
     * significant boost in throughput compared to the non Cooperative Group
     * `contains` at moderate to high load factors.
     *
     * ProbeSequence hashers should be callable with both ProbeKey and Key type.
     * `std::invoke_result<KeyEqual, ProbeKey, Key>` must be well-formed.
     *
     * If `key_equal(probe_key, slot_key)` returns true, `hash(probe_key) == hash(slot_key)` must
     * also be true.
     *
     * @tparam ProbeKey Probe key type
     * @tparam KeyEqual Binary callable type
     *
     * @param g The Cooperative Group used to perform the contains operation
     * @param k The key to search for
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return A boolean indicating whether the key/value pair
     * containing `k` was inserted
     */
    template <typename ProbeKey, typename KeyEqual = thrust::equal_to<key_type>>
    __device__ __forceinline__ bool contains(
      cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
      ProbeKey const& k,
      KeyEqual key_equal = KeyEqual{}) const noexcept;

   private:
    using device_view_base<device_view_impl>::impl_;  ///< Implementation detail of `device_view`
  };                                                  // class device_view

  /**
   * @brief Gets the maximum number of elements the hash map can hold.
   *
   * @return The maximum number of elements the hash map can hold
   */
  std::size_t capacity() const noexcept { return capacity_; }

  /**
   * @brief Gets the number of elements in the hash map.
   *
   * @param stream CUDA stream used to get the number of inserted elements
   * @return The number of elements in the map
   */
  std::size_t size(cudaStream_t stream = 0) const noexcept;

  /**
   * @brief Gets the load factor of the hash map.
   *
   * @param stream CUDA stream used to get the load factor
   * @return The load factor of the hash map
   */
  float load_factor(cudaStream_t stream = 0) const noexcept;

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  Key empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

  /**
   * @brief Constructs a device_view object based on the members of the `static_set`
   * object.
   *
   * @return A device_view object based on the members of the `static_set` object
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
   * `static_set` object
   *
   * @return A device_mutable_view object based on the members of the `static_set` object
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
  counter_allocator_type counter_allocator_{};  ///< Allocator used to allocate counters
  counter_deleter delete_counter_;              ///< Custom counter deleter
  std::unique_ptr<atomic_ctr_type, counter_deleter> d_counter_{};  ///< Preallocated device counter
  std::unique_ptr<Storage> storage_{};                             ///< Pointer to flat slot storage
};

}  // namespace cuco

#include <cuco/detail/static_set/device_view_impl.inl>
#include <cuco/detail/static_set/static_set.inl>
