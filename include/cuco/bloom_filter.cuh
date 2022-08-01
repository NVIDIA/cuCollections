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
#include <cuco/detail/__config>
#include <cuco/detail/bloom_filter_kernels.cuh>
#include <cuco/detail/error.hpp>
#include <cuco/detail/hash_functions.cuh>
#include <cuco/detail/utils.hpp>

#include <cooperative_groups.h>

#include <cuda/std/atomic>

#if defined(CUCO_HAS_CUDA_BARRIER)
#include <cuda/barrier>
#endif

#if defined(CUCO_HAS_CUDA_ANNOTATED_PTR)
#include <cuda/annotated_ptr>
#endif

#include <cstddef>
#include <memory>

namespace cuco {

/**
 * @brief A GPU-accelerated, filter for approximate set membership queries.
 *
 * Allows constant time concurrent inserts or concurrent find operations from threads in device
 * code.
 *
 * Current limitations:
 * - Does not support erasing keys
 * - Capacity is fixed and will not grow automatically
 *
 * The `bloom_filter` supports two types of operations:
 * - Host-side "bulk" operations
 * - Device-side "singular" operations
 *
 * The host-side bulk operations include `insert` and `contains`. These
 * APIs should be used when there are a large number of keys to insert or lookup
 * in the map. For example, given a range of keys specified by device-accessible
 * iterators, the bulk `insert` function will insert all keys into the map.
 *
 * The singular device-side operations allow individual threads to perform
 * independent insert or contains operations from device code. These
 * operations are accessed through non-owning, trivially copyable "view" types:
 * `device_view` and `mutable_device_view`. The `device_view` class is an
 * immutable view that allows only non-modifying operations such as `contains`.
 * The `mutable_device_view` class only allows `insert` operations.
 * The two types are separate to prevent erroneous concurrent 'insert'/'contains'
 * operations.
 *
 * Example:
 * \code{.cpp}
 * // TODO
 * \endcode
 *
 *
 * @tparam Key Arithmetic type used for key
 * @tparam Scope The scope in which insert/find operations will be performed by
 * individual threads.
 * @tparam Allocator Type of allocator used for device storage
 * @tparam Slot Type of bloom filter partition
 */
template <typename Key,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          typename Allocator       = cuco::cuda_allocator<char>,
          typename Slot            = std::uint64_t>
class bloom_filter {
 public:
  using key_type            = Key;                             ///< Key type
  using slot_type           = Slot;                            ///< Filter slot type
  using atomic_slot_type    = cuda::atomic<slot_type, Scope>;  ///< Filter slot type
  using iterator            = atomic_slot_type*;               ///< Filter slot iterator type
  using const_iterator      = atomic_slot_type const*;         ///< Filter slot const iterator type
  using allocator_type      = Allocator;                       ///< Allocator type
  using slot_allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<
    atomic_slot_type>;  ///< Type of the allocator to (de)allocate slots

#if !defined(CUCO_HAS_INDEPENDENT_THREADS)
  static_assert(atomic_slot_type::is_always_lock_free,
                "A slot type larger than 8B is supported for only sm_70 and up.");
#endif

  bloom_filter(bloom_filter const&) = delete;
  bloom_filter(bloom_filter&&)      = delete;
  bloom_filter& operator=(bloom_filter const&) = delete;
  bloom_filter& operator=(bloom_filter&&) = delete;

  /**
   * @brief Construct a fixed-size filter with the specified number of bits.
   *
   * @param num_bits The total number of bits in the filter
   * @param num_hashes The number of hashes to be applied to a key
   * @param alloc Allocator used for allocating device storage
   * @param stream The CUDA stream this operation is executed in
   */
  bloom_filter(std::size_t num_bits,
               std::size_t num_hashes,
               Allocator const& alloc = Allocator{},
               cudaStream_t stream    = 0);

  /**
   * @brief Destroys the filter and frees its contents.
   *
   */
  ~bloom_filter();

  /**
   * @brief (Re-) initializes the filter, i.e., set all bits to 0.
   *
   * @param stream The CUDA stream this operation is executed in
   */
  void initialize(cudaStream_t stream = 0);

  /**
   * @brief Inserts all keys in the range `[first, last)`.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the filter's `key_type`
   * @tparam Hash1 Unary callable type
   * @tparam Hash2 Unary callable type
   * @tparam Hash3 Unary callable type
   * @tparam KeyEqual Binary callable type
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param stream The CUDA stream this operation is executed in
   * @param hash1 First hash function; used to determine a filter slot
   * @param hash2 Second hash function; used to generate a signature of the key
   * @param hash3 Third hash function; used to generate a signature of the key
   */
  template <typename InputIt,
            typename Hash1 = cuco::detail::MurmurHash3_32<key_type>,
            typename Hash2 = Hash1,
            typename Hash3 = Hash2>
  void insert(InputIt first,
              InputIt last,
              cudaStream_t stream = 0,
              Hash1 hash1         = Hash1{},
              Hash2 hash2         = Hash2{1},
              Hash3 hash3         = Hash3{2});

  /**
   * @brief Indicates whether the keys in the range `[first, last)` are
   * contained in the filter.
   *
   * Writes a `bool` to `(output + i)` indicating if the signature of key
   * `*(first + i)` is present in the filter.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the filter's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is
   * convertible to `bool`
   * @tparam Hash1 Unary callable type
   * @tparam Hash2 Unary callable type
   * @tparam Hash3 Unary callable type
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param stream The CUDA stream this operation is executed in
   * @param hash1 First hash function; used to determine a filter slot
   * @param hash2 Second hash function; used to generate a signature of the key
   * @param hash3 Third hash function; used to generate a signature of the key
   */
  template <typename InputIt,
            typename OutputIt,
            typename Hash1 = cuco::detail::MurmurHash3_32<key_type>,
            typename Hash2 = Hash1,
            typename Hash3 = Hash2>
  void contains(InputIt first,
                InputIt last,
                OutputIt output_begin,
                cudaStream_t stream = 0,
                Hash1 hash1         = Hash1{},
                Hash2 hash2         = Hash2{1},
                Hash3 hash3         = Hash3{2});

  /**
   * @brief Gets slots array.
   *
   * @return Slots array
   */
  iterator get_slots() noexcept { return slots_; }

  /**
   * @brief Gets slots array.
   *
   * @return Slots array
   */
  const_iterator get_slots() const noexcept { return slots_; }

  /**
   * @brief Gets the total number of bits in the filter (rounded up to the
   * next multiple of block size).
   *
   * @return The total number of bits in the filter.
   */
  std::size_t get_num_bits() const noexcept { return num_bits_; }

  /**
   * @brief Gets the total number of slots in the filter.
   *
   * @return The total number of slots in the filter.
   */
  std::size_t get_num_slots() const noexcept { return num_slots_; }

  /**
   * @brief Gets the number of hashes to apply to a key.
   *
   * @return The number of hashes to apply to a key.
   */
  std::size_t get_num_hashes() const noexcept { return num_hashes_; }

 private:
  class device_view_base {
   protected:
    // Import member type definitions from `bloom_filter`
    using key_type         = Key;                      ///< Key type
    using slot_type        = slot_type;                ///< Filter slot type
    using atomic_slot_type = atomic_slot_type;         ///< Filter slot type
    using iterator         = atomic_slot_type*;        ///< Filter slot iterator type
    using const_iterator   = atomic_slot_type const*;  ///< Filter slot const iterator type

   private:
    atomic_slot_type* slots_{};  ///< Pointer to flat slots storage
    std::size_t num_bits_{};     ///< Total number of bits
    std::size_t num_slots_{};    ///< Total number of slots
    std::size_t num_hashes_{};   ///< Number of hashes to apply

   protected:
    __host__ __device__ device_view_base(atomic_slot_type* slots,
                                         std::size_t num_bits,
                                         std::size_t num_hashes) noexcept
      : slots_{slots},
        num_bits_{SDIV(num_bits, detail::type_bits<slot_type>()) * detail::type_bits<slot_type>()},
        num_slots_{SDIV(num_bits, detail::type_bits<slot_type>())},
        num_hashes_{num_hashes}
    {
    }

    /**
     * @brief Returns the slot for a given key `k`
     *
     * @tparam Hash Unary callable type
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the slot for `k`
     */
    template <typename Hash>
    __device__ iterator key_slot(Key const& k, Hash hash) noexcept
    {
      return &slots_[hash(k) % num_slots_];
    }

    /**
     * @brief Returns the slot for a given key `k`
     *
     * @tparam Hash Unary callable type
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the slot for `k`
     */
    template <typename Hash>
    __device__ const_iterator key_slot(Key const& k, Hash hash) const noexcept
    {
      return &slots_[hash(k) % num_slots_];
    }

    /**
     * @brief Returns the bit pattern for a given key `k`
     *
     * @tparam Hash1 Unary callable type
     * @tparam Hash2 Unary callable type
     * @param k The key to calculate the pattern for
     * @param hash1 First hash function; used to generate a signature of the key
     * @param hash2 Second hash function; used to generate a signature of the key
     * @return Bit pattern for key `k`
     */
    template <typename Hash1, typename Hash2>
    __device__ slot_type key_pattern(Key const& k, Hash1 hash1, Hash2 hash2) const noexcept;

    /**
     * @brief Initializes the given array of slots using the threads in the group `g`.
     *
     * @note This function synchronizes the group `g`.
     *
     * @tparam CG The type of the cooperative thread group
     * @param g The cooperative thread group used to initialize the slots
     * @param slots Pointer to the array of slots to initialize
     * @param num_slots Number of slots to initialize
     */
    template <typename CG>
    __device__ static void initialize_slots(CG g, atomic_slot_type* slots, std::size_t num_bits)
    {
      auto num_slots = SDIV(num_bits, detail::type_bits<slot_type>());
      auto tid       = g.thread_rank();
      while (tid < num_slots) {
        new (&slots[tid]) atomic_slot_type{0};
        tid += g.size();
      }
      g.sync();
    }

   public:
    /**
     * @brief Gets slots array.
     *
     * @return Slots array
     */
    __host__ __device__ iterator get_slots() noexcept { return slots_; }

    /**
     * @brief Gets slots array.
     *
     * @return Slots array
     */
    __host__ __device__ const_iterator get_slots() const noexcept { return slots_; }

    /**
     * @brief Gets the total number of bits in the filter (rounded up to the
     * next multiple of block size).
     *
     * @return The total number of bits in the filter.
     */
    __host__ __device__ std::size_t get_num_bits() const noexcept { return num_bits_; }

    /**
     * @brief Gets the total number of slots in the filter.
     *
     * @return The total number of slots in the filter.
     */
    __host__ __device__ std::size_t get_num_slots() const noexcept { return num_slots_; }

    /**
     * @brief Gets the number of hashes to apply to a key.
     *
     * @return The number of hashes to apply to a key.
     */
    __host__ __device__ std::size_t get_num_hashes() const noexcept { return num_hashes_; }

    /**
     * @brief Returns iterator to the first slot.
     *
     * @note Unlike `std::map::begin()`, the `begin_slot()` iterator does _not_ point to the first
     * occupied slot. Instead, it refers to the first slot in the array of contiguous slot storage.
     * Iterating from `begin_slot()` to `end_slot()` will iterate over all slots.
     *
     * There is no `begin()` iterator to avoid confusion.
     *
     * @return Iterator to the first slot
     */
    __device__ iterator begin_slot() noexcept { return slots_; }

    /**
     * @brief Returns iterator to the first slot.
     *
     * @note Unlike `std::map::begin()`, the `begin_slot()` iterator does _not_ point to the first
     * occupied slot. Instead, it refers to the first slot in the array of contiguous slot storage.
     * Iterating from `begin_slot()` to `end_slot()` will iterate over all slots.
     *
     * There is no `begin()` iterator to avoid confusion.
     *
     * @return Iterator to the first slot
     */
    __device__ const_iterator begin_slot() const noexcept { return slots_; }

    /**
     * @brief Returns a const_iterator to one past the last slot.
     *
     * @return A const_iterator to one past the last slot
     */
    __host__ __device__ const_iterator end_slot() const noexcept { return slots_ + num_slots_; }

    /**
     * @brief Returns an iterator to one past the last slot.
     *
     * @return An iterator to one past the last slot
     */
    __host__ __device__ iterator end_slot() noexcept { return slots_ + num_slots_; }
  };

 public:
  /**
   * @brief Mutable, non-owning view-type that may be used in device code to
   * perform singular inserts into the filter.
   *
   * `device_mutable_view` is trivially-copyable and is intended to be passed by
   * value.
   *
   * Example:
   * \code{.cpp}
   * cuco::static_map<int> m{100'000, -6};
   *
   * // Inserts a sequence of keys {0, 1, 2, 3}
   * thrust::for_each(thrust::make_counting_iterator(0),
   *                  thrust::make_counting_iterator(50'000),
   *                  [filter = bf.get_mutable_device_view()]
   *                  __device__ (auto i) mutable {
   *                     filter.insert(i);
   *                  });
   * \endcode
   */
  class device_mutable_view : public device_view_base {
   public:
    // Import member type definitions from `bloom_filter`
    using key_type         = Key;                      ///< Key type
    using slot_type        = slot_type;                ///< Filter slot type
    using atomic_slot_type = atomic_slot_type;         ///< Filter slot type
    using iterator         = atomic_slot_type*;        ///< Filter slot iterator type
    using const_iterator   = atomic_slot_type const*;  ///< Filter slot const iterator type

    /**
     * @brief Construct a mutable view of the array pointed to by `slots`.
     *
     * @param slots Pointer to beginning of the initialized slots array
     * @param num_bits The total number of bits in the filter
     * @param num_hashes The number of hashes to be applied to a key
     */
    __host__ __device__ device_mutable_view(atomic_slot_type* slots,
                                            std::size_t num_bits,
                                            std::size_t num_hashes) noexcept
      : device_view_base{slots, num_bits, num_hashes}
    {
    }

   public:
    /**
     * @brief Construct a mutable view of the array pointed to by `slots` and
     * initializes the slot array.
     *
     * @tparam CG Type of the cooperative group this operation is executed with
     * @param g Cooperative group this operation is executed with
     * @param slots Pointer to beginning of the array used for slot storage
     * @param num_bits The total number of bits in the filter
     * @param num_hashes The number of hashes to be applied to a key
     * @return A device_mutable_view object based on the given parameters
     */
    template <typename CG>
    __device__ static device_mutable_view make_from_uninitialized_slots(
      CG g, void* const slots, std::size_t num_bits, std::size_t num_hashes) noexcept
    {
      device_view_base::initialize_slots(g, reinterpret_cast<atomic_slot_type*>(slots), num_bits);
      return device_mutable_view{reinterpret_cast<atomic_slot_type*>(slots), num_bits, num_hashes};
    }

    /**
     * @brief Inserts the specified key into the filter.
     *
     * Returns a `bool` denoting whether the key's signature was not already
     * present in the slot.
     *
     * @tparam Hash1 Unary callable type
     * @tparam Hash2 Unary callable type
     * @tparam Hash3 Unary callable type
     * @param key The key to insert
     * @param hash1 First hash function; used to determine a filter slot
     * @param hash2 Second hash function; used to generate a signature of the key
     * @param hash3 Third hash function; used to generate a signature of the key
     * @return `true` if the pattern was not already in the filter,
     * `false` otherwise.
     */
    template <typename Hash1 = cuco::detail::MurmurHash3_32<key_type>,
              typename Hash2 = Hash1,
              typename Hash3 = Hash2>
    __device__ bool insert(key_type const& key,
                           Hash1 hash1 = Hash1{},
                           Hash2 hash2 = Hash2{1},
                           Hash3 hash3 = Hash3{2}) noexcept;
  };  // class device mutable view

  /**
   * @brief Non-owning view-type that may be used in device code to
   * perform singular find and contains operations for the filter.
   *
   * `device_view` is trivially-copyable and is intended to be passed by
   * value.
   *
   */
  class device_view : public device_view_base {
   public:
    // Import member type definitions from `bloom_filter`
    using key_type         = Key;                      ///< Key type
    using slot_type        = slot_type;                ///< Filter slot type
    using atomic_slot_type = atomic_slot_type;         ///< Filter slot type
    using iterator         = atomic_slot_type*;        ///< Filter slot iterator type
    using const_iterator   = atomic_slot_type const*;  ///< Filter slot const iterator type

    /**
     * @brief Construct a mutable view of the array pointed to by `slots`.
     *
     * @param slots Pointer to beginning of the initialized slots array
     * @param num_bits The total number of bits in the filter
     * @param num_hashes The number of hashes to be applied to a key
     */
    __host__ __device__ device_view(atomic_slot_type* slots,
                                    std::size_t num_bits,
                                    std::size_t num_hashes) noexcept
      : device_view_base{slots, num_bits, num_hashes}
    {
    }

    /**
     * @brief Construct a `device_view` from a `device_mutable_view` object
     *
     * @param mutable_filter object of type `device_mutable_view`
     */
    __host__ __device__ explicit device_view(device_mutable_view mutable_filter)
      : device_view_base{mutable_filter.get_slots(),
                         mutable_filter.get_num_bits(),
                         mutable_filter.get_num_hashes()}
    {
    }

    /**
     * @brief Makes a copy of given `device_view` using non-owned memory.
     *
     * This function is intended to be used to create shared memory copies of
     * small static filters, although global memory can be used as well.
     *
     * Example:
     * @code{.cpp}
     * //TODO
     * @endcode
     *
     * @tparam CG The type of the cooperative thread group
     * @param g The cooperative thread group used to copy the slots
     * @param source_device_view `device_view` to copy from
     * @param memory_to_use Array large enough to support `num_slots` slots.
     * Object does not take the ownership of the memory
     * @return Copy of passed `device_view`
     */
    template <typename CG>
    __device__ static device_view make_copy(CG g,
                                            void* const memory_to_use,
                                            device_view source_device_view) noexcept
    {
      atomic_slot_type* const dest_slots      = reinterpret_cast<atomic_slot_type*>(memory_to_use);
      atomic_slot_type const* const src_slots = source_device_view.get_slots();

#if defined(CUDA_HAS_CUDA_BARRIER)
      __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
      if (g.thread_rank() == 0) { init(&barrier, g.size()); }
      g.sync();

      cuda::memcpy_async(g,
                         dest_slots,
                         src_slots,
                         sizeof(atomic_slot_type) * source_device_view.get_num_slots(),
                         barrier);

      barrier.arrive_and_wait();
#else
      for (std::size_t i = g.thread_rank(); i < source_device_view.get_num_slots(); i += g.size()) {
        new (&dest_slots[i]) atomic_slot_type{src_slots[i].load(cuda::memory_order_relaxed)};
      }
      g.sync();
#endif

      return device_view(
        dest_slots, source_device_view.get_num_bits(), source_device_view.get_num_hashes());
    }

    /**
     * @brief Indicates whether the key's signature is present in the filter.
     *
     * If the siganture of the key `k` was inserted into the filter, `contains`
     * returns `true`. Otherwise, it returns `false`.
     *
     * @tparam Hash1 Unary callable type
     * @tparam Hash2 Unary callable type
     * @tparam Hash3 Unary callable type
     * @param k The key to search for
     * @param hash1 First hash function; used to determine a filter slot
     * @param hash2 Second hash function; used to generate a signature of the key
     * @param hash3 Third hash function; used to generate a signature of the key
     * @return A boolean indicating whether the key's signature is present in
     * the filter.
     */
    template <typename Hash1 = cuco::detail::MurmurHash3_32<key_type>,
              typename Hash2 = Hash1,
              typename Hash3 = Hash2>
    __device__ bool contains(Key const& k,
                             Hash1 hash1 = Hash1{},
                             Hash2 hash2 = Hash2{1},
                             Hash3 hash3 = Hash3{2}) const noexcept;
  };

  /**
   * @brief Constructs a device_view object based on the members of the `bloom_filter` object.
   *
   * @return A device_view object based on the members of the `bloom_filter` object
   */
  device_view get_device_view() const noexcept
  {
    return device_view(slots_, num_bits_, num_hashes_);
  }

  /**
   * @brief Constructs a device_mutable_view object based on the members of the `bloom_filter`
   * object
   *
   * @return A device_mutable_view object based on the members of the `bloom_filter` object
   */
  device_mutable_view get_device_mutable_view() const noexcept
  {
    return device_mutable_view(slots_, num_bits_, num_hashes_);
  }

 private:
  atomic_slot_type* slots_{nullptr};      ///< Pointer to flat slot storage
  std::size_t num_bits_{};                ///< Total number of bits in the filter
  std::size_t num_slots_{};               ///< Total number of slots in the filter
  std::size_t num_hashes_{};              ///< Number of hash functions to apply (k)
  slot_allocator_type slot_allocator_{};  ///< Allocator used to allocate slots
};
}  // namespace cuco

#include <cuco/detail/bloom_filter.inl>
