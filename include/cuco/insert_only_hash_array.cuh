/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>

#include <cu_collections/hash_functions.cuh>
#include <cuco/detail/error.hpp>
#include <cuda/std/atomic>
#include <simt/atomic>

namespace cuco {

/**
 * @brief Gives a power of 2 value equal to or greater than `v`.
 *
 */
constexpr std::size_t next_pow2(std::size_t v) noexcept
{
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return ++v;
}

/**
 * @brief Gives value to use as alignment for a pair type that is at least the
 * size of the sum of the size of the first type and second type, or 16,
 * whichever is smaller.
 */
template <typename First, typename Second>
constexpr std::size_t pair_alignment()
{
  return std::min(std::size_t{16}, next_pow2(sizeof(First) + sizeof(Second)));
}

/**
 * @brief Custom pair type
 *
 * This is necessary because `thrust::pair` is under aligned.
 *
 * @tparam First
 * @tparam Second
 */
template <typename First, typename Second>
struct alignas(pair_alignment<First, Second>()) pair {
  using first_type  = First;
  using second_type = Second;
  First first{};
  Second second{};
  pair() = default;
  __host__ __device__ constexpr pair(First f, Second s) noexcept : first{f}, second{s} {}

  /**
   * @brief Implicit constructor from thrust::tuple<First,Second>
   *
   * @param t
   * @return __host__ constexpr pair
   */
  __host__ __device__ constexpr pair(thrust::tuple<First, Second> const& t) noexcept
    : first{thrust::get<0>(t)}, second{thrust::get<1>(t)}
  {
  }
};

template <typename First, typename Second>
__host__ __device__ bool operator==(pair<First, Second> const& lhs, pair<First, Second> const& rhs)
{
  return thrust::tie(lhs.first, lhs.second) == thrust::tie(rhs.first, rhs.second);
}

template <typename K, typename V>
using pair_type = cuco::pair<K, V>;

template <typename F, typename S>
__host__ __device__ pair_type<F, S> make_pair(F f, S s)
{
  return pair_type<F, S>{f, s};
}

namespace detail {
template <typename Key, typename Value, cuda::thread_scope Scope>
__global__ void initialize(
  pair_type<cuda::atomic<Key, Scope>, cuda::atomic<Value, Scope>>* const __restrict__ slots,
  Key k,
  Value v,
  std::size_t size)
{
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < size) {
    new (&slots[tid].first) cuda::atomic<Key, Scope>{k};
    new (&slots[tid].second) cuda::atomic<Value, Scope>{v};
    tid += gridDim.x * blockDim.x;
  }
}

}  // namespace detail

template <typename Key, typename Value, cuda::thread_scope Scope = cuda::thread_scope_system>
class insert_only_hash_array {
  // TODO: (JH) What static_assert(s) should we have on Key/Value?
  // is_literal_type? (deprecated in C++17, removed c++20)
  // std::has_unique_object_representations? (C++17)
  // probably is_arithmetic
  // document padding bits sharp edges
  static_assert(std::is_arithmetic<Key>::value, "Unsupported, non-arithmetic type.");

 public:
  // TODO: Should be `pair_type<const Key, Value>` but then we can't CAS it
  using value_type         = cuco::pair_type<Key, Value>;
  using key_type           = Key;
  using mapped_type        = Value;
  using atomic_key_type    = cuda::atomic<key_type, Scope>;
  using atomic_mapped_type = cuda::atomic<mapped_type, Scope>;
  using pair_atomic_type   = cuco::pair_type<atomic_key_type, atomic_mapped_type>;

 public:
  /**
   * @brief Construct a new `insert_only_hash_array` with the specified
   * capacity, empty key and mapped value sentinels
   *
   * A `insert_only_hash_array` will be constructed with sufficient "slots"
   * capable of holding `capacity` key/value pairs. The `empty_key_sentinel` and
   * `empty_value_sentinel` are used to indicate an empty slot. Attempting to insert or find a
   * key equal to `empty_key_sentinel` results in undefined behavior. Likewise, attempting to insert
   * a value equal to `empty_value_sentinel` with any associated key results in undefined behavior.
   *
   * Attempting to insert a number of keys beyond `capacity` results in undefined behavior. For best
   * performance, the number of keys inserted should be ~2/3 the `capacity` to reduce the number of
   * collisions.
   *
   * @param capacity The maximum number of key/value pairs that can be inserted
   * @param empty_key_sentinel Key value used to indicate an empty slot.
   * Undefined behavior results from trying to insert or find a key equal to
   * `empty_key_sentinel`.
   * @param empty_value_sentinel Mapped value used to indicate an empty slot. Undefined behavior
   * results from trying to insert/find a mapped value equal to `empty_value_sentinel`.
   */
  explicit insert_only_hash_array(std::size_t capacity,
                                  Key empty_key_sentinel,
                                  Value empty_value_sentinel = Value{})
    : capacity_{capacity},
      empty_key_sentinel_{empty_key_sentinel},
      empty_value_sentinel_{empty_value_sentinel}
  {
    // vector_base? uninitialized fill

    CUCO_CUDA_TRY(cudaMalloc(&slots_, capacity * sizeof(pair_atomic_type)));

    // TODO: (JH) Is this the most efficient way to initialize a vector of
    // atomics?
    // thrust::for_each(
    //    thrust::device, slots_.begin(), slots_.end(),
    //    detail::store_pair<Key, Value>{empty_key_sentinel, empty_value_sentinel});

    auto const block_size = 256;
    auto const grid_size  = (capacity + 4 * block_size - 1) / (4 * block_size);
    detail::initialize<<<grid_size, block_size>>>(
      slots_, empty_key_sentinel, empty_value_sentinel, capacity);
  }

  /**
   * @brief Trivially copyable view of an `insert_only_hash_array` with device
   * accessors and modifiers.
   */
  struct device_view {
    // TODO: (JH): What should the iterator type be? Exposing the
    // atomic seems wrong
    // Only have const_iterator in early version?
    using iterator       = pair_atomic_type*;
    using const_iterator = pair_atomic_type const*;

    /**
     * @brief Construct a new device view object
     *
     * @param slots
     * @param capacity
     * @param empty_key_sentinel
     * @param empty_value_sentinel
     */
    device_view(pair_atomic_type* slots,
                std::size_t capacity,
                Key empty_key_sentinel,
                Value empty_value_sentinel) noexcept
      : slots_{slots},
        capacity_{capacity},
        empty_key_sentinel_{empty_key_sentinel},
        empty_value_sentinel_{empty_value_sentinel}
    {
    }

    /**
     * @brief Insert key/value pair.
     *
     * @tparam Hash
     * @tparam KeyEqual
     * @param insert_pair
     * @param hash
     * @param key_equal
     * @return __device__ insert
     */
    template <typename Hash = MurmurHash3_32<Key>, typename KeyEqual = thrust::equal_to<Key>>
    __device__ thrust::pair<iterator, bool> insert(value_type const& insert_pair,
                                                   KeyEqual key_equal = KeyEqual{},
                                                   Hash hash          = Hash{}) noexcept
    {
      // TODO: What parameter order should key_equal/hash be in?

      iterator current_slot{initial_slot(insert_pair.first, hash)};

      while (true) {
        using cuda::std::memory_order_relaxed;
        auto expected_key   = empty_key_sentinel_;
        auto expected_value = empty_value_sentinel_;

        auto& slot_key   = current_slot->first;
        auto& slot_value = current_slot->second;

        // Update key/value via independent CASes
        bool key_success =
          slot_key.compare_exchange_strong(expected_key, insert_pair.first, memory_order_relaxed);

        bool value_success = slot_value.compare_exchange_strong(
          expected_value, insert_pair.second, memory_order_relaxed);

        // Usually, both will succeed. Otherwise, whoever won the key CAS is
        // guaranteed to eventually update the value
        if (key_success) {
          // If key succeeds and value doesn't, someone else won the value CAS
          // Spin trying to update the value
          while (not value_success) {
            value_success = slot_value.compare_exchange_strong(
              expected_value = empty_value_sentinel_, insert_pair.second, memory_order_relaxed);
          }
          return thrust::make_pair(current_slot, true);
        } else if (value_success) {
          // Key CAS failed, but value succeeded. Restore the value to it's
          // initial value

          // TODO: The pair<atomic<K>, atomic<V>> implementation precludes concurrent insert/find
          // because another thread doing a "find" for the key that's being inserted could observe
          // that the key is updated before the value is updated.
          slot_value.store(empty_value_sentinel_, memory_order_relaxed);
        }

        // expected_key/expected_value contain actual values

        // Key already exists
        if (key_equal(insert_pair.first, expected_key)) {
          return thrust::make_pair(current_slot, false);
        }

        // Slot is occupied by a different key---collision
        // Advance to next slot
        current_slot = next_slot(current_slot);
      }
    }

    /**
     * @brief Find element whose key is equal to `k`.
     *
     * @tparam Hash
     * @tparam KeyEqual
     * @param k
     * @param hash
     * @param key_equal
     * @return __device__ find
     */
    template <typename Hash = MurmurHash3_32<Key>, typename KeyEqual = thrust::equal_to<Key>>
    __device__ iterator find(Key const& k,
                             KeyEqual key_equal = KeyEqual{},
                             Hash hash          = Hash{}) noexcept
    {
      auto current_slot = initial_slot(k, hash);

      while (true) {
        auto const current_key = current_slot->first.load(cuda::std::memory_order_relaxed);
        // Key exists, return iterator to location
        if (key_equal(k, current_key)) { return current_slot; }

        // Key doesn't exist, return end()
        if (key_equal(empty_key_sentinel_, current_key)) { return end(); }

        // TODO: Add check for full hash map?

        // Slot is occupied by a different key---collision
        // Advance to next slot
        current_slot = next_slot(current_slot);
      }
    }

    /**
     * @brief Returns iterator to one past the last element.
     *
     */
    __host__ __device__ const_iterator end() const noexcept { return slots_ + capacity_; }

    /**
     * @brief Returns iterator to one past the last element.
     *
     */
    __host__ __device__ iterator end() noexcept { return slots_ + capacity_; }

    device_view()                   = delete;
    device_view(device_view const&) = default;
    device_view(device_view&&)      = default;
    device_view& operator=(device_view const&) = default;
    device_view& operator=(device_view&&) = default;
    ~device_view()                        = default;

   private:
    pair_atomic_type* __restrict__ const slots_{};  ///< Pointer to slot storage
    std::size_t const capacity_{};                  ///< The number of slots
    Key const empty_key_sentinel_{};                ///< Sentinel key value for empty slots
    Value const empty_value_sentinel_{};            ///< Sentinel mapped value for empty slots

    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * @tparam Hash Unary callable type
     * @param k The key to get the slot for
     * @param hash Hash to use to determine the slot
     * @return Pointer to the initial slot for `k`
     */
    template <typename Hash>
    __device__ iterator initial_slot(Key const& k, Hash hash) const noexcept
    {
      return &slots_[hash(k) % capacity_];
    }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot.
     *
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    __device__ iterator next_slot(iterator s) const noexcept
    {
      // TODO: Since modulus is expensive, I think this should be more
      // efficient than doing (++index % capacity_)
      return (++s < end()) ? s : slots_;
    }
  };  // class device_view

  device_view get_device_view() noexcept
  {
    return device_view{slots_, capacity_, get_empty_key_sentinel(), get_empty_value_sentinel()};
  }

  Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

  Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }

  insert_only_hash_array()                              = default;
  insert_only_hash_array(insert_only_hash_array const&) = delete;
  insert_only_hash_array(insert_only_hash_array&&)      = delete;
  insert_only_hash_array& operator=(insert_only_hash_array const&) = delete;
  insert_only_hash_array& operator=(insert_only_hash_array&&) = delete;

  /**
   * @brief Destroy and free device memory.
   *
   */
  ~insert_only_hash_array() { CUCO_ASSERT_CUDA_SUCCESS(cudaFree(slots_)); }

 private:
  pair_atomic_type* slots_{nullptr};    ///< Pointer to flat slots storage
  std::size_t capacity_{};              ///< Total number of slots
  Key const empty_key_sentinel_{};      ///< Key value that represents an empty slot
  Value const empty_value_sentinel_{};  ///< Initial value of empty slot
};
}  // namespace cuco
