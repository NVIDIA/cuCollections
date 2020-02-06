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

#include <cu_collections/hash_functions.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <simt/atomic>

namespace detail {

// Function object is required because __device__ lambdas aren't allowed in
// ctors
template <typename Key, typename Value>
struct store_pair {
  store_pair(Key k, Value v) : k_{k}, v_{v} {}

  template <simt::thread_scope Scope>
  void operator()(simt::atomic<thrust::pair<Key, Value>, Scope>& p) {
    // TODO: (JH) Use placement new instead?
    p.store(thrust::make_pair(k_, v_), simt::memory_order_relaxed);
  }

 private:
  Key k_;
  Value v_;
};
}  // namespace detail

template <typename Key, typename Value,
          simt::thread_scope Scope = simt::thread_scope_system>
class insert_only_hash_array {
  // TODO: (JH) What static_assert(s) should we have on Key/Value?
  // is_literal_type? (deprecated in C++17, removed c++20)
  // std::has_unique_object_representations? (C++17)
  // probably is_arithmetic
  // document padding bits sharp edges

 public:
  using value_type = thrust::pair<Key, Value>;
  using pair_type = thrust::pair<const Key, Value>;
  using atomic_pair_type = simt::atomic<pair_type, Scope>;

  explicit insert_only_hash_array(std::size_t capacity, Key empty_key_sentinel,
                                  Value empty_value_sentinel)
      : slots_(capacity),
        empty_key_sentinel_{empty_key_sentinel},
        empty_value_sentinel_{empty_value_sentinel_} {
    // vector_base? uninitialized fill

    // TODO: (JH) Is this the most efficient way to initialize a vector of
    // atomics?
    thrust::for_each(slots_.begin(), slots_.end(),
                     detail::store_pair<Key, Value>{empty_key_sentinel,
                                                    empty_value_sentinel});
  }

  /**
   * @brief Trivially copyable view of an `insert_only_hash_array` with device
   * accessors and modifiers.
   */
  struct device_view {
    // TODO: (JH): What should the iterator type be? Exposing the
    // atomic seems wrong
    // Only have const_iterator in early version?
    using iterator = atomic_pair_type*;
    using const_iterator = atomic_pair_type const*;

    /**
     * @brief Construct a new device view object
     *
     * @param slots
     * @param capacity
     * @param empty_key_sentinel
     * @param empty_value_sentinel
     */
    device_view(atomic_pair_type* slots, std::size_t capacity,
                Key empty_key_sentinel, Value empty_value_sentinel) noexcept
        : slots_{slots}, capacity_{capacity} {}

    /**
     * @brief Insert key/value pair.
     *
     * @tparam MurmurHash3_32<Key>
     * @tparam thrust::equal_to<Key>
     * @param insert_pair
     * @param hash
     * @param key_equal
     * @return __device__ insert
     */
    template <typename Hash = MurmurHash3_32<Key>,
              typename KeyEqual = thrust::equal_to<Key>>
    __device__ thrust::pair<iterator, bool> insert(
        pair_type const& insert_pair, Hash hash = Hash{},
        KeyEqual key_equal = KeyEqual{}) noexcept {
      // TODO: What order should key_equal/hash be in?

      auto const key_hash{hash(insert_pair.first)};
      auto const index{key_hash % capacity_};

      // TODO: What kind of atomic exchange to use?
      // MD: use strong w/ acq_rel
    }

    /**
     * @brief Find element whose key is equal to `k`.
     *
     * @tparam MurmurHash3_32<Key>
     * @tparam thrust::equal_to<Key>
     * @param k
     * @param hash
     * @param key_equal
     * @return __device__ find
     */
    template <typename Hash = MurmurHash3_32<Key>,
              typename KeyEqual = thrust::equal_to<Key>>
    __device__ const_iterator find(Key const& k, Hash hash,
                                   KeyEqual key_equal) const noexcept {}

   private:
    atomic_pair_type* const slots_;
    std::size_t const capacity_;
    Key const empty_key_sentinel_;
    Value const empty_value_sentinel_;
  };

  device_view get_device_view() noexcept {
    return device_view{slots_.data().get(), slots_.size(),
                       get_empty_key_sentinel(), get_empty_value_sentinel()};
  }

  Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

  Value get_empty_value_sentinel() const noexcept {
    return empty_value_sentinel_;
  }

  insert_only_hash_array() = default;
  insert_only_hash_array(insert_only_hash_array const&) = default;
  insert_only_hash_array(insert_only_hash_array&&) = default;
  insert_only_hash_array& operator=(insert_only_hash_array const&) = default;
  insert_only_hash_array& operator=(insert_only_hash_array&&) = default;
  ~insert_only_hash_array() = default;

 private:
  thrust::device_vector<atomic_pair_type> slots_{};
  Key const empty_key_sentinel_{};
  Value const empty_value_sentinel_{};
};