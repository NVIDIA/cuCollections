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

#pragma once

#include <cuda/std/atomic>

namespace cuco {

// TODO: replace with custom pair type
template <typename K, V>
using pair_type = thrust::pair<K, V>;

// TODO: Allocator
template <typename Key, typename Value, cuda::thread_scope Scope = cuda::thread_scope_device>
class static_map {
  static_assert(std::is_arithmetic<Key>::value, "Unsupported, non-arithmetic key type.");

 public:
  using value_type         = cuco::pair_type<Key, Value>;
  using key_type           = Key;
  using mapped_type        = Value;
  using atomic_key_type    = cuda::atomic<key_type, Scope>;
  using atomic_mapped_type = cuda::atomic<mapped_type, Scope>;
  using pair_atomic_type   = cuco::pair_type<atomic_key_type, atomic_mapped_type>;

  static_map(static_map const&) = delete;
  static_map(static_map&&)      = delete;
  static_map& operator=(static_map const&) = delete;
  static_map& operator=(static_map&&) = delete;

  static_map(std::size_t capacity, Key empty_key_sentinel, Value empty_value_sentinel);

  ~static_map();

  template <typename InputIt, typename Hash, typename KeyEqual>
  void insert(InputIt first, InputIt last, Hash hash, KeyEqual key_equal);

  template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
  void find(
    InputIt first, InputIt last, OutputIt output_begin, Hash hash, KeyEqual key_equal) noexcept;

  template <typename InputIt, typename OutputIt, typename Hash, typename KeyEqual>
  void contains(
    InputIt first, InputIt last, OutputIt output_begin, Hash hash, KeyEqual key_equal) noexcept;

  class device_mutable_view {
   public:
    using iterator       = pair_atomic_type*;
    using const_iterator = pair_atomic_type const*;

    device_mutable_view(pair_atomic_type* slots,
                        std::size_t capacity,
                        Key empty_key_sentinel,
                        Value empty_value_sentinel) noexcept;

    template <typename Hash, typename KeyEqual>
    __device__ cuco::pair<iterator, bool> insert(value_type const& insert_pair,
                                                 KeyEqual key_equal,
                                                 Hash hash) noexcept;

    template <typename CG, typename Hash, typename KeyEqual>
    __device__ cuco::pair<iterator, bool> insert(CG cg,
                                                 value_type const& insert_pair,
                                                 KeyEqual key_equal,
                                                 Hash hash) noexcept;

    std::size_t capacity();

    Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

    Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }

   private:
    pair_atomic_type* __restrict__ slots_{};
    std::size_t const capacity_{};
    Key const empty_key_sentinel_{};
    Value const empty_value_sentinel_{};
  };

  class device_view {
   public:
    using iterator       = pair_atomic_type*;
    using const_iterator = pair_atomic_type const*;

    device_view(pair_atomic_type* slots,
                std::size_t capacity,
                Key empty_key_sentinel,
                Value empty_value_sentinel) noexcept;

    template <typename Hash, typename KeyEqual>
    __device__ iterator find(Key const& k, KeyEqual key_equal, Hash hash) noexcept;

    template <typename CG, typename Hash, typename KeyEqual>
    __device__ iterator find(CG cg, Key const& k, KeyEqual key_equal, Hash hash) noexcept;

    template <typename Hash, typename KeyEqual>
    __device__ bool contains(Key const& k, KeyEqual key_equal, Hash hash) noexcept;

    template <typename CG, typename Hash, typename KeyEqual>
    __device__ bool contains(CG cg, Key const& k, KeyEqual key_equal, Hash hash) noexcept;

    std::size_t capacity();

    Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

    Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }

   private:
    pair_atomic_type* __restrict__ slots_{};
    std::size_t const capacity_{};
    Key const empty_key_sentinel_{};
    Value const empty_value_sentinel_{};
  };

  Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

  Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }

  device_view get_device_view();

  device_mutable_view get_device_mutable_view();

  std::size_t capacity();

 private:
  pair_atomic_type* slots_{nullptr};    ///< Pointer to flat slots storage
  std::size_t capacity_{};              ///< Total number of slots
  Key const empty_key_sentinel_{};      ///< Key value that represents an empty slot
  Value const empty_value_sentinel_{};  ///< Initial value of empty slot
};

}  // namespace cuco