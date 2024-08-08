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

#include <cuco/hash_functions.cuh>
#include <cuco/operator.hpp>
#include <cuco/storage.cuh>
#include <cuco/types.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cuda/atomic>
#include <cuda/std/bit>

#include <cstdint>
#include <type_traits>

namespace cuco {

template <class Key, cuda::thread_scope Scope, class Hash, class StorageRef, class... Operators>
class bloom_filter_ref
  : public detail::operator_impl<Operators,
                                 bloom_filter_ref<Key, Scope, Hash, StorageRef, Operators...>>... {
 public:
  using key_type         = Key;  ///< Key Type
  using word_type        = uint32_t;
  using storage_ref_type = StorageRef;                                 ///< Type of storage ref
  using window_type      = typename storage_ref_type::window_type;     ///< Window type
  using extent_type      = typename storage_ref_type::extent_type;     ///< Extent type
  using size_type        = typename storage_ref_type::size_type;       ///< Probing scheme size type
  using iterator         = typename storage_ref_type::iterator;        ///< Slot iterator type
  using const_iterator   = typename storage_ref_type::const_iterator;  ///< Const slot iterator type

  static_assert(std::is_same_v<typename storage_ref_type::value_type, word_type>,
                "StorageRef has invalid value type");

  static constexpr auto window_size =
    storage_ref_type::window_size;             ///< Number of elements handled per window
  static constexpr auto thread_scope = Scope;  ///< CUDA thread scope

  __host__ __device__ explicit constexpr bloom_filter_ref(uint32_t pattern_bits,
                                                          cuda_thread_scope<Scope>,
                                                          Hash hash,
                                                          StorageRef storage_ref) noexcept
    : pattern_bits_{pattern_bits}, hash_{hash}, storage_ref_{storage_ref}
  {
  }

  // we use the MSB bits of the hash value to determine the sub filter for this key
  template <class HashValue>
  __device__ size_type sub_filter_idx(HashValue hash_value) const
  {
    auto constexpr hash_value_bits = sizeof(HashValue) * CHAR_BIT;
    auto const index_bits =
      max(hash_value_bits, cuda::std::bit_width(storage_ref_.num_windows() - 1));
    return (hash_value >> (hash_value_bits - index_bits)) % storage_ref_.num_windows();
  }

  // we use the LSB bits of the hash value to determine the pattern bits for each word
  template <class HashValue>
  __device__ window_type pattern(HashValue hash_value) const
  {
    window_type pattern{};
    auto constexpr word_bits           = sizeof(word_type) * CHAR_BIT;
    auto constexpr bit_index_width     = cuda::std::bit_width(word_bits - 1);
    word_type constexpr bit_index_mask = (word_type{1} << bit_index_width) - 1;

    auto const bits_per_word = pattern_bits_ / window_size;
    auto const remainder     = pattern_bits_ % window_size;

    uint32_t k = 0;
#pragma unroll window_size
    for (int32_t i = 0; i < window_size; ++i) {
      for (int32_t j = 0; j < bits_per_word + (i < remainder ? 1 : 0); ++j) {
        if (k++ >= pattern_bits_) { return pattern; }
        pattern[i] |= word_type{1} << (hash_value & bit_index_mask);
        hash_value >>= bit_index_width;
      }
    }

    // #pragma unroll window_size
    // for (int32_t i = 0; i < window_size; ++i) {
    //   for (int32_t j = 0; j < bits_per_word; ++j) {
    //     pattern[i] |= word_type{1} << (hash_value & bit_index_mask);
    //     hash_value >>= bit_index_width;
    //   }
  }

  return pattern;
}

private : uint32_t pattern_bits_;
Hash hash_;
storage_ref_type storage_ref_;

// Mixins need to be friends with this class in order to access private members
template <class Op, class Ref>
friend class detail::operator_impl;

// Refs with other operator sets need to be friends too
template <class Key_,
          cuda::thread_scope Scope_,
          class Hash_,
          class StorageRef_,
          class... Operators_>
friend class bloom_filter_ref;
};

namespace detail {

template <class Key, cuda::thread_scope Scope, class Hash, class StorageRef, class... Operators>
class operator_impl<op::add_tag, bloom_filter_ref<Key, Scope, Hash, StorageRef, Operators...>> {
  using base_type = bloom_filter_ref<Key, Scope, Hash, StorageRef>;
  using ref_type  = bloom_filter_ref<Key, Scope, Hash, StorageRef, Operators...>;
  using word_type = typename base_type::word_type;

  static constexpr auto window_size = base_type::window_size;

 public:
  template <class ProbeKey>
  __device__ void add(ProbeKey const& key)
  {
    // CRTP: cast `this` to the actual ref type
    auto& ref_ = static_cast<ref_type&>(*this);

    auto const hash_value = ref_.hash_(key);
    auto const idx        = ref_.sub_filter_idx(hash_value);
    auto const pattern    = ref_.pattern(hash_value);

#pragma unroll window_size
    for (int32_t i = 0; i < window_size; ++i) {
      if (pattern[i] != 0) {
        // atomicOr(reinterpret_cast<unsigned int*>(((ref_.storage_ref_.data() + idx)->data() + i)),
        // pattern[i]);
        auto atom_word =
          cuda::atomic_ref<word_type, Scope>{*((ref_.storage_ref_.data() + idx)->data() + i)};
        atom_word.fetch_or(pattern[i], cuda::memory_order_relaxed);
      }
    }
  }
};

template <class Key, cuda::thread_scope Scope, class Hash, class StorageRef, class... Operators>
class operator_impl<op::contains_tag,
                    bloom_filter_ref<Key, Scope, Hash, StorageRef, Operators...>> {
  using base_type = bloom_filter_ref<Key, Scope, Hash, StorageRef>;
  using ref_type  = bloom_filter_ref<Key, Scope, Hash, StorageRef, Operators...>;

  static constexpr auto window_size = base_type::window_size;

 public:
  template <class ProbeKey>
  [[nodiscard]] __device__ bool contains(ProbeKey const& key) const
  {
    // CRTP: cast `this` to the actual ref type
    auto const& ref_ = static_cast<ref_type const&>(*this);

    auto const hash_value = ref_.hash_(key);
    auto const idx        = ref_.sub_filter_idx(hash_value);

    auto const stored_pattern   = ref_.storage_ref_[idx];  // vectorized load
    auto const expected_pattern = ref_.pattern(hash_value);

#pragma unroll window_size
    for (int32_t i = 0; i < window_size; ++i) {
      if ((stored_pattern[i] & expected_pattern[i]) != expected_pattern[i]) { return false; }
    }

    return true;
  }
};

}  // namespace detail

}  // namespace cuco