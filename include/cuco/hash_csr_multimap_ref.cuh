/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuco/operator.hpp>
#include <cuco/pair.cuh>

#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/utility>

#include <type_traits>

namespace cuco {
namespace detail {
/// A representative input index and the high bits of its key hash share one 32-bit slot.
using hash_csr_slot_type = cuda::std::uint32_t;

/// Marks an empty slot and an absent group. No input index can produce it.
inline constexpr hash_csr_slot_type hash_csr_npos =
  cuda::std::numeric_limits<hash_csr_slot_type>::max();

template <class Hash, class Key>
__host__ __device__ cuda::std::uint32_t hash_csr_hash(Hash const& hash, Key const& key)
{
  auto const value = hash(key);
  using hash_type  = decltype(value);
  static_assert(std::is_integral_v<hash_type> && !std::is_same_v<hash_type, bool>);
  if constexpr (sizeof(hash_type) > sizeof(cuda::std::uint32_t)) {
    return static_cast<cuda::std::uint32_t>(value ^ (value >> 32));
  } else {
    return static_cast<cuda::std::uint32_t>(value);
  }
}

/**
 * @brief Open-addressed table with one slot per distinct key.
 *
 * The low bits of a slot hold the input index that first claimed the key (its representative) and
 * the remaining high bits hold a hash fingerprint. A fingerprint match is always followed by a
 * key comparison, so a narrow fingerprint cannot produce a false match.
 */
struct hash_csr_table_ref {
  hash_csr_slot_type* slots;
  cuda::std::uint32_t capacity;
  cuda::std::uint32_t index_mask;

  __device__ cuda::std::uint32_t initial_slot(cuda::std::uint32_t hash) const noexcept
  {
    // Reverse the bits so that slot selection uses the low hash bits, which the fingerprint
    // does not cover.
    return cuda::mul_hi(cuda::std::bit_reverse(hash), capacity);
  }

  __device__ cuda::std::uint32_t next_slot(cuda::std::uint32_t slot) const noexcept
  {
    return ++slot == capacity ? 0 : slot;
  }

  __device__ bool same_fingerprint(cuda::std::uint32_t hash, hash_csr_slot_type slot) const noexcept
  {
    return ((hash ^ slot) & ~index_mask) == 0;
  }

  /**
   * @brief Claims a slot for `index`, or returns the representative of an equal key.
   *
   * `equal(other)` compares the key of `index` with the key of input index `other`.
   */
  template <class Equal>
  __device__ cuda::std::uint32_t insert(cuda::std::uint32_t hash,
                                        cuda::std::uint32_t index,
                                        Equal equal) const noexcept
  {
    auto const desired = (hash & ~index_mask) | index;
    auto slot          = initial_slot(hash);
    for (cuda::std::uint32_t step = 0; step < capacity; ++step) {
      auto expected = hash_csr_npos;
      if (cuda::atomic_ref<hash_csr_slot_type, cuda::thread_scope_device>{slots[slot]}
            .compare_exchange_strong(expected, desired, cuda::memory_order_relaxed)) {
        return index;
      }
      if (same_fingerprint(hash, expected) && equal(expected & index_mask)) {
        return expected & index_mask;
      }
      slot = next_slot(slot);
    }
    return hash_csr_npos;
  }
};

template <class Operator, class Ref>
struct hash_csr_multimap_operator;

template <class Ref>
struct hash_csr_multimap_operator<cuco::find_tag, Ref> {
  template <class ProbeKey>
  __device__ auto find(ProbeKey const& key) const noexcept
  {
    auto const& ref  = *static_cast<Ref const*>(this);
    auto const range = ref.equal_range(key);
    return range.first == range.second ? ref.end() : ref.records_ + range.first;
  }
};

template <class Ref>
struct hash_csr_multimap_operator<cuco::contains_tag, Ref> {
  template <class ProbeKey>
  __device__ bool contains(ProbeKey const& key) const noexcept
  {
    auto const range = static_cast<Ref const*>(this)->equal_range(key);
    return range.first != range.second;
  }
};

template <class Ref>
struct hash_csr_multimap_operator<cuco::count_tag, Ref> {
  template <class ProbeKey>
  __device__ auto count(ProbeKey const& key) const noexcept
  {
    auto const range = static_cast<Ref const*>(this)->equal_range(key);
    return range.second - range.first;
  }
};

template <class Ref>
struct hash_csr_multimap_operator<cuco::for_each_tag, Ref> {
  template <class ProbeKey, class Callback>
  __device__ void for_each(ProbeKey const& key, Callback callback) const
  {
    auto const& ref  = *static_cast<Ref const*>(this);
    auto const range = ref.equal_range(key);
    for (auto i = range.first; i < range.second; ++i) {
      callback(ref.records_ + i);
    }
  }
};
}  // namespace detail

/**
 * @brief Non-owning read-only reference to a `hash_csr_multimap`.
 *
 * Construction must complete before queries begin. Rebuilding invalidates all old refs.
 * Each operator runs on one thread per query. Callbacks receive const entry iterators.
 * Equal keys must hash identically; comparison is `pred(probe_key, stored_key)`.
 */
template <class Key, class T, class KeyEqual, class Hash, class... Operators>
class hash_csr_multimap_ref : public detail::hash_csr_multimap_operator<
                                Operators,
                                hash_csr_multimap_ref<Key, T, KeyEqual, Hash, Operators...>>... {
  template <class, class>
  friend struct detail::hash_csr_multimap_operator;
  template <class, class, class, class, class>
  friend class hash_csr_multimap;

 public:
  using key_type    = Key;
  using mapped_type = T;
  using value_type  = cuco::pair<Key, T>;
  using size_type   = cuda::std::uint32_t;
  using iterator    = value_type const*;

  __host__ __device__ hash_csr_multimap_ref(detail::hash_csr_table_ref table,
                                            size_type const* offsets,
                                            value_type const* records,
                                            size_type size,
                                            KeyEqual equal,
                                            Hash hash)
    : table_{table}, offsets_{offsets}, records_{records}, size_{size}, equal_{equal}, hash_{hash}
  {
  }

  __host__ __device__ iterator end() const noexcept { return records_ + size_; }
  __host__ __device__ size_type size() const noexcept { return size_; }

  template <class... NewOperators>
  __host__ __device__ auto rebind_operators(NewOperators...) const noexcept
  {
    return hash_csr_multimap_ref<Key, T, KeyEqual, Hash, NewOperators...>{
      table_, offsets_, records_, size_, equal_, hash_};
  }

 private:
  /// Record positions [first, second) of every occurrence of `key`; empty when it is absent.
  template <class ProbeKey>
  __device__ cuda::std::pair<size_type, size_type> equal_range(ProbeKey const& key) const noexcept
  {
    if (!size_) { return {0, 0}; }
    auto const hash = detail::hash_csr_hash(hash_, key);
    auto slot       = table_.initial_slot(hash);
    for (size_type step = 0; step < table_.capacity; ++step) {
      auto const current = table_.slots[slot];
      if (current == detail::hash_csr_npos) { break; }
      if (table_.same_fingerprint(hash, current)) {
        auto const group = current & table_.index_mask;
        auto const begin = offsets_[group];
        if (equal_(key, records_[begin].first)) { return {begin, offsets_[group + 1]}; }
      }
      slot = table_.next_slot(slot);
    }
    return {0, 0};
  }

  detail::hash_csr_table_ref table_;
  size_type const* offsets_;
  value_type const* records_;
  size_type size_;
  KeyEqual equal_;
  Hash hash_;
};
}  // namespace cuco
