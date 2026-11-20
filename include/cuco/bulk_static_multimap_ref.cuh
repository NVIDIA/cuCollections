/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuco/operator.hpp>
#include <cuco/pair.cuh>

#include <cuda/std/functional>

#include <cooperative_groups/reduce.h>

#include <cstddef>
#include <type_traits>

namespace cuco {
namespace detail {
template <class Operator, class Ref>
struct bulk_multimap_operator;

template <class Ref>
struct bulk_multimap_operator<cuco::find_tag, Ref> {
  template <unsigned Threads, class Parent, class ProbeKey>
  __device__ auto find(cooperative_groups::thread_block_tile<Threads, Parent> group,
                       ProbeKey const& key) const noexcept
  {
    static_assert(Threads <= 32, "Query groups must fit one warp");
    return static_cast<Ref const*>(this)->find_group_impl(group, key);
  }

  template <class ProbeKey>
  __device__ auto find(ProbeKey const& key) const noexcept
  {
    return static_cast<Ref const*>(this)->find_impl(key);
  }
};

template <class Ref>
struct bulk_multimap_operator<cuco::contains_tag, Ref> {
  template <unsigned Threads, class Parent, class ProbeKey>
  __device__ auto contains(cooperative_groups::thread_block_tile<Threads, Parent> group,
                           ProbeKey const& key) const noexcept
  {
    static_assert(Threads <= 32, "Query groups must fit one warp");
    auto const& ref = *static_cast<Ref const*>(this);
    return ref.find_group_impl(group, key) != ref.end();
  }

  template <class ProbeKey>
  __device__ bool contains(ProbeKey const& key) const noexcept
  {
    auto const& ref = *static_cast<Ref const*>(this);
    return ref.find_impl(key) != ref.end();
  }
};

template <class Ref>
struct bulk_multimap_operator<cuco::count_tag, Ref> {
  template <unsigned Threads, class Parent, class ProbeKey>
  __device__ auto count(cooperative_groups::thread_block_tile<Threads, Parent> group,
                        ProbeKey const& key) const noexcept
  {
    static_assert(Threads <= 32, "Query groups must fit one warp");
    return static_cast<Ref const*>(this)->count_group_impl(group, key);
  }

  template <class ProbeKey>
  __device__ auto count(ProbeKey const& key) const noexcept
  {
    return static_cast<Ref const*>(this)->count_impl(key);
  }
};

template <class Ref>
struct bulk_multimap_operator<cuco::for_each_tag, Ref> {
  template <unsigned Threads, class Parent, class ProbeKey, class Callback, class SyncOp>
  __device__ void for_each(cooperative_groups::thread_block_tile<Threads, Parent> group,
                           ProbeKey const& key,
                           Callback callback,
                           SyncOp sync_op) const
  {
    static_assert(Threads <= 32, "Query groups must fit one warp");
    static_cast<Ref const*>(this)->for_each_group_sync_impl(group, key, callback, sync_op);
  }

  template <unsigned Threads, class Parent, class ProbeKey, class Callback>
  __device__ void for_each(cooperative_groups::thread_block_tile<Threads, Parent> group,
                           ProbeKey const& key,
                           Callback callback) const
  {
    static_assert(Threads <= 32, "Query groups must fit one warp");
    static_cast<Ref const*>(this)->for_each_group_impl(group, key, callback);
  }

  template <class ProbeKey, class Callback>
  __device__ void for_each(ProbeKey const& key, Callback callback) const
  {
    static_cast<Ref const*>(this)->for_each_impl(key, callback);
  }
};

template <class Hash, class Key, class Size>
__host__ __device__ Size bulk_bucket(Hash const& hash, Key const& key, Size buckets)
{
  using hash_type = decltype(hash(key));
  static_assert(std::is_integral_v<hash_type> && !std::is_same_v<hash_type, bool>);
  using bits_type = std::make_unsigned_t<hash_type>;
  using wide_type = std::common_type_t<bits_type, Size>;
  return static_cast<Size>(wide_type(static_cast<bits_type>(hash(key))) % buckets);
}
}  // namespace detail

/**
 * @brief Non-owning read-only reference to a whole-input multimap.
 *
 * Construction must complete before queries begin. Rebuilding invalidates all old refs.
 * Scalar operators use one thread per query. Group find/contains return a uniform result;
 * group count returns per-lane matches (reduce across the group for a total). Group for_each
 * distributes matches across lanes; callbacks must not synchronize the group. Callbacks
 * receive const entry iterators.
 * Equal keys must hash identically; comparison is pred(probe_key, stored_key).
 * Only find, contains, count and for_each operator tags are supported.
 */
template <class Key, class T, class Size, class KeyEqual, class Hash, class... Operators>
class bulk_static_multimap_ref
  : public detail::bulk_multimap_operator<
      Operators,
      bulk_static_multimap_ref<Key, T, Size, KeyEqual, Hash, Operators...>>... {
  template <class, class>
  friend struct detail::bulk_multimap_operator;

 public:
  using key_type    = Key;
  using mapped_type = T;
  using value_type  = cuco::pair<Key, T>;
  using size_type   = Size;
  using iterator    = value_type const*;

  __host__ __device__ bulk_static_multimap_ref(value_type const* records,
                                               Size const* offsets,
                                               Size size,
                                               Size buckets,
                                               KeyEqual equal,
                                               Hash hash)
    : records_{records},
      offsets_{offsets},
      size_{size},
      buckets_{buckets},
      equal_{equal},
      hash_{hash}
  {
  }

  __host__ __device__ iterator end() const noexcept { return size_ ? records_ + size_ : records_; }
  __host__ __device__ Size size() const noexcept { return size_; }
  __host__ __device__ Size bucket_count() const noexcept { return buckets_; }

  template <class... NewOperators>
  __host__ __device__ auto rebind_operators(NewOperators...) const noexcept
  {
    return bulk_static_multimap_ref<Key, T, Size, KeyEqual, Hash, NewOperators...>{
      records_, offsets_, size_, buckets_, equal_, hash_};
  }

 private:
  template <class Group, class ProbeKey>
  __device__ iterator find_group_impl(Group group, ProbeKey const& key) const noexcept
  {
    Size candidate = size_;
    if (size_) {
      auto const b     = detail::bulk_bucket(hash_, key, buckets_);
      auto const end   = offsets_[b + 1];
      auto const begin = offsets_[b];
      Size i           = group.thread_rank() < end - begin ? begin + group.thread_rank() : end;
      while (i < end) {
        if (equal_(key, records_[i].first)) {
          candidate = i;
          break;
        }
        if (end - i <= group.size()) break;
        i += group.size();
      }
    }
    candidate = cooperative_groups::reduce(group, candidate, cooperative_groups::less<Size>{});
    return candidate == size_ ? end() : records_ + candidate;
  }

  template <class Group, class ProbeKey>
  __device__ Size count_group_impl(Group group, ProbeKey const& key) const noexcept
  {
    Size matches = 0;
    if (!size_) return matches;
    auto const b     = detail::bulk_bucket(hash_, key, buckets_);
    auto const end   = offsets_[b + 1];
    auto const begin = offsets_[b];
    Size i           = group.thread_rank() < end - begin ? begin + group.thread_rank() : end;
    while (i < end) {
      matches += equal_(key, records_[i].first);
      if (end - i <= group.size()) break;
      i += group.size();
    }
    return matches;
  }

  template <class Group, class ProbeKey, class Callback, class SyncOp>
  __device__ void for_each_group_sync_impl(Group group,
                                           ProbeKey const& key,
                                           Callback callback,
                                           SyncOp sync_op) const
  {
    if (!size_) return;
    auto const b   = detail::bulk_bucket(hash_, key, buckets_);
    auto const end = offsets_[b + 1];
    Size base      = offsets_[b];
    while (base < end) {
      if (group.thread_rank() < end - base) {
        auto const i = base + group.thread_rank();
        if (equal_(key, records_[i].first)) callback(records_ + i);
      }
      sync_op(group);
      if (end - base <= group.size()) break;
      base += group.size();
    }
  }

  template <class Group, class ProbeKey, class Callback>
  __device__ void for_each_group_impl(Group group, ProbeKey const& key, Callback callback) const
  {
    if (!size_) return;
    auto const b     = detail::bulk_bucket(hash_, key, buckets_);
    auto const end   = offsets_[b + 1];
    auto const begin = offsets_[b];
    Size i           = group.thread_rank() < end - begin ? begin + group.thread_rank() : end;
    while (i < end) {
      if (equal_(key, records_[i].first)) callback(records_ + i);
      if (end - i <= group.size()) break;
      i += group.size();
    }
  }

  template <class ProbeKey>
  __device__ iterator find_impl(ProbeKey const& key) const noexcept
  {
    if (!size_) return end();
    auto const b = detail::bulk_bucket(hash_, key, buckets_);
    for (auto i = offsets_[b]; i < offsets_[b + 1]; ++i) {
      if (equal_(key, records_[i].first)) return records_ + i;
    }
    return end();
  }

  template <class ProbeKey>
  __device__ Size count_impl(ProbeKey const& key) const noexcept
  {
    if (!size_) return 0;
    auto const b = detail::bulk_bucket(hash_, key, buckets_);
    Size matches = 0;
    for (auto i = offsets_[b]; i < offsets_[b + 1]; ++i) {
      matches += equal_(key, records_[i].first);
    }
    return matches;
  }

  template <class ProbeKey, class Callback>
  __device__ void for_each_impl(ProbeKey const& key, Callback callback) const
  {
    if (!size_) return;
    auto const b = detail::bulk_bucket(hash_, key, buckets_);
    for (auto i = offsets_[b]; i < offsets_[b + 1]; ++i) {
      if (equal_(key, records_[i].first)) callback(records_ + i);
    }
  }

  value_type const* records_;
  Size const* offsets_;
  Size size_;
  Size buckets_;
  KeyEqual equal_;
  Hash hash_;
};
}  // namespace cuco
