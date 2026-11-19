/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuco/bulk_static_multimap_ref.cuh>
#include <cuco/hash_functions.cuh>

#include <cub/block/block_scan.cuh>
#include <cuda/atomic>
#include <cuda/std/limits>
#include <cooperative_groups.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cuco {
enum class bulk_build_status { success, invalid_size, insufficient_capacity, invalid_storage,
                               invalid_group };

/**
 * @brief Caller-owned storage for one collective build. All regions must be disjoint and aligned.
 * Records and offsets are retained; counters are construction scratch only.
 * Counts/capacities use size_t so narrow Size overflow is rejected before conversion.
 */
template <class Key, class T, class Size = std::size_t>
struct bulk_multimap_storage {
  using value_type = cuco::pair<Key, T>;
  static_assert(std::is_unsigned_v<Size> && (sizeof(Size) == 4 || sizeof(Size) == 8));
  static_assert(std::is_trivially_copyable_v<value_type>);

  value_type* records;
  std::size_t record_capacity;
  Size* offsets;
  std::size_t offset_capacity;
  Size* counters;
  std::size_t counter_capacity;
  std::size_t bucket_count;

  __host__ __device__ bulk_build_status validate(std::size_t count) const noexcept
  {
    constexpr auto limit = cuda::std::numeric_limits<Size>::max();
    constexpr auto bytes_limit = cuda::std::numeric_limits<std::size_t>::max();
    if (!bucket_count || bucket_count >= limit || count > limit ||
        count > bytes_limit / sizeof(value_type) || bucket_count >= bytes_limit / sizeof(Size))
      return bulk_build_status::invalid_size;
    if (count > record_capacity || bucket_count >= offset_capacity ||
        bucket_count > counter_capacity) return bulk_build_status::insufficient_capacity;
    if ((count && !records) || !offsets || !counters || offsets == counters)
      return bulk_build_status::invalid_storage;
    return bulk_build_status::success;
  }

  template <class KeyEqual = cuda::std::equal_to<Key>,
            class Hash = cuco::default_hash_function<Key>, class... Operators>
  __host__ __device__ auto ref(std::size_t count, KeyEqual equal, Hash hash, Operators...) const
  {
    return bulk_static_multimap_ref<Key, T, Size, KeyEqual, Hash, Operators...>{
      records, offsets, static_cast<Size>(count), static_cast<Size>(bucket_count), equal, hash};
  }
};

/**
 * @brief Whole-input construction collectively executed by a one-dimensional CUDA block.
 *
 * Every thread calls build with the same arguments. One block handles each independent table;
 * input and bucket sizes are runtime values and may exceed BlockSize. Input must not overlap
 * output or scratch. A ref is valid only after success; rebuild must exclude existing readers.
 * TempStorage must be private to the block and reusable only after collective completion.
 */
template <class Key, class T, int BlockSize = 128, class Size = std::size_t,
          class Hash = cuco::default_hash_function<Key>>
class bulk_static_multimap_builder {
  static_assert(BlockSize > 0 && BlockSize <= 1024 && BlockSize % 32 == 0);
  using scan_type = cub::BlockScan<Size, BlockSize>;

 public:
  using TempStorage = typename scan_type::TempStorage;

  __device__ explicit bulk_static_multimap_builder(TempStorage& temp) : temp_{temp} {}

  template <class InputIt>
  __device__ bulk_build_status build(cooperative_groups::thread_block group,
                                     InputIt first,
                                     std::size_t count,
                                     bulk_multimap_storage<Key, T, Size> storage,
                                     Hash hash = {}) const
  {
    if (group.size() != BlockSize || blockDim.y != 1 || blockDim.z != 1)
      return bulk_build_status::invalid_group;
    auto const status = storage.validate(count);
    if (status != bulk_build_status::success) return status;
    if constexpr (std::is_pointer_v<InputIt>) {
      if (count && (!first || static_cast<void const*>(first) == storage.records))
        return bulk_build_status::invalid_storage;
    }
    using value_type = cuco::pair<Key, T>;
    auto const lane = group.thread_rank();
    auto const buckets = storage.bucket_count;
    for (std::size_t b = lane; b < buckets; b += BlockSize) storage.counters[b] = 0;
    __syncthreads();
    for (std::size_t i = lane; i < count; i += BlockSize) {
      value_type const pair = first[i];
      auto const b = detail::bulk_bucket(hash, pair.first, static_cast<Size>(buckets));
      cuda::atomic_ref<Size, cuda::thread_scope_block>{storage.counters[b]}.fetch_add(
        1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    Size carry = 0;
    for (std::size_t base = 0; base < buckets; base += BlockSize) {
      auto const b = base + lane;
      Size const value = b < buckets ? storage.counters[b] : 0;
      Size prefix, total;
      scan_type{temp_}.ExclusiveSum(value, prefix, total);
      if (b < buckets) {
        storage.offsets[b] = carry + prefix;
        storage.counters[b] = 0;
      }
      carry += total;
      __syncthreads();
    }
    if (!lane) storage.offsets[buckets] = carry;
    __syncthreads();
    for (std::size_t i = lane; i < count; i += BlockSize) {
      value_type const pair = first[i];
      auto const b = detail::bulk_bucket(hash, pair.first, static_cast<Size>(buckets));
      auto const position = cuda::atomic_ref<Size, cuda::thread_scope_block>{storage.counters[b]}
                              .fetch_add(1, cuda::memory_order_relaxed);
      storage.records[storage.offsets[b] + position] = pair;
    }
    __syncthreads();
    return bulk_build_status::success;
  }

 private:
  TempStorage& temp_;
};
}  // namespace cuco
