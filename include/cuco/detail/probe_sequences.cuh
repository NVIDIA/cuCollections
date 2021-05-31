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

#include <cuco/detail/hash_functions.cuh>
#include <cuco/detail/pair.cuh>

namespace cuco {

template <typename Key,
          typename Value,
          uint32_t CGSize          = 8,
          typename Hash1           = cuco::detail::MurmurHash3_32<Key>,
          typename Hash2           = cuco::detail::MurmurHash3_32<Key>,
          cuda::thread_scope Scope = cuda::thread_scope_device>
class DoubleHashing {
 public:
  using value_type         = cuco::pair_type<Key, Value>;
  using key_type           = Key;
  using mapped_type        = Value;
  using atomic_key_type    = cuda::atomic<key_type, Scope>;
  using atomic_mapped_type = cuda::atomic<mapped_type, Scope>;
  using pair_atomic_type   = cuco::pair_type<atomic_key_type, atomic_mapped_type>;
  using iterator           = pair_atomic_type*;
  using const_iterator     = pair_atomic_type const*;

  __host__ __device__ static constexpr uint32_t cg_size() noexcept { return CGSize; }

  __host__ __device__ explicit DoubleHashing(iterator slots, std::size_t capacity)
    : slots_{slots}, capacity_{capacity}
  {
  }

  __host__ __device__ std::size_t get_capacity() const noexcept { return capacity_; }

  __device__ iterator get_slots() noexcept { return slots_; }
  __device__ const_iterator get_slots() const noexcept { return slots_; }

  template <typename CG>
  __device__ iterator initial_slot(CG const& g, Key const k) noexcept
  {
    step_size_ = (hash2_(k + 1) % (capacity_ / (cg_size() * 2) - 1) + 1) * cg_size() * 2;
    std::size_t index =
      hash1_(k) % (capacity_ / (cg_size() * 2)) * cg_size() * 2 + g.thread_rank() * 2;
    return slots_ + index;
  }

  __device__ iterator next_slot(iterator s) noexcept
  {
    std::size_t index = s - slots_;
    return &slots_[(index + step_size_) % capacity_];
  }

 private:
  iterator slots_;
  const std::size_t capacity_;
  std::size_t step_size_;
  Hash1 hash1_{};
  Hash2 hash2_{};
};  // class DoubleHashing

template <typename Key,
          typename Value,
          uint32_t CGSize          = 8,
          typename Hash            = cuco::detail::MurmurHash3_32<Key>,
          cuda::thread_scope Scope = cuda::thread_scope_device>
class LinearProbing {
 public:
  using value_type         = cuco::pair_type<Key, Value>;
  using key_type           = Key;
  using mapped_type        = Value;
  using atomic_key_type    = cuda::atomic<key_type, Scope>;
  using atomic_mapped_type = cuda::atomic<mapped_type, Scope>;
  using pair_atomic_type   = cuco::pair_type<atomic_key_type, atomic_mapped_type>;
  using iterator           = pair_atomic_type*;
  using const_iterator     = pair_atomic_type const*;

  __host__ __device__ static constexpr uint32_t cg_size() noexcept { return CGSize; }

  __host__ __device__ explicit LinearProbing(iterator slots, std::size_t capacity)
    : slots_{slots}, capacity_{capacity}
  {
  }

  __host__ __device__ std::size_t get_capacity() const noexcept { return capacity_; }

  __device__ iterator get_slots() noexcept { return slots_; }
  __device__ const_iterator get_slots() const noexcept { return slots_; }

  template <typename CG>
  __device__ iterator initial_slot(CG const& g, Key const k) noexcept
  {
    return &slots_[(hash_(k) + g.thread_rank() * 2) % capacity_];
  }

  __device__ iterator next_slot(iterator s) noexcept
  {
    std::size_t index = s - slots_;
    return &slots_[(index + cg_size() * 2) % capacity_];
  }

 private:
  iterator slots_;
  const std::size_t capacity_;
  Hash hash_{};
};  // class LinearProbing

}  // namespace cuco
