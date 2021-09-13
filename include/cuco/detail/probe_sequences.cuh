/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
namespace detail {

template <typename Key,
          typename Value,
          uint32_t CGSize          = 8,
          cuda::thread_scope Scope = cuda::thread_scope_device>
class probe_sequence_base {
 protected:
  using value_type         = cuco::pair_type<Key, Value>;
  using key_type           = Key;
  using mapped_type        = Value;
  using atomic_key_type    = cuda::atomic<key_type, Scope>;
  using atomic_mapped_type = cuda::atomic<mapped_type, Scope>;
  using pair_atomic_type   = cuco::pair_type<atomic_key_type, atomic_mapped_type>;
  using iterator           = pair_atomic_type*;
  using const_iterator     = pair_atomic_type const*;

 public:
  __host__ __device__ static constexpr uint32_t cg_size() noexcept { return CGSize; }

  __host__ __device__ static constexpr uint32_t vector_width() noexcept { return 2u; }

  __host__ __device__ static constexpr bool uses_vector_load() noexcept
  {
    return cuco::detail::is_packable<value_type>();
  }

  __host__ __device__ explicit probe_sequence_base(iterator slots, std::size_t capacity)
    : slots_{slots}, capacity_{capacity}
  {
  }

  __host__ __device__ std::size_t get_capacity() const noexcept { return capacity_; }

  __device__ iterator get_slots() noexcept { return slots_; }
  __device__ const_iterator get_slots() const noexcept { return slots_; }

 protected:
  iterator slots_;
  const std::size_t capacity_;
};  // class probe_sequence_base

template <typename Key,
          typename Value,
          uint32_t CGSize          = 8,
          typename Hash1           = cuco::detail::MurmurHash3_32<Key>,
          typename Hash2           = cuco::detail::MurmurHash3_32<Key>,
          cuda::thread_scope Scope = cuda::thread_scope_device>
class double_hashing : public probe_sequence_base<Key, Value, CGSize, Scope> {
 public:
  using iterator = typename probe_sequence_base<Key, Value, CGSize, Scope>::iterator;
  using probe_sequence_base<Key, Value, CGSize, Scope>::capacity_;
  using probe_sequence_base<Key, Value, CGSize, Scope>::slots_;
  using probe_sequence_base<Key, Value, CGSize, Scope>::cg_size;
  using probe_sequence_base<Key, Value, CGSize, Scope>::vector_width;
  using probe_sequence_base<Key, Value, CGSize, Scope>::uses_vector_load;

  __host__ __device__ explicit double_hashing(iterator slots, std::size_t capacity) noexcept
    : probe_sequence_base<Key, Value, CGSize, Scope>{slots, capacity}
  {
  }

  template <typename CG>
  __device__ iterator initial_slot(CG const& g, Key const k) noexcept
  {
    std::size_t index;
    auto const hash_value = hash1_(k);
    if constexpr (uses_vector_load()) {
      step_size_ = (hash2_(k + 1) % (capacity_ / (cg_size() * vector_width()) - 1) + 1) *
                   cg_size() * vector_width();
      index = hash_value % (capacity_ / (cg_size() * vector_width())) * cg_size() * vector_width() +
              g.thread_rank() * vector_width();
    } else {
      step_size_ = (hash2_(k + 1) % (capacity_ / cg_size() - 1) + 1) * cg_size();
      index      = (hash_value + g.thread_rank()) % capacity_;
    }
    return slots_ + index;
  }

  __device__ iterator next_slot(iterator s) noexcept
  {
    std::size_t index = s - slots_;
    return &slots_[(index + step_size_) % capacity_];
  }

 private:
  std::size_t step_size_;
  Hash1 hash1_{};
  Hash2 hash2_{};
};  // class double_hashing

template <typename Key,
          typename Value,
          uint32_t CGSize          = 8,
          typename Hash            = cuco::detail::MurmurHash3_32<Key>,
          cuda::thread_scope Scope = cuda::thread_scope_device>
class linear_probing : public probe_sequence_base<Key, Value, CGSize, Scope> {
 public:
  using iterator = typename probe_sequence_base<Key, Value, CGSize, Scope>::iterator;
  using probe_sequence_base<Key, Value, CGSize, Scope>::capacity_;
  using probe_sequence_base<Key, Value, CGSize, Scope>::slots_;
  using probe_sequence_base<Key, Value, CGSize, Scope>::cg_size;
  using probe_sequence_base<Key, Value, CGSize, Scope>::vector_width;
  using probe_sequence_base<Key, Value, CGSize, Scope>::uses_vector_load;

  __host__ __device__ explicit linear_probing(iterator slots, std::size_t capacity)
    : probe_sequence_base<Key, Value, CGSize, Scope>{slots, capacity}
  {
  }

  template <typename CG>
  __device__ iterator initial_slot(CG const& g, Key const k) noexcept
  {
    auto hash_value = hash_(k);
    hash_value      = hash_value % 2 ? hash_value + 1 : hash_value;

    std::size_t offset;
    if constexpr (uses_vector_load()) {
      offset = g.thread_rank() * vector_width();
    } else {
      offset = g.thread_rank();
    }
    return &slots_[(hash_value + offset) % capacity_];
  }

  __device__ iterator next_slot(iterator s) noexcept
  {
    std::size_t index = s - slots_;
    std::size_t offset;
    if constexpr (uses_vector_load()) {
      offset = cg_size() * vector_width();
    } else {
      offset = cg_size();
    }
    return &slots_[(index + offset) % capacity_];
  }

 private:
  Hash hash_{};
};  // class linear_probing

}  // namespace detail
}  // namespace cuco
