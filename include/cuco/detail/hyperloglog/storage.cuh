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

#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuco/utility/traits.hpp>

#include <cstddef>
#include <cuda/std/array>

namespace cuco::detail {

template <int32_t Precision>
class hyperloglog_dense_registers {
 public:
  template <class CG>
  __device__ void constexpr clear(CG const& group) noexcept
  {
    for (int i = group.thread_rank(); i < this->registers_.size(); i += group.size()) {
      this->registers_[i] = 0;
    }

    // TODO remove test code
    // int4 constexpr empty{0, 0, 0, 0};
    // auto vec4 = reinterpret_cast<int4*>(this->storage_.data());
    // // #pragma unroll 2
    // for (int i = group.thread_rank(); i < (this->storage_.size() / 4); i += group.size()) {
    //   vec4[i] = empty;
    // }
  }

  __host__ __device__ constexpr int& operator[](std::size_t i) noexcept
  {
    return this->registers_[i];
  }

  __host__ __device__ constexpr int operator[](std::size_t i) const noexcept
  {
    return this->registers_[i];
  }

  __host__ __device__ constexpr std::size_t size() const noexcept
  {
    return this->registers_.size();
  }

  template <cuda::thread_scope Scope>
  __device__ constexpr void update_max(std::size_t i, int value) noexcept
  {
    if constexpr (Scope == cuda::thread_scope_thread) {
      this->registers_[i] = max(this->registers_[i], value);
    } else if constexpr (Scope == cuda::thread_scope_block) {
      atomicMax_block(&(this->registers_[i]), value);
    } else if constexpr (Scope == cuda::thread_scope_device) {
      atomicMax(&(this->registers_[i]), value);
    } else if constexpr (Scope == cuda::thread_scope_system) {
      atomicMax_system(&(this->registers_[i]), value);
    } else {
      static_assert(cuco::dependent_false<decltype(Scope)>, "Unsupported thread scope");
    }
  }

  template <cuda::thread_scope Scope, class CG>
  __device__ void constexpr merge(CG const& group,
                                  hyperloglog_dense_registers const& other) noexcept
  {
    for (int i = group.thread_rank(); i < this->registers_.size(); i += group.size()) {
      this->update_max<Scope>(i, other.registers_[i]);
    }

    // TODO remove test code
    /*
    auto vec4 = reinterpret_cast<int4 const*>(other.storage_.data());
    // #pragma unroll 2
    for (int i = group.thread_rank(); i < (this->storage_.size() / 4); i += group.size()) {
      auto const items = vec4[i];
      if constexpr (Scope == cuda::thread_scope_thread) {
        auto max_vec4  = reinterpret_cast<int4*>(this->storage_.data());
        auto max_items = max_vec4[i];
        max_items.x    = max(max_items.x, items.x);
        max_items.y    = max(max_items.y, items.y);
        max_items.z    = max(max_items.z, items.z);
        max_items.w    = max(max_items.w, items.w);
        max_vec4[i]    = max_items;
      } else if constexpr (Scope == cuda::thread_scope_block) {
        atomicMax_block(this->storage_.data() + (i * 4 + 0), items.x);
        atomicMax_block(this->storage_.data() + (i * 4 + 1), items.y);
        atomicMax_block(this->storage_.data() + (i * 4 + 2), items.z);
        atomicMax_block(this->storage_.data() + (i * 4 + 3), items.w);
      } else if constexpr (Scope == cuda::thread_scope_device) {
        atomicMax(this->storage_.data() + (i * 4 + 0), items.x);
        atomicMax(this->storage_.data() + (i * 4 + 1), items.y);
        atomicMax(this->storage_.data() + (i * 4 + 2), items.z);
        atomicMax(this->storage_.data() + (i * 4 + 3), items.w);
      } else if constexpr (Scope == cuda::thread_scope_system) {
        atomicMax_system(this->storage_.data() + (i * 4 + 0), items.x);
        atomicMax_system(this->storage_.data() + (i * 4 + 1), items.y);
        atomicMax_system(this->storage_.data() + (i * 4 + 2), items.z);
        atomicMax_system(this->storage_.data() + (i * 4 + 3), items.w);
      } else {
        static_assert(cuco::dependent_false<decltype(Scope)>, "Unsupported thread scope");
      }
    }
    */
  }

 private:
  alignas(sizeof(int) * 4) cuda::std::array<int, 1ull << Precision> registers_;
};
}  // namespace cuco::detail
