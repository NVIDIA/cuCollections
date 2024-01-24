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

#include <cuco/detail/hyperloglog/finalizer.cuh>
#include <cuco/detail/hyperloglog/storage.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuco/utility/traits.hpp>

#include <cstddef>
#include <cuda/std/bit>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cuco::detail {
template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash>
class hyperloglog_ref {
 public:
  using fp_type                      = float;
  static constexpr auto thread_scope = Scope;  ///< CUDA thread scope
  static constexpr auto precision    = Precision;

  using storage_type = hyperloglog_dense_registers<Precision>;
  template <cuda::thread_scope NewScope>
  using with_scope = hyperloglog_ref<T, Precision, NewScope, Hash>;

  __host__ __device__ constexpr hyperloglog_ref(storage_type& storage,
                                                cuco::cuda_thread_scope<Scope> = {},
                                                Hash const& hash               = {}) noexcept
    : hash_{hash}, storage_{storage}
  {
  }

  template <class CG>
  __device__ void clear(CG const& group) noexcept
  {
    this->storage_.clear(group);
  }

  __device__ void add(T const& item) noexcept
  {
    // static_assert NumBuckets is not too big
    auto constexpr register_mask = (1 << Precision) - 1;
    auto const h                 = this->hash_(item);
    auto const reg               = h & register_mask;
    auto const zeroes            = cuda::std::countl_zero(h | register_mask) + 1;  // __clz

    this->storage_.update_max<thread_scope>(reg, zeroes);
  }

  template <class CG, cuda::thread_scope OtherScope>
  __device__ void merge(CG const& group,
                        hyperloglog_ref<T, Precision, OtherScope, Hash> const& other) noexcept
  {
    this->storage_.merge<thread_scope>(group, other.storage_);
  }

  [[nodiscard]] __device__ std::size_t estimate(
    cooperative_groups::thread_block const& group) const noexcept
  {
    __shared__ cuda::atomic<fp_type, cuda::thread_scope_block> block_sum;
    __shared__ cuda::atomic<int, cuda::thread_scope_block> block_zeroes;
    __shared__ std::size_t estimate;

    if (group.thread_rank() == 0) {
      new (&block_sum) decltype(block_sum){0};
      new (&block_zeroes) decltype(block_zeroes){0};
    }
    group.sync();

    fp_type thread_sum = 0;
    int thread_zeroes  = 0;
    for (int i = group.thread_rank(); i < this->storage_.size(); i += group.size()) {
      auto const reg = this->storage_[i];
      thread_sum += fp_type{1} / static_cast<fp_type>(1 << reg);
      thread_zeroes += reg == 0;
    }

    // warp reduce Z and V
    auto const warp = cooperative_groups::tiled_partition<32>(group);
    cooperative_groups::reduce_update_async(
      warp, block_sum, thread_sum, cooperative_groups::plus<fp_type>());
    cooperative_groups::reduce_update_async(
      warp, block_zeroes, thread_zeroes, cooperative_groups::plus<int>());
    group.sync();

    if (group.thread_rank() == 0) {
      auto const z = block_sum.load(cuda::std::memory_order_relaxed);
      auto const v = block_zeroes.load(cuda::std::memory_order_relaxed);
      estimate     = cuco::hyperloglog_ns::detail::finalizer<Precision>::finalize(z, v);
    }
    group.sync();

    return estimate;
  }

 private:
  Hash hash_;
  storage_type& storage_;  // TODO is a reference the right choice here??

  template <class T_, int32_t Precision_, cuda::thread_scope Scope_, class Hash_>
  friend class hyperloglog_ref;
};
}  // namespace cuco::detail