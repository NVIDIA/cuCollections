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
/**
 * @brief A GPU-accelerated utility for approximating the number of distinct items in a multiset.
 *
 * @note This class implements the HyperLogLog/HyperLogLog++ algorithm:
 * https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf.
 * @note The `Precision` parameter can be used to trade runtime/memory footprint for better
 * accuracy. A higher value corresponds to a more accurate result, however, setting the precision
 * too high will result in deminishing results.
 *
 * @tparam T Type of items to count
 * @tparam Precision Tuning parameter to trade runtime/memory footprint for better accuracy
 * @tparam Scope The scope in which operations will be performed by individual threads
 * @tparam Hash Hash function used to hash items
 */
template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash>
class hyperloglog_ref {
 public:
  using fp_type                      = float;      ///< Floating point type used for reduction
  static constexpr auto thread_scope = Scope;      ///< CUDA thread scope
  static constexpr auto precision    = Precision;  ///< Precision

  using storage_type = hyperloglog_dense_registers<Precision>;  ///< Storage type

  template <cuda::thread_scope NewScope>
  using with_scope = hyperloglog_ref<T, Precision, NewScope, Hash>;  ///< Ref type with different
                                                                     ///< thread scope

  /**
   * @brief Constructs a non-owning `hyperloglog_ref` object.
   *
   * @param storage Reference to storage object of type `storage_type`
   * @param hash The hash function used to hash items
   */
  // Doxygen cannot document unnamed parameter for scope, see
  // https://github.com/doxygen/doxygen/issues/6926
  __host__ __device__ constexpr hyperloglog_ref(storage_type& storage,
                                                cuco::cuda_thread_scope<Scope>,
                                                Hash const& hash) noexcept
    : hash_{hash}, storage_{storage}
  {
  }

  /**
   * @brief Resets the estimator, i.e., clears the current count estimate.
   *
   * @tparam CG CUDA Cooperative Group type
   *
   * @param group CUDA Cooperative group this operation is executed in
   */
  template <class CG>
  __device__ void clear(CG const& group) noexcept
  {
    this->storage_.clear(group);
  }

  /**
   * @brief Adds an item to the estimator.
   *
   * @param item The item to be counted
   */
  __device__ void add(T const& item) noexcept
  {
    // static_assert NumBuckets is not too big
    auto constexpr register_mask = (1 << Precision) - 1;
    auto const h                 = this->hash_(item);
    auto const reg               = h & register_mask;
    auto const zeroes            = cuda::std::countl_zero(h | register_mask) + 1;  // __clz

    this->storage_.update_max<thread_scope>(reg, zeroes);
  }

  /**
   * @brief Merges the result of `other` estimator reference into `*this` estimator reference.
   *
   * @tparam CG CUDA Cooperative Group type
   * @tparam OtherScope Thread scope of `other` estimator
   *
   * @param group CUDA Cooperative group this operation is executed in
   * @param other Other estimator reference to be merged into `*this`
   */
  template <class CG, cuda::thread_scope OtherScope>
  __device__ void merge(CG const& group,
                        hyperloglog_ref<T, Precision, OtherScope, Hash> const& other) noexcept
  {
    this->storage_.merge<thread_scope>(group, other.storage_);
  }

  /**
   * @brief Compute the estimated distinct items count.
   *
   * @param group CUDA thread block group this operation is executed in
   *
   * @return Approximate distinct items count
   */
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
  Hash hash_;  ///< Hash function used to hash items
  // TODO is a reference the right choice here??
  storage_type& storage_;  ///< Reference to storage object

  template <class T_, int32_t Precision_, cuda::thread_scope Scope_, class Hash_>
  friend class hyperloglog_ref;
};
}  // namespace cuco::detail