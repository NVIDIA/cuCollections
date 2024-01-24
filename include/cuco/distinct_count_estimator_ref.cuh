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

#include <cuco/detail/hyperloglog/hyperloglog_ref.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cooperative_groups.h>

namespace cuco {
template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash>
class distinct_count_estimator_ref {
  using impl_type = detail::hyperloglog_ref<T, Precision, Scope, Hash>;

 public:
  static constexpr auto thread_scope = impl_type::thread_scope;  ///< CUDA thread scope
  static constexpr auto precision    = impl_type::precision;

  using storage_type = typename impl_type::storage_type;
  template <cuda::thread_scope NewScope>
  using with_scope = distinct_count_estimator_ref<T, Precision, NewScope, Hash>;

  // TODO let storage_type be inferred?
  __host__ __device__ constexpr distinct_count_estimator_ref(
    storage_type& storage,
    cuco::cuda_thread_scope<Scope> scope = {},
    Hash const& hash                     = {}) noexcept;

  template <class CG>
  __device__ void clear(CG const& group) noexcept;

  __device__ void add(T const& item) noexcept;

  template <class CG, cuda::thread_scope OtherScope>
  __device__ void merge(
    CG const& group,
    distinct_count_estimator_ref<T, Precision, OtherScope, Hash> const& other) noexcept;

  [[nodiscard]] __device__ std::size_t estimate(
    cooperative_groups::thread_block const& group) const noexcept;

 private:
  impl_type impl_;
};
}  // namespace cuco

#include <cuco/detail/distinct_count_estimator/distinct_count_estimator_ref.inl>