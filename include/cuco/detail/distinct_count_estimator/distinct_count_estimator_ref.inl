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

namespace cuco {

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash>
__host__ __device__ constexpr distinct_count_estimator_ref<T, Precision, Scope, Hash>::
  distinct_count_estimator_ref(storage_type& storage, Hash const& hash) noexcept
  : impl_{storage, hash}
{
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash>
template <class CG>
__device__ void distinct_count_estimator_ref<T, Precision, Scope, Hash>::clear(
  CG const& group) noexcept
{
  this->impl_.clear(group);
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash>
__device__ void distinct_count_estimator_ref<T, Precision, Scope, Hash>::add(T const& item) noexcept
{
  this->impl_.add(item);
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash>
template <class CG, cuda::thread_scope OtherScope>
__device__ void distinct_count_estimator_ref<T, Precision, Scope, Hash>::merge(
  CG const& group,
  distinct_count_estimator_ref<T, Precision, OtherScope, Hash> const& other) noexcept
{
  this->impl_.merge(group, other);
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash>
__device__ std::size_t distinct_count_estimator_ref<T, Precision, Scope, Hash>::estimate(
  cooperative_groups::thread_block const& group) const noexcept
{
  this->impl_.estimate(group);
}
}  // namespace cuco