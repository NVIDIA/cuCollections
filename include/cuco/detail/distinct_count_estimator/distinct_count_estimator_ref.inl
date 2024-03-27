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

template <class T, cuda::thread_scope Scope, class Hash>
__host__
  __device__ constexpr distinct_count_estimator_ref<T, Scope, Hash>::distinct_count_estimator_ref(
    cuda::std::span<std::byte> sketch_span, Hash const& hash)
  : impl_{sketch_span, hash}
{
}

template <class T, cuda::thread_scope Scope, class Hash>
template <class CG>
__device__ constexpr void distinct_count_estimator_ref<T, Scope, Hash>::clear(
  CG const& group) noexcept
{
  this->impl_.clear(group);
}

template <class T, cuda::thread_scope Scope, class Hash>
__host__ constexpr void distinct_count_estimator_ref<T, Scope, Hash>::clear_async(
  cuco::cuda_stream_ref stream) noexcept
{
  this->impl_.clear_async(stream);
}

template <class T, cuda::thread_scope Scope, class Hash>
__host__ constexpr void distinct_count_estimator_ref<T, Scope, Hash>::clear(
  cuco::cuda_stream_ref stream)
{
  this->impl_.clear(stream);
}

template <class T, cuda::thread_scope Scope, class Hash>
__device__ constexpr void distinct_count_estimator_ref<T, Scope, Hash>::add(T const& item) noexcept
{
  this->impl_.add(item);
}

template <class T, cuda::thread_scope Scope, class Hash>
template <class InputIt>
__host__ constexpr void distinct_count_estimator_ref<T, Scope, Hash>::add_async(
  InputIt first, InputIt last, cuco::cuda_stream_ref stream)
{
  this->impl_.add_async(first, last, stream);
}

template <class T, cuda::thread_scope Scope, class Hash>
template <class InputIt>
__host__ constexpr void distinct_count_estimator_ref<T, Scope, Hash>::add(
  InputIt first, InputIt last, cuco::cuda_stream_ref stream)
{
  this->impl_.add(first, last, stream);
}

template <class T, cuda::thread_scope Scope, class Hash>
template <class CG, cuda::thread_scope OtherScope>
__device__ constexpr void distinct_count_estimator_ref<T, Scope, Hash>::merge(
  CG const& group, distinct_count_estimator_ref<T, OtherScope, Hash> const& other)
{
  this->impl_.merge(group, other.impl_);
}

template <class T, cuda::thread_scope Scope, class Hash>
template <cuda::thread_scope OtherScope>
__host__ constexpr void distinct_count_estimator_ref<T, Scope, Hash>::merge_async(
  distinct_count_estimator_ref<T, OtherScope, Hash> const& other, cuco::cuda_stream_ref stream)
{
  this->impl_.merge_async(other, stream);
}

template <class T, cuda::thread_scope Scope, class Hash>
template <cuda::thread_scope OtherScope>
__host__ constexpr void distinct_count_estimator_ref<T, Scope, Hash>::merge(
  distinct_count_estimator_ref<T, OtherScope, Hash> const& other, cuco::cuda_stream_ref stream)
{
  this->impl_.merge(other, stream);
}

template <class T, cuda::thread_scope Scope, class Hash>
__device__ std::size_t distinct_count_estimator_ref<T, Scope, Hash>::estimate(
  cooperative_groups::thread_block const& group) const noexcept
{
  return this->impl_.estimate(group);
}

template <class T, cuda::thread_scope Scope, class Hash>
__host__ constexpr std::size_t distinct_count_estimator_ref<T, Scope, Hash>::estimate(
  cuco::cuda_stream_ref stream) const
{
  return this->impl_.estimate(stream);
}

template <class T, cuda::thread_scope Scope, class Hash>
__host__ __device__ constexpr auto distinct_count_estimator_ref<T, Scope, Hash>::hash_function()
  const noexcept
{
  return this->impl_.hash_function();
}

template <class T, cuda::thread_scope Scope, class Hash>
__host__ __device__ constexpr cuda::std::span<std::byte>
distinct_count_estimator_ref<T, Scope, Hash>::sketch() const noexcept
{
  return this->impl_.sketch();
}

template <class T, cuda::thread_scope Scope, class Hash>
__host__ __device__ constexpr std::size_t
distinct_count_estimator_ref<T, Scope, Hash>::sketch_bytes() const noexcept
{
  return this->impl_.sketch_bytes();
}

template <class T, cuda::thread_scope Scope, class Hash>
__host__ __device__ constexpr std::size_t
distinct_count_estimator_ref<T, Scope, Hash>::sketch_bytes(
  cuco::sketch_size_kb sketch_size_kb) noexcept
{
  return impl_type::sketch_bytes(sketch_size_kb);
}

template <class T, cuda::thread_scope Scope, class Hash>
__host__ __device__ constexpr std::size_t
distinct_count_estimator_ref<T, Scope, Hash>::sketch_bytes(
  cuco::standard_deviation standard_deviation) noexcept
{
  return impl_type::sketch_bytes(standard_deviation);
}

template <class T, cuda::thread_scope Scope, class Hash>
__host__ __device__ constexpr std::size_t
distinct_count_estimator_ref<T, Scope, Hash>::sketch_alignment() noexcept
{
  return impl_type::sketch_alignment();
}

}  // namespace cuco