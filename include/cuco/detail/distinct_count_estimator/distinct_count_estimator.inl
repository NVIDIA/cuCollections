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

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::distinct_count_estimator(
  Hash const& hash, Allocator const& alloc, cuco::cuda_stream_ref stream)
  : impl_{std::make_unique<impl_type>(hash, alloc, stream)}
{
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
void distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::clear_async(
  cuco::cuda_stream_ref stream) noexcept
{
  this->impl_->clear_async(stream);
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
void distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::clear(
  cuco::cuda_stream_ref stream)
{
  this->impl_->clear(stream);
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
template <class InputIt>
void distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::add_async(
  InputIt first, InputIt last, cuco::cuda_stream_ref stream) noexcept
{
  this->impl_->add_async(first, last, stream);
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
template <class InputIt>
void distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::add(
  InputIt first, InputIt last, cuco::cuda_stream_ref stream)
{
  this->impl_->add(first, last, stream);
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
template <cuda::thread_scope OtherScope, class OtherAllocator>
void distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::merge_async(
  distinct_count_estimator<T, Precision, OtherScope, Hash, OtherAllocator> const& other,
  cuco::cuda_stream_ref stream) noexcept
{
  this->impl_->merge_async(other, stream);
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
template <cuda::thread_scope OtherScope, class OtherAllocator>
void distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::merge(
  distinct_count_estimator<T, Precision, OtherScope, Hash, OtherAllocator> const& other,
  cuco::cuda_stream_ref stream)
{
  this->impl_->merge(other, stream);
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
template <cuda::thread_scope OtherScope>
void distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::merge_async(
  ref_type<OtherScope> const& other, cuco::cuda_stream_ref stream) noexcept
{
  this->impl_->merge_async(other, stream);
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
template <cuda::thread_scope OtherScope>
void distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::merge(
  ref_type<OtherScope> const& other, cuco::cuda_stream_ref stream)
{
  this->impl_->merge(other, stream);
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
std::size_t distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::estimate(
  cuco::cuda_stream_ref stream) const
{
  return this->impl_->estimate(stream);
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
typename distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::ref_type<>
distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::ref() const noexcept
{
  return {this->sketch(), this->hash()};
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
auto distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::hash() const noexcept
{
  return this->impl_->hash();
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
auto distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::sketch() const noexcept
{
  return this->impl_->sketch();
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr size_t
distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::sketch_bytes() noexcept
{
  return impl_type::sketch_bytes();
}

template <class T, int32_t Precision, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr size_t
distinct_count_estimator<T, Precision, Scope, Hash, Allocator>::sketch_alignment() noexcept
{
  return impl_type::sketch();
}

}  // namespace cuco