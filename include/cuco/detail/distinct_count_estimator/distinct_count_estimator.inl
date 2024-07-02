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

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr distinct_count_estimator<T, Scope, Hash, Allocator>::distinct_count_estimator(
  cuco::sketch_size_kb sketch_size_kb,
  Hash const& hash,
  Allocator const& alloc,
  cuda::stream_ref stream)
  : impl_{std::make_unique<impl_type>(sketch_size_kb, hash, alloc, stream)}
{
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr distinct_count_estimator<T, Scope, Hash, Allocator>::distinct_count_estimator(
  cuco::standard_deviation standard_deviation,
  Hash const& hash,
  Allocator const& alloc,
  cuda::stream_ref stream)
  : impl_{std::make_unique<impl_type>(standard_deviation, hash, alloc, stream)}
{
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr void distinct_count_estimator<T, Scope, Hash, Allocator>::clear_async(
  cuda::stream_ref stream) noexcept
{
  this->impl_->clear_async(stream);
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr void distinct_count_estimator<T, Scope, Hash, Allocator>::clear(cuda::stream_ref stream)
{
  this->impl_->clear(stream);
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
template <class InputIt>
constexpr void distinct_count_estimator<T, Scope, Hash, Allocator>::add_async(
  InputIt first, InputIt last, cuda::stream_ref stream)
{
  this->impl_->add_async(first, last, stream);
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
template <class InputIt>
constexpr void distinct_count_estimator<T, Scope, Hash, Allocator>::add(InputIt first,
                                                                        InputIt last,
                                                                        cuda::stream_ref stream)
{
  this->impl_->add(first, last, stream);
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
template <cuda::thread_scope OtherScope, class OtherAllocator>
constexpr void distinct_count_estimator<T, Scope, Hash, Allocator>::merge_async(
  distinct_count_estimator<T, OtherScope, Hash, OtherAllocator> const& other,
  cuda::stream_ref stream)
{
  this->impl_->merge_async(*(other.impl_), stream);
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
template <cuda::thread_scope OtherScope, class OtherAllocator>
constexpr void distinct_count_estimator<T, Scope, Hash, Allocator>::merge(
  distinct_count_estimator<T, OtherScope, Hash, OtherAllocator> const& other,
  cuda::stream_ref stream)
{
  this->impl_->merge(*(other.impl_), stream);
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
template <cuda::thread_scope OtherScope>
constexpr void distinct_count_estimator<T, Scope, Hash, Allocator>::merge_async(
  ref_type<OtherScope> const& other_ref, cuda::stream_ref stream)
{
  this->impl_->merge_async(other_ref.impl_, stream);
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
template <cuda::thread_scope OtherScope>
constexpr void distinct_count_estimator<T, Scope, Hash, Allocator>::merge(
  ref_type<OtherScope> const& other_ref, cuda::stream_ref stream)
{
  this->impl_->merge(other_ref.impl_, stream);
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr std::size_t distinct_count_estimator<T, Scope, Hash, Allocator>::estimate(
  cuda::stream_ref stream) const
{
  return this->impl_->estimate(stream);
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr typename distinct_count_estimator<T, Scope, Hash, Allocator>::ref_type<>
distinct_count_estimator<T, Scope, Hash, Allocator>::ref() const noexcept
{
  return {this->sketch(), this->hash_function()};
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr auto distinct_count_estimator<T, Scope, Hash, Allocator>::hash_function() const noexcept
{
  return this->impl_->hash_function();
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr cuda::std::span<std::byte> distinct_count_estimator<T, Scope, Hash, Allocator>::sketch()
  const noexcept
{
  return this->impl_->sketch();
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr size_t distinct_count_estimator<T, Scope, Hash, Allocator>::sketch_bytes() const noexcept
{
  return this->impl_->sketch_bytes();
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr size_t distinct_count_estimator<T, Scope, Hash, Allocator>::sketch_bytes(
  cuco::sketch_size_kb sketch_size_kb) noexcept
{
  return impl_type::sketch_bytes(sketch_size_kb);
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr size_t distinct_count_estimator<T, Scope, Hash, Allocator>::sketch_bytes(
  cuco::standard_deviation standard_deviation) noexcept
{
  return impl_type::sketch_bytes(standard_deviation);
}

template <class T, cuda::thread_scope Scope, class Hash, class Allocator>
constexpr size_t distinct_count_estimator<T, Scope, Hash, Allocator>::sketch_alignment() noexcept
{
  return impl_type::sketch_alignment();
}

}  // namespace cuco