/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuco/cuda_stream_ref.hpp>
#include <cuco/detail/static_map/functors.cuh>
#include <cuco/detail/static_map/kernels.cuh>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/detail/utils.hpp>
#include <cuco/operator.hpp>
#include <cuco/static_map_ref.cuh>

#include <cstddef>

namespace cuco {
namespace experimental {

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
constexpr static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  static_map(Extent capacity,
             empty_key<Key> empty_key_sentinel,
             empty_value<T> empty_value_sentinel,
             KeyEqual const& pred,
             ProbingScheme const& probing_scheme,
             Allocator const& alloc,
             cuda_stream_ref stream)
  : impl_{std::make_unique<impl_type>(capacity,
                                      empty_key_sentinel,
                                      cuco::pair{empty_key_sentinel, empty_value_sentinel},
                                      pred,
                                      probing_scheme,
                                      alloc,
                                      stream)},
    empty_value_sentinel_{empty_value_sentinel}
{
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
constexpr static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  static_map(Extent n,
             double desired_load_factor,
             empty_key<Key> empty_key_sentinel,
             empty_value<T> empty_value_sentinel,
             KeyEqual const& pred,
             ProbingScheme const& probing_scheme,
             Allocator const& alloc,
             cuda_stream_ref stream)
  : impl_{std::make_unique<impl_type>(n,
                                      desired_load_factor,
                                      empty_key_sentinel,
                                      cuco::pair{empty_key_sentinel, empty_value_sentinel},
                                      pred,
                                      probing_scheme,
                                      alloc,
                                      stream)},
    empty_value_sentinel_{empty_value_sentinel}
{
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
constexpr static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  static_map(Extent capacity,
             empty_key<Key> empty_key_sentinel,
             empty_value<T> empty_value_sentinel,
             erased_key<Key> erased_key_sentinel,
             KeyEqual const& pred,
             ProbingScheme const& probing_scheme,
             Allocator const& alloc,
             cuda_stream_ref stream)
  : impl_{std::make_unique<impl_type>(capacity,
                                      empty_key_sentinel,
                                      cuco::pair{empty_key_sentinel, empty_value_sentinel},
                                      erased_key_sentinel,
                                      pred,
                                      probing_scheme,
                                      alloc,
                                      stream)},
    empty_value_sentinel_{empty_value_sentinel}
{
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::clear(
  cuda_stream_ref stream) noexcept
{
  impl_->clear(stream);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::clear_async(
  cuda_stream_ref stream) noexcept
{
  impl_->clear_async(stream);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt>
static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::size_type
static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::insert(
  InputIt first, InputIt last, cuda_stream_ref stream)
{
  return impl_->insert(first, last, ref(op::insert), stream);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::insert_async(
  InputIt first, InputIt last, cuda_stream_ref stream) noexcept
{
  impl_->insert_async(first, last, ref(op::insert), stream);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename StencilIt, typename Predicate>
static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::size_type
static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::insert_if(
  InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda_stream_ref stream)
{
  return impl_->insert_if(first, last, stencil, pred, ref(op::insert), stream);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename StencilIt, typename Predicate>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  insert_if_async(
    InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda_stream_ref stream) noexcept
{
  impl_->insert_if_async(first, last, stencil, pred, ref(op::insert), stream);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  insert_or_assign(InputIt first, InputIt last, cuda_stream_ref stream) noexcept
{
  return this->insert_or_assign_async(first, last, stream);
  stream.synchronize();
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  insert_or_assign_async(InputIt first, InputIt last, cuda_stream_ref stream) noexcept
{
  auto const num = cuco::detail::distance(first, last);
  if (num == 0) { return; }

  auto const grid_size = cuco::detail::grid_size(num, cg_size);

  static_map_ns::detail::insert_or_assign<cg_size, cuco::detail::default_block_size()>
    <<<grid_size, cuco::detail::default_block_size(), 0, stream>>>(
      first, num, ref(op::insert_or_assign));
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::erase(
  InputIt first, InputIt last, cuda_stream_ref stream)
{
  erase_async(first, last, stream);
  stream.synchronize();
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::erase_async(
  InputIt first, InputIt last, cuda_stream_ref stream)
{
  impl_->erase_async(first, last, ref(op::erase), stream);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename OutputIt>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::contains(
  InputIt first, InputIt last, OutputIt output_begin, cuda_stream_ref stream) const
{
  contains_async(first, last, output_begin, stream);
  stream.synchronize();
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename OutputIt>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::contains_async(
  InputIt first, InputIt last, OutputIt output_begin, cuda_stream_ref stream) const noexcept
{
  impl_->contains_async(first, last, output_begin, ref(op::contains), stream);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename StencilIt, typename Predicate, typename OutputIt>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::contains_if(
  InputIt first,
  InputIt last,
  StencilIt stencil,
  Predicate pred,
  OutputIt output_begin,
  cuda_stream_ref stream) const
{
  contains_if_async(first, last, stencil, pred, output_begin, stream);
  stream.synchronize();
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename StencilIt, typename Predicate, typename OutputIt>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  contains_if_async(InputIt first,
                    InputIt last,
                    StencilIt stencil,
                    Predicate pred,
                    OutputIt output_begin,
                    cuda_stream_ref stream) const noexcept
{
  impl_->contains_if_async(first, last, stencil, pred, output_begin, ref(op::contains), stream);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename OutputIt>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::find(
  InputIt first, InputIt last, OutputIt output_begin, cuda_stream_ref stream) const
{
  find_async(first, last, output_begin, stream);
  stream.synchronize();
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename OutputIt>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::find_async(
  InputIt first, InputIt last, OutputIt output_begin, cuda_stream_ref stream) const
{
  auto const num_keys = cuco::detail::distance(first, last);
  if (num_keys == 0) { return; }

  auto const grid_size = cuco::detail::grid_size(num_keys, cg_size);

  static_map_ns::detail::find<cg_size, cuco::detail::default_block_size()>
    <<<grid_size, cuco::detail::default_block_size(), 0, stream>>>(
      first, num_keys, output_begin, ref(op::find));
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename KeyOut, typename ValueOut>
std::pair<KeyOut, ValueOut>
static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::retrieve_all(
  KeyOut keys_out, ValueOut values_out, cuda_stream_ref stream) const
{
  auto const begin = thrust::make_transform_iterator(
    thrust::counting_iterator<size_type>{0},
    static_map_ns::detail::get_slot<storage_ref_type>(impl_->storage_ref()));
  auto const is_filled  = static_map_ns::detail::slot_is_filled<Key, T>(this->empty_key_sentinel(),
                                                                       this->erased_key_sentinel());
  auto zipped_out_begin = thrust::make_zip_iterator(thrust::make_tuple(keys_out, values_out));
  auto const zipped_out_end = impl_->retrieve_all(begin, zipped_out_begin, is_filled, stream);
  auto const num            = std::distance(zipped_out_begin, zipped_out_end);

  return std::make_pair(keys_out + num, values_out + num);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::rehash(
  cuda_stream_ref stream)
{
  auto const is_filled = static_map_ns::detail::slot_is_filled<Key, T>(this->empty_key_sentinel(),
                                                                       this->erased_key_sentinel());
  this->impl_->rehash(*this, is_filled, stream);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::rehash(
  size_type capacity, cuda_stream_ref stream)
{
  auto const is_filled = static_map_ns::detail::slot_is_filled<Key, T>(this->empty_key_sentinel(),
                                                                       this->erased_key_sentinel());
  auto const extent    = make_window_extent<static_map>(capacity);
  this->impl_->rehash(extent, *this, is_filled, stream);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::rehash_async(
  cuda_stream_ref stream)
{
  auto const is_filled = static_map_ns::detail::slot_is_filled<Key, T>(this->empty_key_sentinel(),
                                                                       this->erased_key_sentinel());
  this->impl_->rehash_async(*this, is_filled, stream);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::rehash_async(
  size_type capacity, cuda_stream_ref stream)
{
  auto const is_filled = static_map_ns::detail::slot_is_filled<Key, T>(this->empty_key_sentinel(),
                                                                       this->erased_key_sentinel());
  auto const extent    = make_window_extent<static_map>(capacity);
  this->impl_->rehash_async(extent, *this, is_filled, stream);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::size_type
static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::size(
  cuda_stream_ref stream) const noexcept
{
  auto const is_filled = static_map_ns::detail::slot_is_filled<Key, T>(this->empty_key_sentinel(),
                                                                       this->erased_key_sentinel());
  return impl_->size(is_filled, stream);
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
constexpr auto
static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::capacity()
  const noexcept
{
  return impl_->capacity();
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
constexpr static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::key_type
static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::empty_key_sentinel()
  const noexcept
{
  return impl_->empty_key_sentinel();
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
constexpr static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  mapped_type
  static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
    empty_value_sentinel() const noexcept
{
  return this->empty_value_sentinel_;
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
constexpr static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::key_type
static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  erased_key_sentinel() const noexcept
{
  return impl_->erased_key_sentinel();
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename... Operators>
auto static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::ref(
  Operators...) const noexcept
{
  static_assert(sizeof...(Operators), "No operators specified");
  return this->empty_key_sentinel() == this->erased_key_sentinel()
           ? ref_type<Operators...>{cuco::empty_key<key_type>(this->empty_key_sentinel()),
                                    cuco::empty_value<mapped_type>(this->empty_value_sentinel()),
                                    impl_->key_eq(),
                                    impl_->probing_scheme(),
                                    impl_->storage_ref()}
           : ref_type<Operators...>{cuco::empty_key<key_type>(this->empty_key_sentinel()),
                                    cuco::empty_value<mapped_type>(this->empty_value_sentinel()),
                                    cuco::erased_key<key_type>(this->erased_key_sentinel()),
                                    impl_->key_eq(),
                                    impl_->probing_scheme(),
                                    impl_->storage_ref()};
}
}  // namespace experimental
}  // namespace cuco
