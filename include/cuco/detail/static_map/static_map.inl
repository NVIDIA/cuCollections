/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuco/detail/bitwise_compare.cuh>
#include <cuco/detail/static_map/kernels.cuh>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/detail/utils.hpp>
#include <cuco/operator.hpp>
#include <cuco/static_map_ref.cuh>

#include <cuda/stream_ref>

#include <algorithm>
#include <cstddef>

namespace cuco {

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
             cuda_thread_scope<Scope>,
             Storage,
             Allocator const& alloc,
             cuda::stream_ref stream)
  : impl_{std::make_unique<impl_type>(capacity,
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
             cuda_thread_scope<Scope>,
             Storage,
             Allocator const& alloc,
             cuda::stream_ref stream)
  : impl_{std::make_unique<impl_type>(n,
                                      desired_load_factor,
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
             cuda_thread_scope<Scope>,
             Storage,
             Allocator const& alloc,
             cuda::stream_ref stream)
  : impl_{std::make_unique<impl_type>(capacity,
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
  cuda::stream_ref stream)
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
  cuda::stream_ref stream) noexcept
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
  InputIt first, InputIt last, cuda::stream_ref stream)
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
  InputIt first, InputIt last, cuda::stream_ref stream) noexcept
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
template <typename InputIt, typename FoundIt, typename InsertedIt>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  insert_and_find(InputIt first,
                  InputIt last,
                  FoundIt found_begin,
                  InsertedIt inserted_begin,
                  cuda::stream_ref stream)
{
  insert_and_find_async(first, last, found_begin, inserted_begin, stream);
  stream.wait();
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename FoundIt, typename InsertedIt>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  insert_and_find_async(InputIt first,
                        InputIt last,
                        FoundIt found_begin,
                        InsertedIt inserted_begin,
                        cuda::stream_ref stream) noexcept
{
  impl_->insert_and_find_async(
    first, last, found_begin, inserted_begin, ref(op::insert_and_find), stream);
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
  InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream)
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
  insert_if_async(InputIt first,
                  InputIt last,
                  StencilIt stencil,
                  Predicate pred,
                  cuda::stream_ref stream) noexcept
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
  insert_or_assign(InputIt first, InputIt last, cuda::stream_ref stream)
{
  this->insert_or_assign_async(first, last, stream);
  stream.wait();
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
  insert_or_assign_async(InputIt first, InputIt last, cuda::stream_ref stream) noexcept
{
  auto const num = cuco::detail::distance(first, last);
  if (num == 0) { return; }

  auto const grid_size = cuco::detail::grid_size(num, cg_size);

  static_map_ns::detail::insert_or_assign<cg_size, cuco::detail::default_block_size()>
    <<<grid_size, cuco::detail::default_block_size(), 0, stream.get()>>>(
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
template <typename InputIt, typename Op>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  insert_or_apply(InputIt first, InputIt last, Op op, cuda::stream_ref stream)
{
  this->insert_or_apply_async(first, last, op, stream);
  stream.wait();
}

template <class Key,
          class T,
          class Extent,
          cuda::thread_scope Scope,
          class KeyEqual,
          class ProbingScheme,
          class Allocator,
          class Storage>
template <typename InputIt, typename Op>
void static_map<Key, T, Extent, Scope, KeyEqual, ProbingScheme, Allocator, Storage>::
  insert_or_apply_async(InputIt first, InputIt last, Op op, cuda::stream_ref stream) noexcept
{
  auto const num = cuco::detail::distance(first, last);
  if (num == 0) { return; }

  using shmem_size_type = int32_t;

  int32_t constexpr shmem_block_size = 1024;
  int32_t const default_grid_size    = cuco::detail::grid_size(num, cg_size);

  shmem_size_type constexpr cardinality_threshold   = shmem_block_size;
  shmem_size_type constexpr shared_map_num_elements = cardinality_threshold + shmem_block_size;
  float constexpr load_factor                       = 0.7;
  shmem_size_type constexpr shared_map_size =
    static_cast<shmem_size_type>((1.0 / load_factor) * shared_map_num_elements);

  using extent_type            = cuco::extent<shmem_size_type, shared_map_size>;
  using shared_map_type        = cuco::static_map<Key,
                                                  T,
                                                  extent_type,
                                                  cuda::thread_scope_block,
                                                  KeyEqual,
                                                  ProbingScheme,
                                                  Allocator,
                                                  cuco::storage<1>>;
  using shared_map_ref_type    = typename shared_map_type::ref_type<>;
  auto constexpr window_extent = cuco::make_window_extent<shared_map_ref_type>(extent_type{});

  using ref_type = decltype(ref(op::insert_or_apply));

  auto insert_or_apply_shmem_fn_ptr = static_map_ns::detail::
    insert_or_apply_shmem<cg_size, shmem_block_size, shared_map_ref_type, InputIt, Op, ref_type>;

  int32_t const max_op_grid_size =
    cuco::detail::max_occupancy_grid_size(shmem_block_size, insert_or_apply_shmem_fn_ptr);

  auto const shmem_grid_size      = std::min(default_grid_size, max_op_grid_size);
  auto const num_loops_per_thread = num / (shmem_grid_size * shmem_block_size);

  // use shared_memory only if each thread has atleast 2 elements to process
  if (num_loops_per_thread > 2) {
    static_map_ns::detail::insert_or_apply_shmem<cg_size, shmem_block_size, shared_map_ref_type>
      <<<shmem_grid_size, shmem_block_size, 0, stream.get()>>>(
        first, num, op, ref(op::insert_or_apply), window_extent);
  } else {
    static_map_ns::detail::insert_or_apply<cg_size, cuco::detail::default_block_size()>
      <<<default_grid_size, cuco::detail::default_block_size(), 0, stream.get()>>>(
        first, num, op, ref(op::insert_or_apply));
  }
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
  InputIt first, InputIt last, cuda::stream_ref stream)
{
  erase_async(first, last, stream);
  stream.wait();
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
  InputIt first, InputIt last, cuda::stream_ref stream)
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
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const
{
  contains_async(first, last, output_begin, stream);
  stream.wait();
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
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const noexcept
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
  cuda::stream_ref stream) const
{
  contains_if_async(first, last, stencil, pred, output_begin, stream);
  stream.wait();
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
                    cuda::stream_ref stream) const noexcept
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
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const
{
  find_async(first, last, output_begin, stream);
  stream.wait();
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
  InputIt first, InputIt last, OutputIt output_begin, cuda::stream_ref stream) const
{
  impl_->find_async(first, last, output_begin, ref(op::find), stream);
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
  KeyOut keys_out, ValueOut values_out, cuda::stream_ref stream) const
{
  auto const zipped_out_begin = thrust::make_zip_iterator(thrust::make_tuple(keys_out, values_out));
  auto const zipped_out_end   = impl_->retrieve_all(zipped_out_begin, stream);
  auto const num              = std::distance(zipped_out_begin, zipped_out_end);

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
  cuda::stream_ref stream)
{
  this->impl_->rehash(*this, stream);
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
  size_type capacity, cuda::stream_ref stream)
{
  auto const extent = make_window_extent<static_map>(capacity);
  this->impl_->rehash(extent, *this, stream);
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
  cuda::stream_ref stream)
{
  this->impl_->rehash_async(*this, stream);
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
  size_type capacity, cuda::stream_ref stream)
{
  auto const extent = make_window_extent<static_map>(capacity);
  this->impl_->rehash_async(extent, *this, stream);
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
  cuda::stream_ref stream) const
{
  return impl_->size(stream);
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
  return cuco::detail::bitwise_compare(this->empty_key_sentinel(), this->erased_key_sentinel())
           ? ref_type<Operators...>{cuco::empty_key<key_type>(this->empty_key_sentinel()),
                                    cuco::empty_value<mapped_type>(this->empty_value_sentinel()),
                                    impl_->key_eq(),
                                    impl_->probing_scheme(),
                                    cuda_thread_scope<Scope>{},
                                    impl_->storage_ref()}
           : ref_type<Operators...>{cuco::empty_key<key_type>(this->empty_key_sentinel()),
                                    cuco::empty_value<mapped_type>(this->empty_value_sentinel()),
                                    cuco::erased_key<key_type>(this->erased_key_sentinel()),
                                    impl_->key_eq(),
                                    impl_->probing_scheme(),
                                    cuda_thread_scope<Scope>{},
                                    impl_->storage_ref()};
}
}  // namespace cuco
