/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cuco/detail/utils.cuh>
#include <cuco/detail/utils.hpp>

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>

#include <iterator>

namespace cuco {

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::static_multimap(
  std::size_t capacity,
  sentinel::empty_key<Key> empty_key_sentinel,
  sentinel::empty_value<Value> empty_value_sentinel,
  cudaStream_t stream,
  Allocator const& alloc)
  : capacity_{cuco::detail::get_valid_capacity<cg_size(), vector_width(), uses_vector_load()>(
      capacity)},
    empty_key_sentinel_{empty_key_sentinel.value},
    empty_value_sentinel_{empty_value_sentinel.value},
    counter_allocator_{alloc},
    slot_allocator_{alloc},
    delete_counter_{counter_allocator_},
    delete_slots_{slot_allocator_, capacity_},
    d_counter_{counter_allocator_.allocate(1), delete_counter_},
    slots_{slot_allocator_.allocate(capacity_), delete_slots_}
{
  auto constexpr block_size = 128;
  auto constexpr stride     = 4;
  auto const grid_size      = (get_capacity() + stride * block_size - 1) / (stride * block_size);

  detail::initialize<atomic_key_type, atomic_mapped_type><<<grid_size, block_size, 0, stream>>>(
    slots_.get(), empty_key_sentinel_, empty_value_sentinel_, get_capacity());
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename InputIt>
void static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::insert(InputIt first,
                                                                          InputIt last,
                                                                          cudaStream_t stream)
{
  auto const num_keys = std::distance(first, last);
  if (num_keys == 0) { return; }

  auto constexpr block_size = 128;
  auto constexpr stride     = 1;
  auto const grid_size = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view            = get_device_mutable_view();

  detail::insert<block_size, cg_size()>
    <<<grid_size, block_size, 0, stream>>>(first, first + num_keys, view);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename InputIt, typename StencilIt, typename Predicate>
void static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::insert_if(
  InputIt first, InputIt last, StencilIt stencil, Predicate pred, cudaStream_t stream)
{
  auto const num_keys = std::distance(first, last);
  if (num_keys == 0) { return; }

  auto constexpr block_size = 128;
  auto constexpr stride     = 1;
  auto const grid_size = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view            = get_device_mutable_view();

  detail::insert_if_n<block_size, cg_size()>
    <<<grid_size, block_size, 0, stream>>>(first, stencil, num_keys, view, pred);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename InputIt, typename OutputIt, typename KeyEqual>
void static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::contains(
  InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream, KeyEqual key_equal) const
{
  auto const num_keys = std::distance(first, last);
  if (num_keys == 0) { return; }

  auto constexpr is_pair_contains = false;
  auto constexpr block_size       = 128;
  auto constexpr stride           = 1;
  auto const grid_size = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);
  auto view            = get_device_view();

  detail::contains<is_pair_contains, block_size, cg_size()>
    <<<grid_size, block_size, 0, stream>>>(first, last, output_begin, view, key_equal);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename InputIt, typename OutputIt, typename PairEqual>
void static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::pair_contains(
  InputIt first, InputIt last, OutputIt output_begin, PairEqual pair_equal, cudaStream_t stream)
  const
{
  auto const num_pairs = std::distance(first, last);
  if (num_pairs == 0) { return; }

  auto constexpr is_pair_contains = true;
  auto constexpr block_size       = 128;
  auto constexpr stride           = 1;
  auto const grid_size = (cg_size() * num_pairs + stride * block_size - 1) / (stride * block_size);
  auto view            = get_device_view();

  detail::contains<is_pair_contains, block_size, cg_size()>
    <<<grid_size, block_size, 0, stream>>>(first, last, output_begin, view, pair_equal);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename InputIt, typename KeyEqual>
std::size_t static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::count(
  InputIt first, InputIt last, cudaStream_t stream, KeyEqual key_equal) const
{
  auto const num_keys = std::distance(first, last);
  if (num_keys == 0) { return 0; }

  auto constexpr is_outer   = false;
  auto constexpr block_size = 128;
  auto constexpr stride     = 1;

  auto view            = get_device_view();
  auto const grid_size = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);

  cudaMemsetAsync(d_counter_.get(), 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  detail::count<block_size, cg_size(), is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, d_counter_.get(), view, key_equal);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_.get(), sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  return h_counter;
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename InputIt, typename KeyEqual>
std::size_t static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::count_outer(
  InputIt first, InputIt last, cudaStream_t stream, KeyEqual key_equal) const
{
  auto const num_keys = std::distance(first, last);
  if (num_keys == 0) { return 0; }

  auto constexpr is_outer   = true;
  auto constexpr block_size = 128;
  auto constexpr stride     = 1;

  auto view            = get_device_view();
  auto const grid_size = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);

  cudaMemsetAsync(d_counter_.get(), 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  detail::count<block_size, cg_size(), is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, d_counter_.get(), view, key_equal);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_.get(), sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  return h_counter;
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename InputIt, typename PairEqual>
std::size_t static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::pair_count(
  InputIt first, InputIt last, PairEqual pair_equal, cudaStream_t stream) const
{
  auto const num_keys = std::distance(first, last);
  if (num_keys == 0) { return 0; }

  auto constexpr is_outer   = false;
  auto constexpr block_size = 128;
  auto constexpr stride     = 1;

  auto view            = get_device_view();
  auto const grid_size = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);

  cudaMemsetAsync(d_counter_.get(), 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  detail::pair_count<block_size, cg_size(), is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, d_counter_.get(), view, pair_equal);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_.get(), sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  return h_counter;
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename InputIt, typename PairEqual>
std::size_t static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::pair_count_outer(
  InputIt first, InputIt last, PairEqual pair_equal, cudaStream_t stream) const
{
  auto const num_keys = std::distance(first, last);
  if (num_keys == 0) { return 0; }

  auto constexpr is_outer   = true;
  auto constexpr block_size = 128;
  auto constexpr stride     = 1;

  auto view            = get_device_view();
  auto const grid_size = (cg_size() * num_keys + stride * block_size - 1) / (stride * block_size);

  cudaMemsetAsync(d_counter_.get(), 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  detail::pair_count<block_size, cg_size(), is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, d_counter_.get(), view, pair_equal);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_.get(), sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  return h_counter;
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename InputIt, typename OutputIt, typename KeyEqual>
OutputIt static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::retrieve(
  InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream, KeyEqual key_equal) const
{
  auto const num_keys = std::distance(first, last);
  if (num_keys == 0) { return output_begin; }

  // Using per-warp buffer for vector loads and per-CG buffer for scalar loads
  constexpr auto buffer_size = uses_vector_load() ? (warp_size() * 3u) : (cg_size() * 3u);
  constexpr auto block_size  = 128;
  constexpr auto is_outer    = false;

  auto view                   = get_device_view();
  auto const flushing_cg_size = [&]() {
    if constexpr (uses_vector_load()) { return warp_size(); }
    return cg_size();
  }();

  auto const grid_size = detail::get_grid_size(detail::retrieve<block_size,
                                                                flushing_cg_size,
                                                                cg_size(),
                                                                buffer_size,
                                                                is_outer,
                                                                InputIt,
                                                                OutputIt,
                                                                atomic_ctr_type,
                                                                device_view,
                                                                KeyEqual>,
                                               block_size);

  cudaMemsetAsync(d_counter_.get(), 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  detail::retrieve<block_size, flushing_cg_size, cg_size(), buffer_size, is_outer>
    <<<grid_size, block_size, 0, stream>>>(
      first, last, output_begin, d_counter_.get(), view, key_equal);

  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_.get(), sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  auto output_end = output_begin + h_counter;
  return output_end;
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename InputIt, typename OutputIt, typename KeyEqual>
OutputIt static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::retrieve_outer(
  InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream, KeyEqual key_equal) const
{
  auto const num_keys = std::distance(first, last);
  if (num_keys == 0) { return output_begin; }

  // Using per-warp buffer for vector loads and per-CG buffer for scalar loads
  constexpr auto buffer_size = uses_vector_load() ? (warp_size() * 3u) : (cg_size() * 3u);
  constexpr auto block_size  = 128;
  constexpr auto is_outer    = true;

  auto view                   = get_device_view();
  auto const flushing_cg_size = [&]() {
    if constexpr (uses_vector_load()) { return warp_size(); }
    return cg_size();
  }();

  auto const grid_size = detail::get_grid_size(detail::retrieve<block_size,
                                                                flushing_cg_size,
                                                                cg_size(),
                                                                buffer_size,
                                                                is_outer,
                                                                InputIt,
                                                                OutputIt,
                                                                atomic_ctr_type,
                                                                device_view,
                                                                KeyEqual>,
                                               block_size);

  cudaMemsetAsync(d_counter_.get(), 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  detail::retrieve<block_size, flushing_cg_size, cg_size(), buffer_size, is_outer>
    <<<grid_size, block_size, 0, stream>>>(
      first, last, output_begin, d_counter_.get(), view, key_equal);

  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_.get(), sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  auto output_end = output_begin + h_counter;
  return output_end;
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename InputIt, typename OutputIt1, typename OutputIt2, typename PairEqual>
std::pair<OutputIt1, OutputIt2>
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::pair_retrieve(
  InputIt first,
  InputIt last,
  OutputIt1 probe_output_begin,
  OutputIt2 contained_output_begin,
  PairEqual pair_equal,
  cudaStream_t stream) const
{
  auto const num_pairs = std::distance(first, last);
  if (num_pairs == 0) { return std::make_pair(probe_output_begin, contained_output_begin); }

  // Using per-warp buffer for vector loads and per-CG buffer for scalar loads
  constexpr auto buffer_size = uses_vector_load() ? (warp_size() * 3u) : (cg_size() * 3u);
  constexpr auto block_size  = 128;
  constexpr auto is_outer    = false;
  constexpr auto stride      = 1;

  auto view                   = get_device_view();
  auto const flushing_cg_size = [&]() {
    if constexpr (uses_vector_load()) { return warp_size(); }
    return cg_size();
  }();
  auto const grid_size = (cg_size() * num_pairs + stride * block_size - 1) / (stride * block_size);

  cudaMemsetAsync(d_counter_.get(), 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  detail::pair_retrieve<block_size, flushing_cg_size, cg_size(), buffer_size, is_outer>
    <<<grid_size, block_size, 0, stream>>>(
      first, last, probe_output_begin, contained_output_begin, d_counter_.get(), view, pair_equal);

  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_.get(), sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  return std::make_pair(probe_output_begin + h_counter, contained_output_begin + h_counter);
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename InputIt, typename OutputIt1, typename OutputIt2, typename PairEqual>
std::pair<OutputIt1, OutputIt2>
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::pair_retrieve_outer(
  InputIt first,
  InputIt last,
  OutputIt1 probe_output_begin,
  OutputIt2 contained_output_begin,
  PairEqual pair_equal,
  cudaStream_t stream) const
{
  auto const num_pairs = std::distance(first, last);
  if (num_pairs == 0) { return std::make_pair(probe_output_begin, contained_output_begin); }

  // Using per-warp buffer for vector loads and per-CG buffer for scalar loads
  constexpr auto buffer_size = uses_vector_load() ? (warp_size() * 3u) : (cg_size() * 3u);
  constexpr auto block_size  = 128;
  constexpr auto is_outer    = true;
  constexpr auto stride      = 1;

  auto view                   = get_device_view();
  auto const flushing_cg_size = [&]() {
    if constexpr (uses_vector_load()) { return warp_size(); }
    return cg_size();
  }();
  auto const grid_size = (cg_size() * num_pairs + stride * block_size - 1) / (stride * block_size);

  cudaMemsetAsync(d_counter_.get(), 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  detail::pair_retrieve<block_size, flushing_cg_size, cg_size(), buffer_size, is_outer>
    <<<grid_size, block_size, 0, stream>>>(
      first, last, probe_output_begin, contained_output_begin, d_counter_.get(), view, pair_equal);

  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_.get(), sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  return std::make_pair(probe_output_begin + h_counter, contained_output_begin + h_counter);
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
__device__ __forceinline__ void
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_mutable_view::insert(
  cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
  value_type const& insert_pair) noexcept
{
  impl_.insert<uses_vector_load()>(g, insert_pair);
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename CG>
__device__ __forceinline__ static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::make_copy(
  CG g, pair_atomic_type* const memory_to_use, device_view source_device_view) noexcept
{
#if defined(CUCO_HAS_CUDA_BARRIER)
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
  if (g.thread_rank() == 0) { init(&barrier, g.size()); }
  g.sync();

  cuda::memcpy_async(g,
                     memory_to_use,
                     source_device_view.get_slots(),
                     sizeof(pair_atomic_type) * source_device_view.get_capacity(),
                     barrier);

  barrier.arrive_and_wait();
#else
  pair_atomic_type const* const slots_ptr = source_device_view.get_slots();
  for (std::size_t i = g.thread_rank(); i < source_device_view.get_capacity(); i += g.size()) {
    new (&memory_to_use[i].first)
      atomic_key_type{slots_ptr[i].first.load(cuda::memory_order_relaxed)};
    new (&memory_to_use[i].second)
      atomic_mapped_type{slots_ptr[i].second.load(cuda::memory_order_relaxed)};
  }
  g.sync();
#endif

  return device_view(memory_to_use,
                     source_device_view.get_capacity(),
                     source_device_view.get_empty_key_sentinel(),
                     source_device_view.get_empty_value_sentinel());
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename CG, typename atomicT, typename OutputIt>
__device__ __forceinline__ void
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::flush_output_buffer(
  CG const& g,
  uint32_t const num_outputs,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin) noexcept
{
  impl_.flush_output_buffer(g, num_outputs, output_buffer, num_matches, output_begin);
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename CG, typename atomicT, typename OutputIt1, typename OutputIt2>
__device__ __forceinline__ void
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::flush_output_buffer(
  CG const& g,
  uint32_t const num_outputs,
  value_type* probe_output_buffer,
  value_type* contained_output_buffer,
  atomicT* num_matches,
  OutputIt1 probe_output_begin,
  OutputIt2 contained_output_begin) noexcept
{
  impl_.flush_output_buffer(g,
                            num_outputs,
                            probe_output_buffer,
                            contained_output_buffer,
                            num_matches,
                            probe_output_begin,
                            contained_output_begin);
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename KeyEqual>
__device__ __forceinline__ bool
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::contains(
  cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
  Key const& k,
  KeyEqual key_equal) noexcept
{
  constexpr bool is_pair_contains = false;
  return impl_.contains<is_pair_contains, uses_vector_load()>(g, k, key_equal);
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename PairEqual>
__device__ __forceinline__ bool
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::pair_contains(
  cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
  value_type const& p,
  PairEqual pair_equal) noexcept
{
  constexpr bool is_pair_contains = true;
  return impl_.contains<is_pair_contains, uses_vector_load()>(g, p, pair_equal);
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename KeyEqual>
__device__ __forceinline__ std::size_t
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::count(
  cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
  Key const& k,
  KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = false;
  return impl_.count<uses_vector_load(), is_outer>(g, k, key_equal);
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename KeyEqual>
__device__ __forceinline__ std::size_t
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::count_outer(
  cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
  Key const& k,
  KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = true;
  return impl_.count<uses_vector_load(), is_outer>(g, k, key_equal);
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename PairEqual>
__device__ __forceinline__ std::size_t
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::pair_count(
  cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
  value_type const& pair,
  PairEqual pair_equal) noexcept
{
  constexpr bool is_outer = false;
  return impl_.pair_count<uses_vector_load(), is_outer>(g, pair, pair_equal);
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename PairEqual>
__device__ __forceinline__ std::size_t
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::pair_count_outer(
  cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& g,
  value_type const& pair,
  PairEqual pair_equal) noexcept
{
  constexpr bool is_outer = true;
  return impl_.pair_count<uses_vector_load(), is_outer>(g, pair, pair_equal);
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <uint32_t buffer_size,
          typename FlushingCG,
          typename atomicT,
          typename OutputIt,
          typename KeyEqual>
__device__ __forceinline__ void
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::retrieve(
  FlushingCG const& flushing_cg,
  cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& probing_cg,
  Key const& k,
  uint32_t* flushing_cg_counter,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin,
  KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = false;
  if constexpr (uses_vector_load()) {
    impl_.retrieve<buffer_size, is_outer>(flushing_cg,
                                          probing_cg,
                                          k,
                                          flushing_cg_counter,
                                          output_buffer,
                                          num_matches,
                                          output_begin,
                                          key_equal);
  } else  // In the case of scalar load, flushing CG is the same as probing CG
  {
    impl_.retrieve<buffer_size, is_outer>(
      probing_cg, k, flushing_cg_counter, output_buffer, num_matches, output_begin, key_equal);
  }
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <uint32_t buffer_size,
          typename FlushingCG,
          typename atomicT,
          typename OutputIt,
          typename KeyEqual>
__device__ __forceinline__ void
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::retrieve_outer(
  FlushingCG const& flushing_cg,
  cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& probing_cg,
  Key const& k,
  uint32_t* flushing_cg_counter,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin,
  KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = true;
  if constexpr (uses_vector_load()) {
    impl_.retrieve<buffer_size, is_outer>(flushing_cg,
                                          probing_cg,
                                          k,
                                          flushing_cg_counter,
                                          output_buffer,
                                          num_matches,
                                          output_begin,
                                          key_equal);
  } else  // In the case of scalar load, flushing CG is the same as probing CG
  {
    impl_.retrieve<buffer_size, is_outer>(
      probing_cg, k, flushing_cg_counter, output_buffer, num_matches, output_begin, key_equal);
  }
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename OutputIt1,
          typename OutputIt2,
          typename OutputIt3,
          typename OutputIt4,
          typename PairEqual>
__device__ __forceinline__ void
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::pair_retrieve(
  cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& probing_cg,
  value_type const& pair,
  OutputIt1 probe_key_begin,
  OutputIt2 probe_val_begin,
  OutputIt3 contained_key_begin,
  OutputIt4 contained_val_begin,
  PairEqual pair_equal) noexcept
{
  constexpr bool is_outer = false;
  impl_.pair_retrieve<is_outer, uses_vector_load()>(probing_cg,
                                                    pair,
                                                    probe_key_begin,
                                                    probe_val_begin,
                                                    contained_key_begin,
                                                    contained_val_begin,
                                                    pair_equal);
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <uint32_t buffer_size,
          typename FlushingCG,
          typename atomicT,
          typename OutputIt1,
          typename OutputIt2,
          typename PairEqual>
__device__ __forceinline__ void
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::pair_retrieve(
  FlushingCG const& flushing_cg,
  cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& probing_cg,
  value_type const& pair,
  uint32_t* flushing_cg_counter,
  value_type* probe_output_buffer,
  value_type* contained_output_buffer,
  atomicT* num_matches,
  OutputIt1 probe_output_begin,
  OutputIt2 contained_output_begin,
  PairEqual pair_equal) noexcept
{
  constexpr bool is_outer = false;
  if constexpr (uses_vector_load()) {
    impl_.pair_retrieve<buffer_size, is_outer>(flushing_cg,
                                               probing_cg,
                                               pair,
                                               flushing_cg_counter,
                                               probe_output_buffer,
                                               contained_output_buffer,
                                               num_matches,
                                               probe_output_begin,
                                               contained_output_begin,
                                               pair_equal);
  } else  // In the case of scalar load, flushing CG is the same as probing CG
  {
    impl_.pair_retrieve<buffer_size, is_outer>(probing_cg,
                                               pair,
                                               flushing_cg_counter,
                                               probe_output_buffer,
                                               contained_output_buffer,
                                               num_matches,
                                               probe_output_begin,
                                               contained_output_begin,
                                               pair_equal);
  }
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <typename OutputIt1,
          typename OutputIt2,
          typename OutputIt3,
          typename OutputIt4,
          typename PairEqual>
__device__ __forceinline__ void
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::pair_retrieve_outer(
  cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& probing_cg,
  value_type const& pair,
  OutputIt1 probe_key_begin,
  OutputIt2 probe_val_begin,
  OutputIt3 contained_key_begin,
  OutputIt4 contained_val_begin,
  PairEqual pair_equal) noexcept
{
  constexpr bool is_outer = true;
  impl_.pair_retrieve<is_outer, uses_vector_load()>(probing_cg,
                                                    pair,
                                                    probe_key_begin,
                                                    probe_val_begin,
                                                    contained_key_begin,
                                                    contained_val_begin,
                                                    pair_equal);
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
template <uint32_t buffer_size,
          typename FlushingCG,
          typename atomicT,
          typename OutputIt1,
          typename OutputIt2,
          typename PairEqual>
__device__ __forceinline__ void
static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::device_view::pair_retrieve_outer(
  FlushingCG const& flushing_cg,
  cooperative_groups::thread_block_tile<ProbeSequence::cg_size> const& probing_cg,
  value_type const& pair,
  uint32_t* flushing_cg_counter,
  value_type* probe_output_buffer,
  value_type* contained_output_buffer,
  atomicT* num_matches,
  OutputIt1 probe_output_begin,
  OutputIt2 contained_output_begin,
  PairEqual pair_equal) noexcept
{
  constexpr bool is_outer = true;
  if constexpr (uses_vector_load()) {
    impl_.pair_retrieve<buffer_size, is_outer>(flushing_cg,
                                               probing_cg,
                                               pair,
                                               flushing_cg_counter,
                                               probe_output_buffer,
                                               contained_output_buffer,
                                               num_matches,
                                               probe_output_begin,
                                               contained_output_begin,
                                               pair_equal);
  } else  // In the case of scalar load, flushing CG is the same as probing CG
  {
    impl_.pair_retrieve<buffer_size, is_outer>(probing_cg,
                                               pair,
                                               flushing_cg_counter,
                                               probe_output_buffer,
                                               contained_output_buffer,
                                               num_matches,
                                               probe_output_begin,
                                               contained_output_begin,
                                               pair_equal);
  }
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
std::size_t static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::get_size(
  cudaStream_t stream) const noexcept
{
  auto begin  = thrust::make_transform_iterator(raw_slots(), detail::slot_to_tuple<Key, Value>{});
  auto filled = cuco::detail::slot_is_filled<Key>{get_empty_key_sentinel()};

  return thrust::count_if(thrust::cuda::par.on(stream), begin, begin + get_capacity(), filled);
}

template <typename Key,
          typename Value,
          cuda::thread_scope Scope,
          typename Allocator,
          class ProbeSequence>
float static_multimap<Key, Value, Scope, Allocator, ProbeSequence>::get_load_factor(
  cudaStream_t stream) const noexcept
{
  auto size = get_size(stream);
  return static_cast<float>(size) / static_cast<float>(capacity_);
}

}  // namespace cuco
