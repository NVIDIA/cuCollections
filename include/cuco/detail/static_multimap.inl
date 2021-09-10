/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <thrust/tuple.h>

namespace cuco {

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::static_multimap(
  std::size_t capacity,
  Key empty_key_sentinel,
  Value empty_value_sentinel,
  cudaStream_t stream,
  Allocator const& alloc)
  : empty_key_sentinel_{empty_key_sentinel},
    empty_value_sentinel_{empty_value_sentinel},
    slot_allocator_{alloc},
    counter_allocator_{alloc},
    stream_{stream}
{
  if constexpr (uses_vector_load()) {
    capacity_ = cuco::detail::get_valid_capacity<cg_size() * vector_width()>(capacity);
  } else {
    capacity_ = cuco::detail::get_valid_capacity<cg_size()>(capacity);
  }

  d_counter_ = counter_allocator_.allocate(1, stream_);
  slots_     = slot_allocator_.allocate(get_capacity(), stream_);

  auto constexpr block_size = 128;
  int grid_size{-1};
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &grid_size,
    detail::initialize<atomic_key_type, atomic_mapped_type, Key, Value, pair_atomic_type>,
    block_size,
    0);
  int dev_id{-1};
  cudaGetDevice(&dev_id);
  int num_sms{-1};
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
  grid_size *= num_sms;

  detail::initialize<atomic_key_type, atomic_mapped_type><<<grid_size, block_size, 0, stream_>>>(
    slots_, empty_key_sentinel, empty_value_sentinel, get_capacity());
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::~static_multimap()
{
  counter_allocator_.deallocate(d_counter_, 1, stream_);
  slot_allocator_.deallocate(slots_, get_capacity(), stream_);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt>
void static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::insert(InputIt first,
                                                                          InputIt last,
                                                                          cudaStream_t stream)
{
  auto num_keys = std::distance(first, last);
  auto view     = get_device_mutable_view();

  auto constexpr block_size = 128;
  int grid_size{-1};
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &grid_size, detail::insert<block_size, cg_size(), InputIt, device_mutable_view>, block_size, 0);
  int dev_id{-1};
  cudaGetDevice(&dev_id);
  int num_sms{-1};
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
  grid_size *= num_sms;

  detail::insert<block_size, cg_size()>
    <<<grid_size, block_size, 0, stream>>>(first, first + num_keys, view);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename StencilIt, typename Predicate>
void static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::insert_if(
  InputIt first, InputIt last, StencilIt stencil, Predicate pred, cudaStream_t stream)
{
  auto num_elements = std::distance(first, last);
  auto view         = get_device_mutable_view();

  auto constexpr block_size = 128;
  int grid_size{-1};
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &grid_size,
    detail::insert_if_n<block_size, cg_size(), InputIt, StencilIt, device_mutable_view, Predicate>,
    block_size,
    0);
  int dev_id{-1};
  cudaGetDevice(&dev_id);
  int num_sms{-1};
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
  grid_size *= num_sms;

  detail::insert_if_n<block_size, cg_size()>
    <<<grid_size, block_size, 0, stream>>>(first, stencil, num_elements, view, pred);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename OutputIt, typename KeyEqual>
void static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::contains(
  InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream, KeyEqual key_equal) const
{
  auto num_keys = std::distance(first, last);
  auto view     = get_device_view();

  auto constexpr block_size = 128;
  int grid_size{-1};
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &grid_size,
    detail::contains<block_size, cg_size(), InputIt, OutputIt, device_view, KeyEqual>,
    block_size,
    0);
  int dev_id{-1};
  cudaGetDevice(&dev_id);
  int num_sms{-1};
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
  grid_size *= num_sms;

  detail::contains<block_size, cg_size()>
    <<<grid_size, block_size, 0, stream>>>(first, last, output_begin, view, key_equal);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename KeyEqual>
std::size_t static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::count(
  InputIt first, InputIt last, cudaStream_t stream, KeyEqual key_equal) const
{
  auto num_keys = std::distance(first, last);
  auto view     = get_device_view();

  constexpr bool is_outer = false;

  auto constexpr block_size = 128;
  int grid_size{-1};
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &grid_size,
    detail::count<block_size, cg_size(), is_outer, InputIt, atomic_ctr_type, device_view, KeyEqual>,
    block_size,
    0);
  int dev_id{-1};
  cudaGetDevice(&dev_id);
  int num_sms{-1};
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
  grid_size *= num_sms;

  cudaMemsetAsync(d_counter_, 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  detail::count<block_size, cg_size(), is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, d_counter_, view, key_equal);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  return h_counter;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename KeyEqual>
std::size_t static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::count_outer(
  InputIt first, InputIt last, cudaStream_t stream, KeyEqual key_equal) const
{
  auto num_keys = std::distance(first, last);
  auto view     = get_device_view();

  constexpr bool is_outer = true;

  auto constexpr block_size = 128;
  int grid_size{-1};
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &grid_size,
    detail::count<block_size, cg_size(), is_outer, InputIt, atomic_ctr_type, device_view, KeyEqual>,
    block_size,
    0);
  int dev_id{-1};
  cudaGetDevice(&dev_id);
  int num_sms{-1};
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
  grid_size *= num_sms;

  cudaMemsetAsync(d_counter_, 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  detail::count<block_size, cg_size(), is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, d_counter_, view, key_equal);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  return h_counter;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename PairEqual>
std::size_t static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::pair_count(
  InputIt first, InputIt last, PairEqual pair_equal, cudaStream_t stream) const
{
  auto num_keys = std::distance(first, last);
  auto view     = get_device_view();

  constexpr bool is_outer = false;

  auto constexpr block_size = 128;
  int grid_size{-1};
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &grid_size,
    detail::
      pair_count<block_size, cg_size(), is_outer, InputIt, atomic_ctr_type, device_view, PairEqual>,
    block_size,
    0);
  int dev_id{-1};
  cudaGetDevice(&dev_id);
  int num_sms{-1};
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
  grid_size *= num_sms;

  cudaMemsetAsync(d_counter_, 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  detail::pair_count<block_size, cg_size(), is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, d_counter_, view, pair_equal);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  return h_counter;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename PairEqual>
std::size_t static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::pair_count_outer(
  InputIt first, InputIt last, PairEqual pair_equal, cudaStream_t stream) const
{
  auto num_keys = std::distance(first, last);
  auto view     = get_device_view();

  constexpr bool is_outer = true;

  auto constexpr block_size = 128;
  int grid_size{-1};
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &grid_size,
    detail::
      pair_count<block_size, cg_size(), is_outer, InputIt, atomic_ctr_type, device_view, PairEqual>,
    block_size,
    0);
  int dev_id{-1};
  cudaGetDevice(&dev_id);
  int num_sms{-1};
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
  grid_size *= num_sms;

  cudaMemsetAsync(d_counter_, 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  detail::pair_count<block_size, cg_size(), is_outer>
    <<<grid_size, block_size, 0, stream>>>(first, last, d_counter_, view, pair_equal);
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  return h_counter;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename OutputIt, typename KeyEqual>
OutputIt static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::retrieve(
  InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream, KeyEqual key_equal) const
{
  auto num_keys = std::distance(first, last);
  auto view     = get_device_view();

  // Using per-warp buffer for vector loads and per-CG buffer for scalar loads
  constexpr auto buffer_size = uses_vector_load() ? (warp_size() * 3u) : (cg_size() * 3u);
  constexpr bool is_outer    = false;

  auto constexpr block_size = 128;
  int grid_size{-1};
  if constexpr (uses_vector_load()) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size,
                                                  detail::vectorized_retrieve<block_size,
                                                                              warp_size(),
                                                                              cg_size(),
                                                                              buffer_size,
                                                                              is_outer,
                                                                              InputIt,
                                                                              OutputIt,
                                                                              atomic_ctr_type,
                                                                              device_view,
                                                                              KeyEqual>,
                                                  block_size,
                                                  0);
  } else {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size,
                                                  detail::retrieve<block_size,
                                                                   warp_size(),
                                                                   cg_size(),
                                                                   buffer_size,
                                                                   is_outer,
                                                                   InputIt,
                                                                   OutputIt,
                                                                   atomic_ctr_type,
                                                                   device_view,
                                                                   KeyEqual>,
                                                  block_size,
                                                  0);
  }
  int dev_id{-1};
  cudaGetDevice(&dev_id);
  int num_sms{-1};
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
  grid_size *= num_sms;

  cudaMemsetAsync(d_counter_, 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  if constexpr (uses_vector_load()) {
    detail::vectorized_retrieve<block_size, warp_size(), cg_size(), buffer_size, is_outer>
      <<<grid_size, block_size, 0, stream>>>(
        first, last, output_begin, d_counter_, view, key_equal);
  } else {
    detail::retrieve<block_size, warp_size(), cg_size(), buffer_size, is_outer>
      <<<grid_size, block_size, 0, stream>>>(
        first, last, output_begin, d_counter_, view, key_equal);
  }
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  auto output_end = output_begin + h_counter;
  return output_end;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename OutputIt, typename KeyEqual>
OutputIt static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::retrieve_outer(
  InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream, KeyEqual key_equal) const
{
  auto num_keys = std::distance(first, last);
  auto view     = get_device_view();

  // Using per-warp buffer for vector loads and per-CG buffer for scalar loads
  constexpr auto buffer_size = uses_vector_load() ? (warp_size() * 3u) : (cg_size() * 3u);
  constexpr bool is_outer    = true;

  auto constexpr block_size = 128;
  int grid_size{-1};
  if constexpr (uses_vector_load()) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size,
                                                  detail::vectorized_retrieve<block_size,
                                                                              warp_size(),
                                                                              cg_size(),
                                                                              buffer_size,
                                                                              is_outer,
                                                                              InputIt,
                                                                              OutputIt,
                                                                              atomic_ctr_type,
                                                                              device_view,
                                                                              KeyEqual>,
                                                  block_size,
                                                  0);
  } else {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size,
                                                  detail::retrieve<block_size,
                                                                   warp_size(),
                                                                   cg_size(),
                                                                   buffer_size,
                                                                   is_outer,
                                                                   InputIt,
                                                                   OutputIt,
                                                                   atomic_ctr_type,
                                                                   device_view,
                                                                   KeyEqual>,
                                                  block_size,
                                                  0);
  }
  int dev_id{-1};
  cudaGetDevice(&dev_id);
  int num_sms{-1};
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
  grid_size *= num_sms;

  cudaMemsetAsync(d_counter_, 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  if constexpr (uses_vector_load()) {
    detail::vectorized_retrieve<block_size, warp_size(), cg_size(), buffer_size, is_outer>
      <<<grid_size, block_size, 0, stream>>>(
        first, last, output_begin, d_counter_, view, key_equal);
  } else {
    detail::retrieve<block_size, warp_size(), cg_size(), buffer_size, is_outer>
      <<<grid_size, block_size, 0, stream>>>(
        first, last, output_begin, d_counter_, view, key_equal);
  }
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  auto output_end = output_begin + h_counter;
  return output_end;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename OutputZipIt1, typename OutputZipIt2, typename PairEqual>
std::size_t static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::pair_retrieve(
  InputIt first,
  InputIt last,
  OutputZipIt1 probe_output_begin,
  OutputZipIt2 contained_output_begin,
  PairEqual pair_equal,
  cudaStream_t stream) const
{
  auto num_pairs = std::distance(first, last);
  auto view      = get_device_view();

  // Using per-warp buffer for vector loads and per-CG buffer for scalar loads
  constexpr auto buffer_size = uses_vector_load() ? (warp_size() * 3u) : (cg_size() * 3u);
  constexpr bool is_outer    = false;

  auto constexpr block_size = 128;
  int grid_size{-1};
  if constexpr (uses_vector_load()) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size,
                                                  detail::vectorized_pair_retrieve<block_size,
                                                                                   warp_size(),
                                                                                   cg_size(),
                                                                                   buffer_size,
                                                                                   is_outer,
                                                                                   InputIt,
                                                                                   OutputZipIt1,
                                                                                   OutputZipIt2,
                                                                                   atomic_ctr_type,
                                                                                   device_view,
                                                                                   PairEqual>,
                                                  block_size,
                                                  0);
  } else {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size,
                                                  detail::pair_retrieve<block_size,
                                                                        warp_size(),
                                                                        cg_size(),
                                                                        buffer_size,
                                                                        is_outer,
                                                                        InputIt,
                                                                        OutputZipIt1,
                                                                        OutputZipIt2,
                                                                        atomic_ctr_type,
                                                                        device_view,
                                                                        PairEqual>,
                                                  block_size,
                                                  0);
  }
  int dev_id{-1};
  cudaGetDevice(&dev_id);
  int num_sms{-1};
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
  grid_size *= num_sms;

  cudaMemsetAsync(d_counter_, 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  if constexpr (uses_vector_load()) {
    detail::vectorized_pair_retrieve<block_size, warp_size(), cg_size(), buffer_size, is_outer>
      <<<grid_size, block_size, 0, stream>>>(
        first, last, probe_output_begin, contained_output_begin, d_counter_, view, pair_equal);
  } else {
    detail::pair_retrieve<block_size, warp_size(), cg_size(), buffer_size, is_outer>
      <<<grid_size, block_size, 0, stream>>>(
        first, last, probe_output_begin, contained_output_begin, d_counter_, view, pair_equal);
  }
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  return h_counter;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename InputIt, typename OutputZipIt1, typename OutputZipIt2, typename PairEqual>
std::size_t static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::pair_retrieve_outer(
  InputIt first,
  InputIt last,
  OutputZipIt1 probe_output_begin,
  OutputZipIt2 contained_output_begin,
  PairEqual pair_equal,
  cudaStream_t stream) const
{
  auto num_pairs = std::distance(first, last);
  auto view      = get_device_view();

  // Using per-warp buffer for vector loads and per-CG buffer for scalar loads
  constexpr auto buffer_size = uses_vector_load() ? (warp_size() * 3u) : (cg_size() * 3u);
  constexpr bool is_outer    = true;

  auto constexpr block_size = 128;
  int grid_size{-1};
  if constexpr (uses_vector_load()) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size,
                                                  detail::vectorized_pair_retrieve<block_size,
                                                                                   warp_size(),
                                                                                   cg_size(),
                                                                                   buffer_size,
                                                                                   is_outer,
                                                                                   InputIt,
                                                                                   OutputZipIt1,
                                                                                   OutputZipIt2,
                                                                                   atomic_ctr_type,
                                                                                   device_view,
                                                                                   PairEqual>,
                                                  block_size,
                                                  0);
  } else {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size,
                                                  detail::pair_retrieve<block_size,
                                                                        warp_size(),
                                                                        cg_size(),
                                                                        buffer_size,
                                                                        is_outer,
                                                                        InputIt,
                                                                        OutputZipIt1,
                                                                        OutputZipIt2,
                                                                        atomic_ctr_type,
                                                                        device_view,
                                                                        PairEqual>,
                                                  block_size,
                                                  0);
  }
  int dev_id{-1};
  cudaGetDevice(&dev_id);
  int num_sms{-1};
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
  grid_size *= num_sms;

  cudaMemsetAsync(d_counter_, 0, sizeof(atomic_ctr_type), stream);
  std::size_t h_counter;

  if constexpr (uses_vector_load()) {
    detail::vectorized_pair_retrieve<block_size, warp_size(), cg_size(), buffer_size, is_outer>
      <<<grid_size, block_size, 0, stream>>>(
        first, last, probe_output_begin, contained_output_begin, d_counter_, view, pair_equal);
  } else {
    detail::pair_retrieve<block_size, warp_size(), cg_size(), buffer_size, is_outer>
      <<<grid_size, block_size, 0, stream>>>(
        first, last, probe_output_begin, contained_output_begin, d_counter_, view, pair_equal);
  }
  CUCO_CUDA_TRY(cudaMemcpyAsync(
    &h_counter, d_counter_, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream));
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  return h_counter;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view_base::load_pair_array(
  value_type* arr, const_iterator current_slot) noexcept
{
  if constexpr (sizeof(value_type) == 4) {
    auto const tmp = *reinterpret_cast<ushort4 const*>(current_slot);
    memcpy(&arr[0], &tmp, 2 * sizeof(value_type));
  } else {
    auto const tmp = *reinterpret_cast<uint4 const*>(current_slot);
    memcpy(&arr[0], &tmp, 2 * sizeof(value_type));
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
__device__
  static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::insert_result
  static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::packed_cas(
    iterator current_slot, value_type const& insert_pair) noexcept
{
  auto expected_key   = this->get_empty_key_sentinel();
  auto expected_value = this->get_empty_value_sentinel();

  cuco::detail::pair_converter<value_type> expected_pair{
    cuco::make_pair<Key, Value>(std::move(expected_key), std::move(expected_value))};
  cuco::detail::pair_converter<value_type> new_pair{insert_pair};

  auto slot = reinterpret_cast<
    cuda::atomic<typename cuco::detail::pair_converter<value_type>::packed_type, Scope>*>(
    current_slot);

  bool success = slot->compare_exchange_strong(
    expected_pair.packed, new_pair.packed, cuda::std::memory_order_relaxed);
  if (success) { return insert_result::SUCCESS; }

  return insert_result::CONTINUE;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
__device__
  static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::insert_result
  static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::
    back_to_back_cas(iterator current_slot, value_type const& insert_pair) noexcept
{
  using cuda::std::memory_order_relaxed;

  auto expected_key   = this->get_empty_key_sentinel();
  auto expected_value = this->get_empty_value_sentinel();

  // Back-to-back CAS for 8B/8B key/value pairs
  auto& slot_key   = current_slot->first;
  auto& slot_value = current_slot->second;

  bool key_success =
    slot_key.compare_exchange_strong(expected_key, insert_pair.first, memory_order_relaxed);
  bool value_success =
    slot_value.compare_exchange_strong(expected_value, insert_pair.second, memory_order_relaxed);

  if (key_success) {
    while (not value_success) {
      value_success =
        slot_value.compare_exchange_strong(expected_value = this->get_empty_value_sentinel(),
                                           insert_pair.second,
                                           memory_order_relaxed);
    }
    return insert_result::SUCCESS;
  } else if (value_success) {
    slot_value.store(this->get_empty_value_sentinel(), memory_order_relaxed);
  }

  return insert_result::CONTINUE;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
__device__
  static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::insert_result
  static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::
    cas_dependent_write(iterator current_slot, value_type const& insert_pair) noexcept
{
  using cuda::std::memory_order_relaxed;
  auto expected_key = this->get_empty_key_sentinel();

  auto& slot_key = current_slot->first;

  auto const key_success =
    slot_key.compare_exchange_strong(expected_key, insert_pair.first, memory_order_relaxed);

  if (key_success) {
    auto& slot_value = current_slot->second;
    slot_value.store(insert_pair.second, memory_order_relaxed);
    return insert_result::SUCCESS;
  }

  return insert_result::CONTINUE;
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool uses_vector_load, typename CG>
__device__ std::enable_if_t<uses_vector_load, void>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::insert_impl(
  CG g, value_type const& insert_pair) noexcept
{
  auto current_slot = initial_slot(g, insert_pair.first);
  while (true) {
    value_type arr[2];
    load_pair_array(&arr[0], current_slot);

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as
    // the sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const first_slot_is_empty =
      (detail::bitwise_compare(arr[0].first, this->get_empty_key_sentinel()));
    auto const second_slot_is_empty =
      (detail::bitwise_compare(arr[1].first, this->get_empty_key_sentinel()));
    auto const window_contains_empty = g.ballot(first_slot_is_empty or second_slot_is_empty);

    if (window_contains_empty) {
      // the first lane in the group with an empty slot will attempt the insert
      insert_result status{insert_result::CONTINUE};
      uint32_t src_lane = __ffs(window_contains_empty) - 1;
      if (g.thread_rank() == src_lane) {
        auto insert_location = first_slot_is_empty ? current_slot : current_slot + 1;
        // One single CAS operation since vector loads are dedicated to packable pairs
        status = packed_cas(insert_location, insert_pair);
      }

      // successful insert
      if (g.any(status == insert_result::SUCCESS)) { return; }
      // if we've gotten this far, a different key took our spot
      // before we could insert. We need to retry the insert on the
      // same window
    }
    // if there are no empty slots in the current window,
    // we move onto the next window
    else {
      current_slot = next_slot(current_slot);
    }
  }  // while true
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool uses_vector_load, typename CG>
__device__ std::enable_if_t<not uses_vector_load, void>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::insert_impl(
  CG g, value_type const& insert_pair) noexcept
{
  auto current_slot = initial_slot(g, insert_pair.first);

  while (true) {
    value_type slot_contents = *reinterpret_cast<value_type const*>(current_slot);
    auto const& existing_key = slot_contents.first;

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as the
    // sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty =
      detail::bitwise_compare(existing_key, this->get_empty_key_sentinel());
    auto const window_contains_empty = g.ballot(slot_is_empty);

    if (window_contains_empty) {
      // the first lane in the group with an empty slot will attempt the insert
      insert_result status{insert_result::CONTINUE};
      uint32_t src_lane = __ffs(window_contains_empty) - 1;

      if (g.thread_rank() == src_lane) {
#if __CUDA_ARCH__ < 700
        status = cas_dependent_write(current_slot, insert_pair);
#else
        status = back_to_back_cas(current_slot, insert_pair);
#endif
      }

      // successful insert
      if (g.any(status == insert_result::SUCCESS)) { return; }
      // if we've gotten this far, a different key took our spot
      // before we could insert. We need to retry the insert on the
      // same window
    }
    // if there are no empty slots in the current window,
    // we move onto the next window
    else {
      current_slot = next_slot(current_slot);
    }
  }  // while true
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_mutable_view::insert(
  CG g, value_type const& insert_pair) noexcept
{
  insert_impl<uses_vector_load()>(g, insert_pair);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool uses_vector_load, typename CG, typename KeyEqual>
__device__ std::enable_if_t<uses_vector_load, bool>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::contains_impl(
  CG g, Key const& k, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k);

  while (true) {
    value_type arr[2];
    load_pair_array(&arr[0], current_slot);

    auto const first_slot_is_empty =
      detail::bitwise_compare(arr[0].first, this->get_empty_key_sentinel());
    auto const second_slot_is_empty =
      detail::bitwise_compare(arr[1].first, this->get_empty_key_sentinel());
    auto const first_equals  = (not first_slot_is_empty and key_equal(arr[0].first, k));
    auto const second_equals = (not second_slot_is_empty and key_equal(arr[1].first, k));

    // the key we were searching for was found by one of the threads, so we return true
    if (g.any(first_equals or second_equals)) { return true; }

    // we found an empty slot, meaning that the key we're searching for isn't present
    if (g.any(first_slot_is_empty or second_slot_is_empty)) { return false; }

    // otherwise, all slots in the current window are full with other keys, so we move onto the next
    // window
    current_slot = next_slot(current_slot);
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool uses_vector_load, typename CG, typename KeyEqual>
__device__ std::enable_if_t<not uses_vector_load, bool>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::contains_impl(
  CG g, Key const& k, KeyEqual key_equal) noexcept
{
  auto current_slot = initial_slot(g, k);

  while (true) {
    value_type slot_contents = *reinterpret_cast<value_type const*>(current_slot);
    auto const& existing_key = slot_contents.first;

    // The user provide `key_equal` can never be used to compare against `empty_key_sentinel` as the
    // sentinel is not a valid key value. Therefore, first check for the sentinel
    auto const slot_is_empty =
      detail::bitwise_compare(existing_key, this->get_empty_key_sentinel());

    auto const equals = (not slot_is_empty and key_equal(existing_key, k));

    // the key we were searching for was found by one of the threads, so we return true
    if (g.any(equals)) { return true; }

    // we found an empty slot, meaning that the key we're searching for isn't present
    if (g.any(slot_is_empty)) { return false; }

    // otherwise, all slots in the current window are full with other keys, so we move onto the next
    // window
    current_slot = next_slot(current_slot);
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool uses_vector_load, bool is_outer, typename CG, typename KeyEqual>
__device__ std::enable_if_t<uses_vector_load, std::size_t>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::count_impl(
  CG const& g, Key const& k, KeyEqual key_equal) noexcept
{
  std::size_t count = 0;
  auto current_slot = initial_slot(g, k);

  [[maybe_unused]] bool found_match = false;

  while (true) {
    value_type arr[2];
    load_pair_array(&arr[0], current_slot);

    auto const first_slot_is_empty =
      detail::bitwise_compare(arr[0].first, this->get_empty_key_sentinel());
    auto const second_slot_is_empty =
      detail::bitwise_compare(arr[1].first, this->get_empty_key_sentinel());
    auto const first_equals  = (not first_slot_is_empty and key_equal(arr[0].first, k));
    auto const second_equals = (not second_slot_is_empty and key_equal(arr[1].first, k));

    if constexpr (is_outer) {
      if (g.any(first_equals or second_equals)) { found_match = true; }
    }

    count += (first_equals + second_equals);

    if (g.any(first_slot_is_empty or second_slot_is_empty)) {
      if constexpr (is_outer) {
        if ((not found_match) && (g.thread_rank() == 0)) { count++; }
      }
      return count;
    }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool uses_vector_load, bool is_outer, typename CG, typename KeyEqual>
__device__ std::enable_if_t<not uses_vector_load, std::size_t>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::count_impl(
  CG const& g, Key const& k, KeyEqual key_equal) noexcept
{
  std::size_t count = 0;
  auto current_slot = initial_slot(g, k);

  [[maybe_unused]] bool found_match = false;

  while (true) {
    value_type slot_contents = *reinterpret_cast<value_type const*>(current_slot);
    auto const& current_key  = slot_contents.first;

    auto const slot_is_empty = detail::bitwise_compare(current_key, this->get_empty_key_sentinel());
    auto const equals        = not slot_is_empty and key_equal(current_key, k);

    if constexpr (is_outer) {
      if (g.any(equals)) { found_match = true; }
    }

    count += equals;

    if (g.any(slot_is_empty)) {
      if constexpr (is_outer) {
        if ((not found_match) && (g.thread_rank() == 0)) { count++; }
      }
      return count;
    }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool uses_vector_load, bool is_outer, typename CG, typename PairEqual>
__device__ std::enable_if_t<uses_vector_load, std::size_t>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::pair_count_impl(
  CG const& g, value_type const& pair, PairEqual pair_equal) noexcept
{
  std::size_t count = 0;
  auto key          = pair.first;
  auto current_slot = initial_slot(g, key);

  [[maybe_unused]] bool found_match = false;

  while (true) {
    value_type arr[2];
    load_pair_array(&arr[0], current_slot);

    auto const first_slot_is_empty =
      detail::bitwise_compare(arr[0].first, this->get_empty_key_sentinel());
    auto const second_slot_is_empty =
      detail::bitwise_compare(arr[1].first, this->get_empty_key_sentinel());

    auto const first_slot_equals  = (not first_slot_is_empty and pair_equal(arr[0], pair));
    auto const second_slot_equals = (not second_slot_is_empty and pair_equal(arr[1], pair));

    if constexpr (is_outer) {
      if (g.any(first_slot_equals or second_slot_equals)) { found_match = true; }
    }

    count += (first_slot_equals + second_slot_equals);

    if (g.any(first_slot_is_empty or second_slot_is_empty)) {
      if constexpr (is_outer) {
        if ((not found_match) && (g.thread_rank() == 0)) { count++; }
      }
      return count;
    }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <bool uses_vector_load, bool is_outer, typename CG, typename PairEqual>
__device__ std::enable_if_t<not uses_vector_load, std::size_t>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::pair_count_impl(
  CG const& g, value_type const& pair, PairEqual pair_equal) noexcept
{
  std::size_t count = 0;
  auto key          = pair.first;
  auto current_slot = initial_slot(g, key);

  [[maybe_unused]] bool found_match = false;

  while (true) {
    auto slot_contents = *reinterpret_cast<value_type const*>(current_slot);

    auto const slot_is_empty =
      detail::bitwise_compare(slot_contents.first, this->get_empty_key_sentinel());

    auto const equals = not slot_is_empty and pair_equal(slot_contents, pair);

    if constexpr (is_outer) {
      if (g.any(equals)) { found_match = true; }
    }

    count += equals;

    if (g.any(slot_is_empty)) {
      if constexpr (is_outer) {
        if ((not found_match) && (g.thread_rank() == 0)) { count++; }
      }
      return count;
    }

    current_slot = next_slot(current_slot);
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t buffer_size,
          bool is_outer,
          typename warpT,
          typename CG,
          typename atomicT,
          typename OutputIt,
          typename KeyEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::retrieve_impl(
  warpT const& warp,
  CG const& g,
  Key const& k,
  uint32_t* warp_counter,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin,
  KeyEqual key_equal) noexcept
{
  const uint32_t cg_lane_id = g.thread_rank();

  auto current_slot = initial_slot(g, k);

  bool running                      = true;
  [[maybe_unused]] bool found_match = false;

  while (warp.any(running)) {
    if (running) {
      value_type arr[2];
      load_pair_array(&arr[0], current_slot);

      auto const first_slot_is_empty =
        detail::bitwise_compare(arr[0].first, this->get_empty_key_sentinel());
      auto const second_slot_is_empty =
        detail::bitwise_compare(arr[1].first, this->get_empty_key_sentinel());
      auto const first_equals  = (not first_slot_is_empty and key_equal(arr[0].first, k));
      auto const second_equals = (not second_slot_is_empty and key_equal(arr[1].first, k));
      auto const first_exists  = g.ballot(first_equals);
      auto const second_exists = g.ballot(second_equals);

      if (first_exists or second_exists) {
        if constexpr (is_outer) { found_match = true; }

        auto num_first_matches  = __popc(first_exists);
        auto num_second_matches = __popc(second_exists);

        uint32_t output_idx;
        if (0 == cg_lane_id) {
          output_idx = atomicAdd(warp_counter, (num_first_matches + num_second_matches));
        }
        output_idx = g.shfl(output_idx, 0);

        if (first_equals) {
          auto lane_offset = __popc(first_exists & ((1 << cg_lane_id) - 1));
          Key key          = k;
          output_buffer[output_idx + lane_offset] =
            cuco::make_pair<Key, Value>(std::move(key), std::move(arr[0].second));
        }
        if (second_equals) {
          auto lane_offset = __popc(second_exists & ((1 << cg_lane_id) - 1));
          Key key          = k;
          output_buffer[output_idx + num_first_matches + lane_offset] =
            cuco::make_pair<Key, Value>(std::move(key), std::move(arr[1].second));
        }
      }
      if (g.any(first_slot_is_empty or second_slot_is_empty)) {
        running = false;
        if constexpr (is_outer) {
          if ((not found_match) && (cg_lane_id == 0)) {
            auto output_idx           = atomicAdd(warp_counter, 1);
            Key key                   = k;
            output_buffer[output_idx] = cuco::make_pair<Key, Value>(
              std::move(key), std::move(this->get_empty_value_sentinel()));
          }
        }
      }
    }  // if running

    warp.sync();
    if (*warp_counter + warp.size() * vector_width() > buffer_size) {
      flush_output_buffer(warp, *warp_counter, output_buffer, num_matches, output_begin);
      // First lane reset warp-level counter
      if (warp.thread_rank() == 0) { *warp_counter = 0; }
    }

    current_slot = next_slot(current_slot);
  }  // while running
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t cg_size,
          uint32_t buffer_size,
          bool is_outer,
          typename CG,
          typename atomicT,
          typename OutputIt,
          typename KeyEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::retrieve_impl(
  CG const& g,
  Key const& k,
  uint32_t* cg_counter,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin,
  KeyEqual key_equal) noexcept
{
  const uint32_t lane_id = g.thread_rank();

  auto current_slot = initial_slot(g, k);

  bool running                      = true;
  [[maybe_unused]] bool found_match = false;

  while (running) {
    // TODO: Replace reinterpret_cast with atomic ref when possible. The current implementation
    // is unsafe!
    static_assert(sizeof(Key) == sizeof(cuda::atomic<Key>));
    static_assert(sizeof(Value) == sizeof(cuda::atomic<Value>));
    value_type slot_contents = *reinterpret_cast<value_type const*>(current_slot);

    auto const slot_is_empty =
      detail::bitwise_compare(slot_contents.first, this->get_empty_key_sentinel());
    auto const equals   = (not slot_is_empty and key_equal(slot_contents.first, k));
    uint32_t output_idx = *cg_counter;
    auto const exists   = g.ballot(equals);

    if (exists) {
      if constexpr (is_outer) { found_match = true; }
      auto num_matches = __popc(exists);
      if (equals) {
        // Each match computes its lane-level offset
        auto lane_offset = __popc(exists & ((1 << lane_id) - 1));
        Key key          = k;
        output_buffer[output_idx + lane_offset] =
          cuco::make_pair<Key, Value>(std::move(key), std::move(slot_contents.second));
      }
      if (0 == lane_id) { (*cg_counter) += num_matches; }
    }
    if (g.any(slot_is_empty)) {
      running = false;
      if constexpr (is_outer) {
        if ((not found_match) && (lane_id == 0)) {
          output_idx                = (*cg_counter)++;
          Key key                   = k;
          output_buffer[output_idx] = cuco::make_pair<Key, Value>(
            std::move(key), std::move(this->get_empty_value_sentinel()));
        }
      }
    }

    g.sync();

    // Flush if the next iteration won't fit into buffer
    if ((*cg_counter + cg_size) > buffer_size) {
      flush_output_buffer(g, *cg_counter, output_buffer, num_matches, output_begin);
      // First lane reset CG-level counter
      if (lane_id == 0) { *cg_counter = 0; }
    }
    current_slot = next_slot(current_slot);
  }  // while running
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t buffer_size,
          bool is_outer,
          typename warpT,
          typename CG,
          typename atomicT,
          typename OutputZipIt1,
          typename OutputZipIt2,
          typename PairEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::pair_retrieve_impl(
  warpT const& warp,
  CG const& g,
  value_type const& pair,
  uint32_t* warp_counter,
  value_type* probe_output_buffer,
  value_type* contained_output_buffer,
  atomicT* num_matches,
  OutputZipIt1 probe_output_begin,
  OutputZipIt2 contained_output_begin,
  PairEqual pair_equal) noexcept
{
  const uint32_t cg_lane_id = g.thread_rank();

  auto key          = pair.first;
  auto current_slot = initial_slot(g, key);

  bool running                      = true;
  [[maybe_unused]] bool found_match = false;

  while (warp.any(running)) {
    if (running) {
      value_type arr[2];
      load_pair_array(&arr[0], current_slot);

      auto const first_slot_is_empty =
        detail::bitwise_compare(arr[0].first, this->get_empty_key_sentinel());
      auto const second_slot_is_empty =
        detail::bitwise_compare(arr[1].first, this->get_empty_key_sentinel());
      auto const first_equals  = (not first_slot_is_empty and pair_equal(arr[0], pair));
      auto const second_equals = (not second_slot_is_empty and pair_equal(arr[1], pair));
      auto const first_exists  = g.ballot(first_equals);
      auto const second_exists = g.ballot(second_equals);

      if (first_exists or second_exists) {
        if constexpr (is_outer) { found_match = true; }

        auto num_first_matches  = __popc(first_exists);
        auto num_second_matches = __popc(second_exists);

        uint32_t output_idx;
        if (0 == cg_lane_id) {
          output_idx = atomicAdd(warp_counter, (num_first_matches + num_second_matches));
        }
        output_idx = g.shfl(output_idx, 0);

        if (first_equals) {
          auto lane_offset = __popc(first_exists & ((1 << cg_lane_id) - 1));
          probe_output_buffer[output_idx + lane_offset]     = pair;
          contained_output_buffer[output_idx + lane_offset] = arr[0];
        }
        if (second_equals) {
          auto lane_offset = __popc(second_exists & ((1 << cg_lane_id) - 1));
          probe_output_buffer[output_idx + num_first_matches + lane_offset]     = pair;
          contained_output_buffer[output_idx + num_first_matches + lane_offset] = arr[1];
        }
      }
      if (g.any(first_slot_is_empty or second_slot_is_empty)) {
        running = false;
        if constexpr (is_outer) {
          if ((not found_match) && (cg_lane_id == 0)) {
            auto output_idx                 = atomicAdd(warp_counter, 1);
            probe_output_buffer[output_idx] = pair;
            contained_output_buffer[output_idx] =
              cuco::make_pair<Key, Value>(std::move(this->get_empty_key_sentinel()),
                                          std::move(this->get_empty_value_sentinel()));
          }
        }
      }
    }  // if running

    warp.sync();
    if (*warp_counter + warp.size() * vector_width() > buffer_size) {
      flush_output_buffer(warp,
                          *warp_counter,
                          probe_output_buffer,
                          contained_output_buffer,
                          num_matches,
                          probe_output_begin,
                          contained_output_begin);
      // First lane reset warp-level counter
      if (warp.thread_rank() == 0) { *warp_counter = 0; }
    }

    current_slot = next_slot(current_slot);
  }  // while running
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t cg_size,
          uint32_t buffer_size,
          bool is_outer,
          typename CG,
          typename atomicT,
          typename OutputZipIt1,
          typename OutputZipIt2,
          typename PairEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::pair_retrieve_impl(
  CG const& g,
  value_type const& pair,
  uint32_t* cg_counter,
  value_type* probe_output_buffer,
  value_type* contained_output_buffer,
  atomicT* num_matches,
  OutputZipIt1 probe_output_begin,
  OutputZipIt2 contained_output_begin,
  PairEqual pair_equal) noexcept
{
  const uint32_t lane_id = g.thread_rank();

  auto key          = pair.first;
  auto current_slot = initial_slot(g, key);

  bool running                      = true;
  [[maybe_unused]] bool found_match = false;

  while (running) {
    // TODO: Replace reinterpret_cast with atomic ref when possible. The current implementation
    // is unsafe!
    static_assert(sizeof(Key) == sizeof(cuda::atomic<Key>));
    static_assert(sizeof(Value) == sizeof(cuda::atomic<Value>));
    value_type slot_contents = *reinterpret_cast<value_type const*>(current_slot);

    auto const slot_is_empty =
      detail::bitwise_compare(slot_contents.first, this->get_empty_key_sentinel());
    auto const equals   = (not slot_is_empty and pair_equal(slot_contents, pair));
    uint32_t output_idx = *cg_counter;
    auto const exists   = g.ballot(equals);

    if (exists) {
      if constexpr (is_outer) { found_match = true; }
      auto num_matches = __popc(exists);
      if (equals) {
        // Each match computes its lane-level offset
        auto lane_offset                                  = __popc(exists & ((1 << lane_id) - 1));
        probe_output_buffer[output_idx + lane_offset]     = pair;
        contained_output_buffer[output_idx + lane_offset] = slot_contents;
      }
      if (0 == lane_id) { (*cg_counter) += num_matches; }
    }
    if (g.any(slot_is_empty)) {
      running = false;
      if constexpr (is_outer) {
        if ((not found_match) && (lane_id == 0)) {
          output_idx                          = (*cg_counter)++;
          probe_output_buffer[output_idx]     = pair;
          contained_output_buffer[output_idx] = cuco::make_pair<Key, Value>(
            std::move(this->get_empty_key_sentinel()), std::move(this->get_empty_value_sentinel()));
        }
      }
    }

    g.sync();

    // Flush if the next iteration won't fit into buffer
    if ((*cg_counter + cg_size) > buffer_size) {
      flush_output_buffer(g,
                          *cg_counter,
                          probe_output_buffer,
                          contained_output_buffer,
                          num_matches,
                          probe_output_begin,
                          contained_output_begin);
      // First lane reset CG-level counter
      if (lane_id == 0) { *cg_counter = 0; }
    }
    current_slot = next_slot(current_slot);
  }  // while running
}

// public APIs

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename atomicT, typename OutputIt>
__inline__ __device__
  std::enable_if_t<thrust::is_contiguous_iterator<OutputIt>::value and SUPPORTS_CG_MEMCPY_ASYNC,
                   void>
  static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::flush_output_buffer(
    CG const& g,
    uint32_t const num_outputs,
    value_type* output_buffer,
    atomicT* num_matches,
    OutputIt output_begin) noexcept
{
  std::size_t offset;
  const auto lane_id = g.thread_rank();
  if (0 == lane_id) {
    offset = num_matches->fetch_add(num_outputs, cuda::std::memory_order_relaxed);
  }
  offset = g.shfl(offset, 0);

#if defined(CUCO_HAS_CUDA_BARRIER)
  cooperative_groups::memcpy_async(
    g,
    output_begin + offset,
    output_buffer,
    cuda::aligned_size_t<alignof(value_type)>(sizeof(value_type) * num_outputs));
#else
  cooperative_groups::memcpy_async(
    g, output_begin + offset, output_buffer, sizeof(value_type) * num_outputs);
#endif
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename atomicT, typename OutputIt>
__inline__ __device__ std::enable_if_t<not(thrust::is_contiguous_iterator<OutputIt>::value and
                                           SUPPORTS_CG_MEMCPY_ASYNC),
                                       void>
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::flush_output_buffer(
  CG const& g,
  uint32_t const num_outputs,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin) noexcept
{
  std::size_t offset;
  const auto lane_id = g.thread_rank();
  if (0 == lane_id) {
    offset = num_matches->fetch_add(num_outputs, cuda::std::memory_order_relaxed);
  }
  offset = g.shfl(offset, 0);

  for (auto index = lane_id; index < num_outputs; index += g.size()) {
    *(output_begin + offset + index) = output_buffer[index];
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename atomicT, typename OutputZipIt1, typename OutputZipIt2>
__inline__ __device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::flush_output_buffer(
  CG const& g,
  uint32_t const num_outputs,
  value_type* probe_output_buffer,
  value_type* contained_output_buffer,
  atomicT* num_matches,
  OutputZipIt1 probe_output_begin,
  OutputZipIt2 contained_output_begin) noexcept
{
  std::size_t offset;
  const auto lane_id = g.thread_rank();
  if (0 == lane_id) {
    offset = num_matches->fetch_add(num_outputs, cuda::std::memory_order_relaxed);
  }
  offset = g.shfl(offset, 0);

  for (auto index = lane_id; index < num_outputs; index += g.size()) {
    auto& probe_pair                                           = probe_output_buffer[index];
    auto& contained_pair                                       = contained_output_buffer[index];
    thrust::get<0>(*(probe_output_begin + offset + index))     = probe_pair.first;
    thrust::get<1>(*(probe_output_begin + offset + index))     = probe_pair.second;
    thrust::get<0>(*(contained_output_begin + offset + index)) = contained_pair.first;
    thrust::get<1>(*(contained_output_begin + offset + index)) = contained_pair.second;
  }
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename KeyEqual>
__device__ bool static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::contains(
  CG g, Key const& k, KeyEqual key_equal) noexcept
{
  return contains_impl<uses_vector_load()>(g, k, key_equal);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename KeyEqual>
__device__ std::size_t
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::count(
  CG const& g, Key const& k, KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = false;
  return count_impl<uses_vector_load(), is_outer>(g, k, key_equal);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename KeyEqual>
__device__ std::size_t
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::count_outer(
  CG const& g, Key const& k, KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = true;
  return count_impl<uses_vector_load(), is_outer>(g, k, key_equal);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename PairEqual>
__device__ std::size_t
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::pair_count(
  CG const& g, value_type const& pair, PairEqual pair_equal) noexcept
{
  constexpr bool is_outer = false;
  return pair_count_impl<uses_vector_load(), is_outer>(g, pair, pair_equal);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <typename CG, typename PairEqual>
__device__ std::size_t
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::pair_count_outer(
  CG const& g, value_type const& pair, PairEqual pair_equal) noexcept
{
  constexpr bool is_outer = true;
  return pair_count_impl<uses_vector_load(), is_outer>(g, pair, pair_equal);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t buffer_size,
          typename warpT,
          typename CG,
          typename atomicT,
          typename OutputIt,
          typename KeyEqual>
__device__ void static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::retrieve(
  warpT const& warp,
  CG const& g,
  Key const& k,
  uint32_t* warp_counter,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin,
  KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = false;
  retrieve_impl<buffer_size, is_outer>(
    warp, g, k, warp_counter, output_buffer, num_matches, output_begin);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t buffer_size,
          typename warpT,
          typename CG,
          typename atomicT,
          typename OutputIt,
          typename KeyEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::retrieve_outer(
  warpT const& warp,
  CG const& g,
  Key const& k,
  uint32_t* warp_counter,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin,
  KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = true;
  retrieve_impl<buffer_size, is_outer>(
    warp, g, k, warp_counter, output_buffer, num_matches, output_begin);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t cg_size,
          uint32_t buffer_size,
          typename CG,
          typename atomicT,
          typename OutputIt,
          typename KeyEqual>
__device__ void static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::retrieve(
  CG const& g,
  Key const& k,
  uint32_t* cg_counter,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin,
  KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = false;
  retrieve_impl<cg_size, buffer_size, is_outer>(
    g, k, cg_counter, output_buffer, num_matches, output_begin);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t cg_size,
          uint32_t buffer_size,
          typename CG,
          typename atomicT,
          typename OutputIt,
          typename KeyEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::retrieve_outer(
  CG const& g,
  Key const& k,
  uint32_t* cg_counter,
  value_type* output_buffer,
  atomicT* num_matches,
  OutputIt output_begin,
  KeyEqual key_equal) noexcept
{
  constexpr bool is_outer = true;
  retrieve_impl<cg_size, buffer_size, is_outer>(
    g, k, cg_counter, output_buffer, num_matches, output_begin);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t buffer_size,
          typename warpT,
          typename CG,
          typename atomicT,
          typename OutputZipIt1,
          typename OutputZipIt2,
          typename PairEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::pair_retrieve(
  warpT const& warp,
  CG const& g,
  value_type const& pair,
  uint32_t* warp_counter,
  value_type* probe_output_buffer,
  value_type* contained_output_buffer,
  atomicT* num_matches,
  OutputZipIt1 probe_output_begin,
  OutputZipIt2 contained_output_begin,
  PairEqual pair_equal) noexcept
{
  constexpr bool is_outer = false;
  pair_retrieve_impl<buffer_size, is_outer>(warp,
                                            g,
                                            pair,
                                            warp_counter,
                                            probe_output_buffer,
                                            contained_output_buffer,
                                            num_matches,
                                            probe_output_begin,
                                            contained_output_begin,
                                            pair_equal);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t buffer_size,
          typename warpT,
          typename CG,
          typename atomicT,
          typename OutputZipIt1,
          typename OutputZipIt2,
          typename PairEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::pair_retrieve_outer(
  warpT const& warp,
  CG const& g,
  value_type const& pair,
  uint32_t* warp_counter,
  value_type* probe_output_buffer,
  value_type* contained_output_buffer,
  atomicT* num_matches,
  OutputZipIt1 probe_output_begin,
  OutputZipIt2 contained_output_begin,
  PairEqual pair_equal) noexcept
{
  constexpr bool is_outer = true;
  pair_retrieve_impl<buffer_size, is_outer>(warp,
                                            g,
                                            pair,
                                            warp_counter,
                                            probe_output_buffer,
                                            contained_output_buffer,
                                            num_matches,
                                            probe_output_begin,
                                            contained_output_begin,
                                            pair_equal);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t cg_size,
          uint32_t buffer_size,
          typename CG,
          typename atomicT,
          typename OutputZipIt1,
          typename OutputZipIt2,
          typename PairEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::pair_retrieve(
  CG const& g,
  value_type const& pair,
  uint32_t* cg_counter,
  value_type* probe_output_buffer,
  value_type* contained_output_buffer,
  atomicT* num_matches,
  OutputZipIt1 probe_output_begin,
  OutputZipIt2 contained_output_begin,
  PairEqual pair_equal) noexcept
{
  constexpr bool is_outer = false;
  pair_retrieve_impl<cg_size, buffer_size, is_outer>(g,
                                                     pair,
                                                     cg_counter,
                                                     probe_output_buffer,
                                                     contained_output_buffer,
                                                     num_matches,
                                                     probe_output_begin,
                                                     contained_output_begin,
                                                     pair_equal);
}

template <typename Key,
          typename Value,
          class ProbeSequence,
          cuda::thread_scope Scope,
          typename Allocator>
template <uint32_t cg_size,
          uint32_t buffer_size,
          typename CG,
          typename atomicT,
          typename OutputZipIt1,
          typename OutputZipIt2,
          typename PairEqual>
__device__ void
static_multimap<Key, Value, ProbeSequence, Scope, Allocator>::device_view::pair_retrieve_outer(
  CG const& g,
  value_type const& pair,
  uint32_t* cg_counter,
  value_type* probe_output_buffer,
  value_type* contained_output_buffer,
  atomicT* num_matches,
  OutputZipIt1 probe_output_begin,
  OutputZipIt2 contained_output_begin,
  PairEqual pair_equal) noexcept
{
  constexpr bool is_outer = true;
  pair_retrieve_impl<cg_size, buffer_size, is_outer>(g,
                                                     pair,
                                                     cg_counter,
                                                     probe_output_buffer,
                                                     contained_output_buffer,
                                                     num_matches,
                                                     probe_output_begin,
                                                     contained_output_begin,
                                                     pair_equal);
}

}  // namespace cuco
