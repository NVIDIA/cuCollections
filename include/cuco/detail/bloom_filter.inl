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

namespace cuco {

template <typename Key, cuda::thread_scope Scope, typename Allocator, typename Slot>
bloom_filter<Key, Scope, Allocator, Slot>::bloom_filter(std::size_t num_bits,
                                                        std::size_t num_hashes,
                                                        Allocator const& alloc)
  : num_bits_{SDIV(num_bits, detail::type_bits<slot_type>()) * detail::type_bits<slot_type>()},
    num_slots_{SDIV(num_bits, detail::type_bits<slot_type>())},
    num_hashes_{num_hashes},
    slot_allocator_{alloc}
{
  slots_ = std::allocator_traits<slot_allocator_type>::allocate(slot_allocator_, num_slots_);

  initialize();
}

template <typename Key, cuda::thread_scope Scope, typename Allocator, typename Slot>
bloom_filter<Key, Scope, Allocator, Slot>::~bloom_filter()
{
  std::allocator_traits<slot_allocator_type>::deallocate(slot_allocator_, slots_, num_slots_);
}

template <typename Key, cuda::thread_scope Scope, typename Allocator, typename Slot>
void bloom_filter<Key, Scope, Allocator, Slot>::initialize(cudaStream_t stream)
{
  std::size_t constexpr block_size = 256;
  std::size_t constexpr stride     = 4;
  std::size_t const grid_size      = SDIV(num_slots_, stride * block_size);

  detail::initialize<block_size><<<grid_size, block_size, 0, stream>>>(slots_, num_slots_);
}

template <typename Key, cuda::thread_scope Scope, typename Allocator, typename Slot>
template <typename InputIt, typename Hash>
void bloom_filter<Key, Scope, Allocator, Slot>::insert(InputIt first,
                                                       InputIt last,
                                                       cudaStream_t stream,
                                                       Hash hash)
{
  auto num_keys = std::distance(first, last);
  if (num_keys == 0) { return; }

  std::size_t constexpr block_size = 256;
  std::size_t constexpr stride     = 4;
  std::size_t const grid_size      = SDIV(num_keys, stride * block_size);
  detail::insert<block_size>
    <<<grid_size, block_size, 0, stream>>>(first, last, get_device_mutable_view(), hash);
}

template <typename Key, cuda::thread_scope Scope, typename Allocator, typename Slot>
template <typename InputIt, typename OutputIt, typename Hash>
void bloom_filter<Key, Scope, Allocator, Slot>::contains(
  InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream, Hash hash)
{
  auto num_keys = std::distance(first, last);
  if (num_keys == 0) { return; }

  std::size_t constexpr block_size = 256;
  std::size_t constexpr stride     = 4;
  std::size_t const grid_size      = SDIV(num_keys, stride * block_size);
  detail::contains<block_size>
    <<<grid_size, block_size, 0, stream>>>(first, last, output_begin, get_device_view(), hash);
}
//////////////////////////////////////////
template <typename Key, cuda::thread_scope Scope, typename Allocator, typename Slot>
template <typename Hash>
__device__ Slot bloom_filter<Key, Scope, Allocator, Slot>::device_view_base::key_pattern(
  Key const& key, Hash hash) const noexcept
{
  slot_type pattern        = 0;
  std::size_t k            = 0;
  std::size_t i            = 0;
  auto const h             = hash(key + 42);  // seed mitigates secondary clustering
  constexpr slot_type step = 7919;            // prime step

  while (k < num_hashes_) {
    slot_type const bit = slot_type{1} << ((h + i * step) % detail::type_bits<slot_type>());

    if (not(pattern & bit)) {
      pattern += bit;
      k++;
    }
    i++;
  }
  return pattern;
}

template <typename Key, cuda::thread_scope Scope, typename Allocator, typename Slot>
template <typename Hash>
__device__ bool bloom_filter<Key, Scope, Allocator, Slot>::device_mutable_view::insert(
  Key const& key, Hash hash) noexcept
{
  auto slot          = key_slot(key, hash);
  auto const pattern = key_pattern(key, hash);
  auto const result  = slot->fetch_or(pattern, cuda::std::memory_order_relaxed);

  // return `true` if the key's pattern was not already present in the filter,
  // else return `false`.
  return (result & pattern) != pattern;
}

template <typename Key, cuda::thread_scope Scope, typename Allocator, typename Slot>
template <typename Hash>
__device__ bool bloom_filter<Key, Scope, Allocator, Slot>::device_view::contains(Key const& key,
                                                                                 Hash hash) noexcept
{
  auto slot          = key_slot(key, hash);
  auto const pattern = key_pattern(key, hash);
  auto const result  = slot->load(cuda::std::memory_order_relaxed);

  // return `true` if the key's pattern was already present in the filter,
  // else return `false`.
  return (result & pattern) == pattern;
}
}  // namespace cuco
