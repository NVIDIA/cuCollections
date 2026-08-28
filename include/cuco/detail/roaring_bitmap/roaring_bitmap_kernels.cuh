/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/roaring_bitmap/util.cuh>
#include <cuco/detail/utility/cuda.cuh>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/memory>

namespace cuco::experimental::detail {

struct roaring_bitmap_build_state {
  cuda::std::int64_t num_keys;
  cuda::std::int64_t num_containers;
  cuda::std::uint32_t size_bytes;
  cuda::std::uint32_t num_array_containers;
  cuda::std::uint32_t num_bitset_containers;
};

template <class KeyIt>
struct is_container_start {
  KeyIt keys;
  roaring_bitmap_build_state const* state;

  __device__ bool operator()(cuda::std::int64_t index) const noexcept
  {
    auto const num_keys = state->num_keys;
    if (index >= num_keys) { return false; }
    if (index == 0) { return true; }
    return (keys[index] >> 16) != (keys[index - 1] >> 16);
  }
};

template <class KeyIt>
is_container_start(KeyIt, roaring_bitmap_build_state const*) -> is_container_start<KeyIt>;

struct container_payload_size {
  cuda::std::int64_t const* container_starts;
  roaring_bitmap_build_state const* state;

  __device__ cuda::std::uint32_t operator()(cuda::std::int64_t index) const noexcept
  {
    using metadata_type = roaring_bitmap_metadata<cuda::std::uint32_t>;

    auto const num_containers = state->num_containers;
    if (index >= num_containers) { return 0; }

    auto const begin = container_starts[index];
    auto const end   = index + 1 < num_containers ? container_starts[index + 1] : state->num_keys;
    auto const cardinality = static_cast<cuda::std::uint32_t>(end - begin);
    return cardinality <= metadata_type::max_array_container_card
             ? cardinality * sizeof(cuda::std::uint16_t)
             : metadata_type::bitset_container_bytes;
  }
};

template <class ContainerStartIt>
CUCO_KERNEL void compute_container_payload_sizes(cuda::std::uint32_t* payload_sizes,
                                                 cuda::std::int64_t num_container_slots,
                                                 ContainerStartIt container_starts,
                                                 roaring_bitmap_build_state const* state)
{
  auto const index = cuco::detail::global_thread_id();
  if (index >= num_container_slots) { return; }
  // Slots beyond the selected container count contribute zero to the fixed-size scan.
  payload_sizes[index] = container_payload_size{container_starts, state}(index);
}

template <class ContainerStartIt, class PayloadOffsetIt>
CUCO_KERNEL void compute_roaring_bitmap_build_size(roaring_bitmap_build_state* state,
                                                   ContainerStartIt container_starts,
                                                   PayloadOffsetIt payload_offsets)
{
  using metadata_type = roaring_bitmap_metadata<cuda::std::uint32_t>;

  if (blockIdx.x != 0 || threadIdx.x != 0) { return; }

  auto const num_containers = state->num_containers;
  if (num_containers == 0) {
    state->size_bytes = 2 * sizeof(cuda::std::uint32_t);
    return;
  }

  auto const last          = num_containers - 1;
  auto const begin         = container_starts[last];
  auto const cardinality   = static_cast<cuda::std::uint32_t>(state->num_keys - begin);
  auto const payload_size  = cardinality <= metadata_type::max_array_container_card
                               ? cardinality * sizeof(cuda::std::uint16_t)
                               : metadata_type::bitset_container_bytes;
  auto const payload_begin = 2 * sizeof(cuda::std::uint32_t) +
                             static_cast<cuda::std::uint32_t>(num_containers) *
                               (2 * sizeof(cuda::std::uint16_t) + sizeof(cuda::std::uint32_t));

  state->size_bytes = payload_begin + payload_offsets[last] + payload_size;
}

template <class ContainerStartIt>
CUCO_KERNEL void collect_container_indexes(cuda::std::uint32_t* container_indexes,
                                           cuda::std::int64_t num_container_slots,
                                           ContainerStartIt container_starts,
                                           roaring_bitmap_build_state* state)
{
  using metadata_type = roaring_bitmap_metadata<cuda::std::uint32_t>;

  auto const index = cuco::detail::global_thread_id();
  if (index >= state->num_containers || index >= num_container_slots) { return; }

  auto const begin = container_starts[index];
  auto const end =
    index + 1 < state->num_containers ? container_starts[index + 1] : state->num_keys;
  auto const cardinality = static_cast<cuda::std::uint32_t>(end - begin);
  if (cardinality <= metadata_type::max_array_container_card) {
    // The queue does not need to preserve container order: writers use the stored container index
    // to find the final payload offset.
    auto const output_index         = atomicAdd(&state->num_array_containers, 1);
    container_indexes[output_index] = static_cast<cuda::std::uint32_t>(index);
  } else {
    // Bitset indexes grow backward so both work lists share one num_container_slots-sized buffer.
    auto const output_index = atomicAdd(&state->num_bitset_containers, 1);
    container_indexes[num_container_slots - output_index - 1] =
      static_cast<cuda::std::uint32_t>(index);
  }
}

template <class KeyIt>
__device__ cuda::std::int64_t lower_bound_low_bits(KeyIt keys,
                                                   cuda::std::int64_t first,
                                                   cuda::std::int64_t last,
                                                   cuda::std::uint32_t value)
{
  while (first < last) {
    auto const middle = first + (last - first) / 2;
    auto const lower  = static_cast<cuda::std::uint16_t>(keys[middle]);
    if (lower < value) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  return first;
}

template <class KeyIt, class ContainerStartIt, class PayloadOffsetIt>
CUCO_KERNEL void write_roaring_bitmap_header(cuda::std::byte* bitmap,
                                             KeyIt keys,
                                             roaring_bitmap_build_state state,
                                             ContainerStartIt container_starts,
                                             PayloadOffsetIt payload_offsets)
{
  using metadata_type = roaring_bitmap_metadata<cuda::std::uint32_t>;

  auto const index = cuco::detail::global_thread_id();

  if (index == 0) {
    misaligned_store(bitmap, metadata_type::serial_cookie_no_runcontainer);
    misaligned_store(bitmap + sizeof(cuda::std::uint32_t),
                     static_cast<cuda::std::uint32_t>(state.num_containers));
  }

  if (index >= state.num_containers) { return; }

  auto const begin = container_starts[index];
  auto const end = index + 1 < state.num_containers ? container_starts[index + 1] : state.num_keys;
  auto const cardinality    = static_cast<cuda::std::uint32_t>(end - begin);
  auto const key            = static_cast<cuda::std::uint16_t>(keys[begin] >> 16);
  auto const card_minus_one = static_cast<cuda::std::uint16_t>(cardinality - 1);

  auto* const key_cards = bitmap + 2 * sizeof(cuda::std::uint32_t);
  misaligned_store(key_cards + index * 2 * sizeof(cuda::std::uint16_t), key);
  misaligned_store(key_cards + (index * 2 + 1) * sizeof(cuda::std::uint16_t), card_minus_one);

  auto* const offsets      = key_cards + state.num_containers * 2 * sizeof(cuda::std::uint16_t);
  auto const payload_begin = 2 * sizeof(cuda::std::uint32_t) +
                             static_cast<cuda::std::uint32_t>(state.num_containers) *
                               (2 * sizeof(cuda::std::uint16_t) + sizeof(cuda::std::uint32_t));
  reinterpret_cast<cuda::std::uint32_t*>(offsets)[index] = payload_begin + payload_offsets[index];
}

template <cuda::std::uint32_t BlockSize,
          class KeyIt,
          class ContainerStartIt,
          class PayloadOffsetIt,
          class ContainerIndexIt>
CUCO_KERNEL void write_roaring_containers(cuda::std::byte* bitmap,
                                          KeyIt keys,
                                          roaring_bitmap_build_state state,
                                          ContainerStartIt container_starts,
                                          PayloadOffsetIt payload_offsets,
                                          ContainerIndexIt array_containers,
                                          ContainerIndexIt bitset_containers)
{
  using metadata_type = roaring_bitmap_metadata<cuda::std::uint32_t>;

  constexpr cuda::std::uint32_t warp_size       = 32;
  constexpr cuda::std::uint32_t warps_per_block = BlockSize / warp_size;
  constexpr cuda::std::uint32_t bitset_words =
    metadata_type::bitset_container_bytes / sizeof(unsigned long long);
  constexpr cuda::std::uint32_t bitset_blocks_per_container = bitset_words / BlockSize;
  static_assert(BlockSize % warp_size == 0);
  static_assert(bitset_words % BlockSize == 0);

  auto const payload_begin = 2 * sizeof(cuda::std::uint32_t) +
                             static_cast<cuda::std::uint32_t>(state.num_containers) *
                               (2 * sizeof(cuda::std::uint16_t) + sizeof(cuda::std::uint32_t));
  auto const array_blocks = (state.num_array_containers + warps_per_block - 1) / warps_per_block;
  auto const block        = static_cast<cuda::std::uint32_t>(blockIdx.x);

  // Array containers use one warp each. Remaining blocks are divided into four 256-word pieces of
  // a bitset container.
  if (block < array_blocks) {
    auto const warp_index = block * warps_per_block + threadIdx.x / warp_size;
    if (warp_index >= state.num_array_containers) { return; }

    auto const lane            = static_cast<cuda::std::uint32_t>(threadIdx.x) % warp_size;
    auto const container_index = static_cast<cuda::std::int64_t>(array_containers[warp_index]);
    auto const begin           = container_starts[container_index];
    auto const end             = container_index + 1 < state.num_containers
                                   ? container_starts[container_index + 1]
                                   : state.num_keys;
    auto const cardinality     = static_cast<cuda::std::uint32_t>(end - begin);
    auto* const container      = bitmap + payload_begin + payload_offsets[container_index];

    for (auto index = lane; index < cardinality; index += warp_size) {
      auto const value = static_cast<cuda::std::uint16_t>(keys[begin + index]);
      misaligned_store(container + index * sizeof(cuda::std::uint16_t), value);
    }
  } else {
    auto const bitset_block = block - array_blocks;
    auto const bitset_index = bitset_block / bitset_blocks_per_container;
    if (bitset_index >= state.num_bitset_containers) { return; }

    auto const quadrant        = bitset_block % bitset_blocks_per_container;
    auto const word            = quadrant * BlockSize + threadIdx.x;
    auto const container_index = static_cast<cuda::std::int64_t>(bitset_containers[bitset_index]);
    auto const begin           = container_starts[container_index];
    auto const end             = container_index + 1 < state.num_containers
                                   ? container_starts[container_index + 1]
                                   : state.num_keys;
    auto* const container      = bitmap + payload_begin + payload_offsets[container_index];
    auto const word_begin      = word * 64;
    auto const word_end        = word_begin + 64;
    auto const first           = lower_bound_low_bits(keys, begin, end, word_begin);
    auto const last            = lower_bound_low_bits(keys, first, end, word_end);

    // One thread constructs one 64-bit word entirely in registers before issuing one final store.
    unsigned long long mask = 0;
    for (auto index = first; index < last; ++index) {
      auto const value = static_cast<cuda::std::uint16_t>(keys[index]);
      mask |= 1ULL << (value - word_begin);
    }
    misaligned_store(container + word * sizeof(unsigned long long), mask);
  }
}

}  // namespace cuco::experimental::detail
