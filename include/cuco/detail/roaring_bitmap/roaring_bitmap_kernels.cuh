/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/roaring_bitmap/util.cuh>
#include <cuco/detail/utility/cuda.cuh>

#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/iterator>

namespace cuco::experimental::detail {

CUCO_SUPPRESS_KERNEL_WARNINGS

/**
 * @brief Device-computed scalar state shared by the Roaring construction kernels.
 */
struct roaring_bitmap_build_state {
  cuco::detail::index_type num_indices;       ///< Number of sorted unique indices
  cuda::std::int64_t num_containers;          ///< Number of high-16-bit containers
  cuda::std::uint32_t size_bytes;             ///< Size of the serialized bitmap
  cuda::std::uint32_t num_array_containers;   ///< Number of array containers
  cuda::std::uint32_t num_bitset_containers;  ///< Number of bitset containers
};

/**
 * @brief Predicate selecting the first index in each high-16-bit container.
 *
 * @tparam IndexIt Random access iterator over sorted unique indices
 */
template <class IndexIt>
struct is_container_start {
  IndexIt indices;                          ///< Sorted unique indices
  roaring_bitmap_build_state const* state;  ///< Device build state

  /**
   * @brief Tests whether an index begins a container.
   *
   * @param index Index in the normalized input range
   * @return `true` if `index` begins a container
   */
  __device__ bool operator()(cuco::detail::index_type index) const noexcept
  {
    auto const num_indices = state->num_indices;
    if (index >= num_indices) { return false; }
    if (index == 0) { return true; }
    return roaring_bitmap_metadata<cuda::std::uint32_t>::container_key(indices[index]) !=
           roaring_bitmap_metadata<cuda::std::uint32_t>::container_key(indices[index - 1]);
  }
};

template <class IndexIt>
is_container_start(IndexIt, roaring_bitmap_build_state const*) -> is_container_start<IndexIt>;

/**
 * @brief Half-open range of normalized indices belonging to one container.
 */
struct roaring_bitmap_container_bounds {
  cuco::detail::index_type begin;  ///< First index in the container
  cuco::detail::index_type end;    ///< One-past-the-end index in the container
};

/**
 * @brief Returns the normalized input range belonging to a container.
 *
 * @param container Container index
 * @param container_starts Starting normalized input index of each container
 * @param state Device build state
 * @return Half-open normalized input range
 */
[[nodiscard]] __device__ inline roaring_bitmap_container_bounds container_bounds(
  cuco::detail::index_type container,
  cuco::detail::index_type const* container_starts,
  roaring_bitmap_build_state const* state) noexcept
{
  auto const begin = container_starts[container];
  auto const end =
    container + 1 < state->num_containers ? container_starts[container + 1] : state->num_indices;
  return {begin, end};
}

/**
 * @brief Computes the encoded payload size of a container.
 */
struct container_payload_size {
  cuco::detail::index_type const* container_starts;  ///< Starting index of each container
  roaring_bitmap_build_state const* state;           ///< Device build state

  /**
   * @brief Returns the encoded payload size for one container slot.
   *
   * @param index Container slot index
   * @return Payload size in bytes, or zero for an unused slot
   */
  __device__ cuda::std::uint32_t operator()(cuco::detail::index_type index) const noexcept
  {
    using metadata_type = roaring_bitmap_metadata<cuda::std::uint32_t>;

    auto const num_containers = state->num_containers;
    if (index >= num_containers) { return 0; }

    auto const bounds      = container_bounds(index, container_starts, state);
    auto const cardinality = static_cast<cuda::std::uint32_t>(bounds.end - bounds.begin);
    return metadata_type::container_payload_bytes(cardinality);
  }
};

/**
 * @brief Partitions container indexes into array and bitset work queues.
 *
 * The final container also computes the exact serialized bitmap size.
 *
 * @param container_indexes Shared array/bitset work queue
 * @param num_container_slots Number of allocated container slots
 * @param container_starts Starting normalized input index of each container
 * @param payload_offsets Exclusive payload offsets for every container
 * @param state Device build state
 */
static CUCO_KERNEL void collect_container_indexes(cuda::std::uint32_t* container_indexes,
                                                  cuco::detail::index_type num_container_slots,
                                                  cuco::detail::index_type const* container_starts,
                                                  cuda::std::uint32_t const* payload_offsets,
                                                  roaring_bitmap_build_state* state)
{
  using metadata_type = roaring_bitmap_metadata<cuda::std::uint32_t>;

  auto const index = cuco::detail::global_thread_id();
  if (index >= state->num_containers || index >= num_container_slots) { return; }

  auto const bounds      = container_bounds(index, container_starts, state);
  auto const cardinality = static_cast<cuda::std::uint32_t>(bounds.end - bounds.begin);
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

  if (index == state->num_containers - 1) {
    state->size_bytes =
      metadata_type::no_run_header_bytes(static_cast<cuda::std::uint32_t>(state->num_containers)) +
      payload_offsets[index] + metadata_type::container_payload_bytes(cardinality);
  }
}

/**
 * @brief Finds the first normalized index whose low 16 bits are not less than `value`.
 *
 * @tparam IndexIt Random access iterator over sorted unique indices
 *
 * @param indices Sorted unique indices
 * @param first Beginning of the search range
 * @param last End of the search range
 * @param value Low-16-bit value to locate
 * @return Position of the first matching or greater value
 */
template <class IndexIt>
__device__ cuco::detail::index_type lower_bound_low_bits(IndexIt indices,
                                                         cuco::detail::index_type first,
                                                         cuco::detail::index_type last,
                                                         cuda::std::uint32_t value)
{
  auto const begin = indices + first;
  auto const found =
    cuda::std::lower_bound(begin, indices + last, value, [] __device__(auto index, auto lower) {
      return static_cast<cuda::std::uint16_t>(index) < lower;
    });
  return first + cuda::std::distance(begin, found);
}

/**
 * @brief Writes the no-run Roaring header, descriptors, and container offsets.
 *
 * @tparam IndexIt Random access iterator over sorted unique indices
 *
 * @param bitmap Output serialized bitmap
 * @param indices Sorted unique indices
 * @param state Completed build state
 * @param container_starts Starting normalized input index of each container
 * @param payload_offsets Exclusive payload offsets for every container
 */
template <class IndexIt>
CUCO_KERNEL void write_roaring_bitmap_header(cuda::std::byte* bitmap,
                                             IndexIt indices,
                                             roaring_bitmap_build_state state,
                                             cuco::detail::index_type const* container_starts,
                                             cuda::std::uint32_t const* payload_offsets)
{
  using metadata_type = roaring_bitmap_metadata<cuda::std::uint32_t>;

  auto const index = cuco::detail::global_thread_id();

  if (index == 0) {
    misaligned_store(bitmap, metadata_type::serial_cookie_no_runcontainer);
    misaligned_store(bitmap + sizeof(cuda::std::uint32_t),
                     static_cast<cuda::std::uint32_t>(state.num_containers));
  }

  if (index >= state.num_containers) { return; }

  auto const bounds         = container_bounds(index, container_starts, &state);
  auto const cardinality    = static_cast<cuda::std::uint32_t>(bounds.end - bounds.begin);
  auto const container_key  = metadata_type::container_key(indices[bounds.begin]);
  auto const card_minus_one = static_cast<cuda::std::uint16_t>(cardinality - 1);

  auto* const key_cards = bitmap + metadata_type::no_run_key_cards_offset;
  misaligned_store(key_cards + index * 2 * sizeof(cuda::std::uint16_t), container_key);
  misaligned_store(key_cards + (index * 2 + 1) * sizeof(cuda::std::uint16_t), card_minus_one);

  auto* const offsets = bitmap + metadata_type::no_run_container_offsets_offset(
                                   static_cast<cuda::std::uint32_t>(state.num_containers));
  auto const offset = static_cast<cuda::std::uint32_t>(
    metadata_type::no_run_header_bytes(static_cast<cuda::std::uint32_t>(state.num_containers)) +
    payload_offsets[index]);
  misaligned_store(offsets + index * sizeof(cuda::std::uint32_t), offset);
}

/**
 * @brief Writes array and bitset container payloads.
 *
 * @tparam BlockSize Number of threads per block
 * @tparam IndexIt Random access iterator over sorted unique indices
 *
 * @param bitmap Output serialized bitmap
 * @param indices Sorted unique indices
 * @param state Completed build state
 * @param container_starts Starting normalized input index of each container
 * @param payload_offsets Exclusive payload offsets for every container
 * @param array_containers Array-container work queue
 * @param bitset_containers Bitset-container work queue
 */
template <cuda::std::uint32_t BlockSize, class IndexIt>
CUCO_KERNEL __launch_bounds__(BlockSize) void write_roaring_containers(
  cuda::std::byte* bitmap,
  IndexIt indices,
  roaring_bitmap_build_state state,
  cuco::detail::index_type const* container_starts,
  cuda::std::uint32_t const* payload_offsets,
  cuda::std::uint32_t const* array_containers,
  cuda::std::uint32_t const* bitset_containers)
{
  using metadata_type    = roaring_bitmap_metadata<cuda::std::uint32_t>;
  using bitset_word_type = typename metadata_type::bitset_word_type;

  constexpr cuda::std::uint32_t warps_per_block             = BlockSize / cuco::detail::warp_size();
  constexpr cuda::std::uint32_t bitset_words                = metadata_type::bitset_container_words;
  constexpr cuda::std::uint32_t bitset_blocks_per_container = bitset_words / BlockSize;
  static_assert(BlockSize % cuco::detail::warp_size() == 0);
  static_assert(bitset_words % BlockSize == 0);

  auto const payload_begin =
    metadata_type::no_run_header_bytes(static_cast<cuda::std::uint32_t>(state.num_containers));
  auto const array_blocks = (state.num_array_containers + warps_per_block - 1) / warps_per_block;
  auto const block        = static_cast<cuda::std::uint32_t>(blockIdx.x);

  // Array containers use one warp each. Remaining blocks are divided into four 256-word pieces of
  // a bitset container.
  if (block < array_blocks) {
    auto const warp_index = block * warps_per_block + threadIdx.x / cuco::detail::warp_size();
    if (warp_index >= state.num_array_containers) { return; }

    auto const lane = static_cast<cuda::std::uint32_t>(threadIdx.x) % cuco::detail::warp_size();
    auto const container_index =
      static_cast<cuco::detail::index_type>(array_containers[warp_index]);
    auto const bounds      = container_bounds(container_index, container_starts, &state);
    auto const cardinality = static_cast<cuda::std::uint32_t>(bounds.end - bounds.begin);
    auto* const container  = bitmap + payload_begin + payload_offsets[container_index];

    for (auto index = lane; index < cardinality; index += cuco::detail::warp_size()) {
      auto const value = static_cast<cuda::std::uint16_t>(indices[bounds.begin + index]);
      misaligned_store(container + index * sizeof(cuda::std::uint16_t), value);
    }
  } else {
    auto const bitset_block = block - array_blocks;
    auto const bitset_index = bitset_block / bitset_blocks_per_container;
    if (bitset_index >= state.num_bitset_containers) { return; }

    auto const quadrant = bitset_block % bitset_blocks_per_container;
    auto const word     = quadrant * BlockSize + threadIdx.x;
    auto const container_index =
      static_cast<cuco::detail::index_type>(bitset_containers[bitset_index]);
    auto const bounds     = container_bounds(container_index, container_starts, &state);
    auto* const container = bitmap + payload_begin + payload_offsets[container_index];
    auto const word_begin = word * 64;
    auto const word_end   = word_begin + 64;
    auto const first      = lower_bound_low_bits(indices, bounds.begin, bounds.end, word_begin);
    auto const last       = lower_bound_low_bits(indices, first, bounds.end, word_end);

    // One thread constructs one 64-bit word entirely in registers before issuing one final store.
    bitset_word_type mask = 0;
    for (auto index = first; index < last; ++index) {
      auto const value = static_cast<cuda::std::uint16_t>(indices[index]);
      mask |= bitset_word_type{1} << (value - word_begin);
    }
    misaligned_store(container + word * sizeof(bitset_word_type), mask);
  }
}

}  // namespace cuco::experimental::detail
