/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/error.hpp>
#include <cuco/utility/traits.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/memory>
#include <cuda/std/span>

#include <nv/target>
#include <vector>

namespace cuco::experimental::detail {

template <class T>
__host__ __device__ __forceinline__ T aligned_load(cuda::std::byte const* ptr)
{
  return *reinterpret_cast<T const*>(cuda::std::assume_aligned<alignof(T)>(ptr));
}

template <class T>
__host__ __device__ __forceinline__ T misaligned_load(cuda::std::byte const* ptr)
{
  T value;
  cuda::std::memcpy(&value, ptr, sizeof(T));
  return value;
}

__host__ __device__ __forceinline__ bool check_bit(cuda::std::byte const* bitmap,
                                                   cuda::std::uint32_t index)
{
  // check if the bit at index is set
  return static_cast<cuda::std::uint8_t>(bitmap[index / 8]) &
         (cuda::std::uint8_t(1) << (index % 8));
}

inline void expect_valid_range(cuda::std::span<cuda::std::byte const> bitmap,
                               cuda::std::size_t offset,
                               cuda::std::size_t size)
{
  CUCO_EXPECTS(offset <= bitmap.size() and size <= bitmap.size() - offset,
               "Invalid bitmap format: serialized data is truncated");
}

template <class T>
T checked_load(cuda::std::span<cuda::std::byte const> bitmap, cuda::std::size_t offset)
{
  expect_valid_range(bitmap, offset, sizeof(T));
  return misaligned_load<T>(bitmap.data() + offset);
}

template <class T>
struct roaring_bitmap_metadata {
  static_assert(cuco::dependent_false<T>, "T must be either uint32_t or uint64_t");
};

/**
 * @brief Metadata structure for 32-bit roaring bitmap
 *
 * Contains metadata information for a 32-bit roaring bitmap including size, container information,
 * and validity status.
 */
template <>
struct roaring_bitmap_metadata<cuda::std::uint32_t> {
  /// Maximum number of elements in an array container before converting to bitmap
  static constexpr cuda::std::uint32_t max_array_container_card = 4096;
  /// Threshold for omitting container offsets in serialized format
  static constexpr cuda::std::int32_t no_offset_threshold = 4;
  /// Fixed size of a bitset container in bytes
  static constexpr cuda::std::uint32_t bitset_container_bytes = 8192;

  /// Total size of the bitmap in bytes
  cuda::std::size_t size_bytes = 0;
  /// Number of keys/elements in the bitmap
  cuda::std::size_t num_keys = 0;
  /// Bitmap indicating which containers are run containers
  cuda::std::uint32_t run_container_bitmap = 0;
  /// Offset to key cardinality data
  cuda::std::uint32_t key_cards = 0;
  /// Offset to container offset data (only valid when offsets_in_serialized_data is true)
  cuda::std::uint32_t container_offsets = 0;
  /// Computed container offsets (used when offsets are not in serialized data)
  cuda::std::uint32_t computed_offsets[no_offset_threshold] = {};
  /// Number of containers in the bitmap
  cuda::std::int32_t num_containers = 0;
  /// Whether the bitmap contains run containers
  bool has_run = false;
  /// Whether the metadata is valid
  bool valid = false;
  /// Whether container offsets are stored in the serialized data
  bool offsets_in_serialized_data = true;

  /**
   * @brief Constructs metadata from a bounded serialized bitmap
   *
   * @param bitmap Serialized bitmap bytes
   */
  __host__ roaring_bitmap_metadata(cuda::std::span<cuda::std::byte const> bitmap)
  {
    constexpr cuda::std::uint32_t serial_cookie_no_runcontainer = 12346;
    constexpr cuda::std::uint32_t serial_cookie                 = 12347;
    constexpr cuda::std::uint32_t max_containers                = 1 << 16;
    constexpr cuda::std::uint32_t cookie_mask                   = 0xFFFF;
    constexpr cuda::std::uint32_t cookie_shift                  = 16;

    cuda::std::size_t offset = 0;
    auto const cookie        = checked_load<cuda::std::uint32_t>(bitmap, offset);
    offset += sizeof(cuda::std::uint32_t);

    if ((cookie & cookie_mask) != serial_cookie && cookie != serial_cookie_no_runcontainer) {
      CUCO_FAIL("Invalid bitmap format: cookie type invalid or not supported");
    }

    cuda::std::uint32_t container_count;
    if ((cookie & cookie_mask) == serial_cookie) {
      container_count = (cookie >> cookie_shift) + 1;
    } else {
      container_count = checked_load<cuda::std::uint32_t>(bitmap, offset);
      offset += sizeof(cuda::std::uint32_t);
    }
    CUCO_EXPECTS(container_count <= max_containers,
                 "Invalid bitmap format: num_containers out of range");
    num_containers = static_cast<cuda::std::int32_t>(container_count);

    has_run = (cookie & cookie_mask) == serial_cookie;
    if (has_run) {
      auto const run_container_bitmap_size =
        static_cast<cuda::std::size_t>((num_containers + 7) / 8);
      expect_valid_range(bitmap, offset, run_container_bitmap_size);
      run_container_bitmap = static_cast<cuda::std::uint32_t>(offset);
      offset += run_container_bitmap_size;
    }

    key_cards = static_cast<cuda::std::uint32_t>(offset);
    auto const key_cards_size =
      static_cast<cuda::std::size_t>(num_containers) * 2 * sizeof(cuda::std::uint16_t);
    expect_valid_range(bitmap, offset, key_cards_size);
    offset += key_cards_size;

    if ((!has_run) || (num_containers >= no_offset_threshold)) {
      offsets_in_serialized_data = true;
      container_offsets          = static_cast<cuda::std::uint32_t>(offset);
      auto const container_offsets_size =
        static_cast<cuda::std::size_t>(num_containers) * sizeof(cuda::std::uint32_t);
      expect_valid_range(bitmap, offset, container_offsets_size);
      offset += container_offsets_size;
    } else {
      offsets_in_serialized_data = false;
      container_offsets          = 0;
    }

    if (num_containers == 0) {
      size_bytes = offset;
      valid      = true;
      return;
    }

    for (cuda::std::int32_t i = 0; i < num_containers; ++i) {
      auto const card_offset =
        static_cast<cuda::std::size_t>(key_cards) +
        static_cast<cuda::std::size_t>(i * 2 + 1) * sizeof(cuda::std::uint16_t);
      auto const card = 1u + checked_load<cuda::std::uint16_t>(bitmap, card_offset);
      CUCO_EXPECTS(card <= cuda::std::numeric_limits<cuda::std::size_t>::max() - num_keys,
                   "Invalid bitmap format: cardinality overflow");
      num_keys += card;
    }

    auto const container_size = [&](cuda::std::size_t container_offset, cuda::std::int32_t index) {
      bool const is_run_container =
        has_run and check_bit(bitmap.data() + run_container_bitmap, index);
      if (is_run_container) {
        auto const num_runs = checked_load<cuda::std::uint16_t>(bitmap, container_offset);
        return sizeof(cuda::std::uint16_t) +
               static_cast<cuda::std::size_t>(num_runs) * 2 * sizeof(cuda::std::uint16_t);
      }

      auto const card_offset =
        static_cast<cuda::std::size_t>(key_cards) +
        static_cast<cuda::std::size_t>(index * 2 + 1) * sizeof(cuda::std::uint16_t);
      auto const card = 1u + checked_load<cuda::std::uint16_t>(bitmap, card_offset);
      return card <= max_array_container_card
               ? static_cast<cuda::std::size_t>(card) * sizeof(cuda::std::uint16_t)
               : static_cast<cuda::std::size_t>(bitset_container_bytes);
    };

    if (offsets_in_serialized_data) {
      auto const containers_start = offset;
      auto previous_end           = containers_start;
      for (cuda::std::int32_t i = 0; i < num_containers; ++i) {
        auto const offset_offset = static_cast<cuda::std::size_t>(container_offsets) +
                                   static_cast<cuda::std::size_t>(i) * sizeof(cuda::std::uint32_t);
        auto const container_offset =
          static_cast<cuda::std::size_t>(checked_load<cuda::std::uint32_t>(bitmap, offset_offset));
        CUCO_EXPECTS(container_offset >= containers_start and container_offset >= previous_end,
                     "Invalid bitmap format: container offsets are invalid");
        auto const size = container_size(container_offset, i);
        expect_valid_range(bitmap, container_offset, size);
        previous_end = container_offset + size;
      }
      size_bytes = previous_end;
    } else {
      for (cuda::std::int32_t i = 0; i < num_containers; ++i) {
        CUCO_EXPECTS(offset <= cuda::std::numeric_limits<cuda::std::uint32_t>::max(),
                     "Invalid bitmap format: container offset is out of range");
        computed_offsets[i] = static_cast<cuda::std::uint32_t>(offset);
        auto const size     = container_size(offset, i);
        expect_valid_range(bitmap, offset, size);
        offset += size;
      }
      size_bytes = offset;
    }

    valid = true;
  }

  /**
   * @brief Constructs metadata from a serialized bitmap
   *
   * @param bitmap Pointer to the beginning of the serialized bitmap
   */
  __host__ __device__ roaring_bitmap_metadata(cuda::std::byte const* bitmap)
  {
    constexpr cuda::std::uint32_t serial_cookie_no_runcontainer = 12346;
    constexpr cuda::std::uint32_t serial_cookie                 = 12347;
    // constexpr cuda::std::uint32_t frozen_cookie                 = 13766; // not implemented
    constexpr cuda::std::int32_t max_containers = 1 << 16;
    constexpr cuda::std::uint32_t cookie_mask   = 0xFFFF;
    constexpr cuda::std::uint32_t cookie_shift  = 16;

    cuda::std::byte const* buf = bitmap;

    cuda::std::uint32_t cookie;
    cuda::std::memcpy(&cookie, buf, sizeof(cuda::std::uint32_t));
    buf += sizeof(cuda::std::uint32_t);
    if ((cookie & cookie_mask) != serial_cookie && cookie != serial_cookie_no_runcontainer) {
      valid = false;
      NV_IF_TARGET(
        NV_IS_HOST,
        CUCO_FAIL(
          "Invalid bitmap format: cookie type invalid or not supported");)  // TODO device error
                                                                            // handling
      return;
    }

    if ((cookie & cookie_mask) == serial_cookie)
      // upper 16 bits of cookie are the number of containers - 1
      num_containers = (cookie >> cookie_shift) + 1;
    else {
      // following 4 bytes are the number of containers
      cuda::std::memcpy(&num_containers, buf, sizeof(cuda::std::uint32_t));
      buf += sizeof(cuda::std::uint32_t);
    }
    if (num_containers < 0 or num_containers > max_containers) {
      valid = false;
      NV_IF_TARGET(
        NV_IS_HOST,
        CUCO_FAIL(
          "Invalid bitmap format: num_containers out of range");)  // TODO device error handling
      return;
    }

    has_run = (cookie & cookie_mask) == serial_cookie;
    if (has_run) {
      cuda::std::size_t s  = (num_containers + 7) / 8;  // ceil bytes to store run container bitmap
      run_container_bitmap = cuda::std::distance(bitmap, buf);
      buf += s;
    }

    key_cards = cuda::std::distance(bitmap, buf);
    // if the current address is aligned to 2 bytes, then all containers are aligned to at least 2
    // bytes
    bool const aligned_16 = (reinterpret_cast<cuda::std::uintptr_t>(bitmap + key_cards) %
                             sizeof(cuda::std::uint16_t)) == 0;
    buf += num_containers * 2 * sizeof(cuda::std::uint16_t);

    if ((!has_run) || (num_containers >= no_offset_threshold)) {
      // Container offsets are stored in the serialized data
      offsets_in_serialized_data = true;
      container_offsets          = cuda::std::distance(bitmap, buf);
      buf += num_containers * sizeof(cuda::std::uint32_t);
    } else {
      // Container offsets are NOT stored in the serialized data
      // We need to compute them by walking through the containers
      offsets_in_serialized_data = false;
      container_offsets          = 0;

      cuda::std::byte const* container_ptr = buf;
      for (cuda::std::int32_t i = 0; i < num_containers; ++i) {
        // Store the computed offset for this container
        computed_offsets[i] =
          static_cast<cuda::std::uint32_t>(cuda::std::distance(bitmap, container_ptr));

        // Get cardinality for this container
        cuda::std::byte const* card_ptr =
          bitmap + key_cards + (i * 2 + 1) * sizeof(cuda::std::uint16_t);
        cuda::std::uint32_t card_i = 1u + misaligned_load<cuda::std::uint16_t>(card_ptr);

        // Check if this is a run container
        bool is_run_container = check_bit(bitmap + run_container_bitmap, i);

        // Compute container size and advance pointer
        if (is_run_container) {
          // Run container: first uint16_t is num_runs, followed by num_runs (start, length) pairs
          cuda::std::uint16_t num_runs = misaligned_load<cuda::std::uint16_t>(container_ptr);
          container_ptr += sizeof(cuda::std::uint16_t) + num_runs * 2 * sizeof(cuda::std::uint16_t);
        } else if (card_i <= max_array_container_card) {
          // Array container
          container_ptr += card_i * sizeof(cuda::std::uint16_t);
        } else {
          // Bitset container (fixed size)
          container_ptr += bitset_container_bytes;
        }
      }
      // buf now points past all containers
      buf = container_ptr;
    }

    cuda::std::uint32_t card = 0;
    for (cuda::std::int32_t i = 0; i < num_containers; i++) {
      cuda::std::byte const* card_ptr =
        bitmap + key_cards + (i * 2 + 1) * sizeof(cuda::std::uint16_t);
      if (aligned_16) {
        card = 1u + aligned_load<cuda::std::uint16_t>(card_ptr);
      } else {
        card = 1u + misaligned_load<cuda::std::uint16_t>(card_ptr);
      }
      num_keys += card;
    }

    // find end of roaring bitmap (re-use card from last container)
    cuda::std::byte const* end;
    if (offsets_in_serialized_data) {
      end =
        bitmap + misaligned_load<cuda::std::uint32_t>(
                   bitmap + container_offsets + (num_containers - 1) * sizeof(cuda::std::uint32_t));
    } else {
      end = bitmap + computed_offsets[num_containers - 1];
    }

    if (has_run and check_bit(bitmap + run_container_bitmap, num_containers - 1)) {
      cuda::std::uint16_t const num_runs = misaligned_load<cuda::std::uint16_t>(end);
      end += sizeof(cuda::std::uint16_t) + num_runs * 2 * sizeof(cuda::std::uint16_t);
    } else {
      if (card <= max_array_container_card) {
        end += card * sizeof(cuda::std::uint16_t);
      } else {
        end += bitset_container_bytes;  // fixed size bitset container
      }
    }

    size_bytes = static_cast<cuda::std::size_t>(cuda::std::distance(bitmap, end));
    valid      = true;
  }
};

/**
 * @brief Metadata structure for 64-bit roaring bitmap
 *
 * Contains metadata information for a 64-bit roaring bitmap including bucket information,
 * size, and validity status.
 */
template <>
struct roaring_bitmap_metadata<cuda::std::uint64_t> {
  /// Number of buckets in the 64-bit bitmap
  cuda::std::size_t num_buckets = 0;
  /// Total size of the bitmap in bytes
  cuda::std::size_t size_bytes = 0;
  /// Number of keys/elements in the bitmap
  cuda::std::size_t num_keys = 0;
  /// Whether the metadata is valid
  bool valid = false;

  /**
   * @brief Metadata for individual buckets in a 64-bit roaring bitmap
   *
   * Each bucket contains a 32-bit roaring bitmap with its own metadata.
   */
  struct bucket_metadata {
    /// Byte offset of this bucket in the serialized data
    cuda::std::size_t byte_offset;
    /// Key associated with this bucket (upper 32 bits)
    cuda::std::uint32_t key;
    /// Metadata for the 32-bit roaring bitmap in this bucket
    roaring_bitmap_metadata<cuda::std::uint32_t> metadata;

    /**
     * @brief Constructs bucket metadata
     *
     * @param offset Byte offset of the bucket
     * @param k Key associated with the bucket
     * @param meta Metadata for the bucket's roaring bitmap
     */
    bucket_metadata(cuda::std::size_t offset,
                    cuda::std::uint32_t k,
                    roaring_bitmap_metadata<cuda::std::uint32_t> const& meta)
      : byte_offset{offset}, key{k}, metadata{meta}
    {
    }
  };

  /**
   * @brief Constructs metadata from a bounded serialized 64-bit bitmap
   *
   * @param bitmap Serialized bitmap bytes
   * @param bucket_metadata Vector to store metadata for each bucket
   */
  __host__ roaring_bitmap_metadata(cuda::std::span<cuda::std::byte const> bitmap,
                                   std::vector<bucket_metadata>& bucket_metadata)
  {
    cuda::std::size_t byte_offset     = 0;
    auto const serialized_num_buckets = checked_load<cuda::std::uint64_t>(bitmap, byte_offset);
    byte_offset += sizeof(cuda::std::uint64_t);

    CUCO_EXPECTS(serialized_num_buckets <= cuda::std::numeric_limits<cuda::std::size_t>::max(),
                 "Invalid bitmap format: num_buckets out of range");
    num_buckets = static_cast<cuda::std::size_t>(serialized_num_buckets);

    constexpr cuda::std::size_t minimum_bucket_prefix_size =
      sizeof(cuda::std::uint32_t) + sizeof(cuda::std::uint32_t);
    CUCO_EXPECTS(num_buckets <= (bitmap.size() - byte_offset) / minimum_bucket_prefix_size,
                 "Invalid bitmap format: num_buckets exceeds the serialized data size");

    bucket_metadata.clear();

    for (cuda::std::size_t i = 0; i < num_buckets; ++i) {
      auto const bucket_key = checked_load<cuda::std::uint32_t>(bitmap, byte_offset);
      byte_offset += sizeof(cuda::std::uint32_t);

      roaring_bitmap_metadata<cuda::std::uint32_t> bucket_meta{bitmap.subspan(byte_offset)};
      CUCO_EXPECTS(bucket_meta.valid and bucket_meta.size_bytes > 0,
                   "Invalid bitmap format: bucket metadata is invalid");
      expect_valid_range(bitmap, byte_offset, bucket_meta.size_bytes);
      CUCO_EXPECTS(
        bucket_meta.num_keys <= cuda::std::numeric_limits<cuda::std::size_t>::max() - num_keys,
        "Invalid bitmap format: cardinality overflow");

      bucket_metadata.emplace_back(byte_offset, bucket_key, bucket_meta);
      num_keys += bucket_meta.num_keys;
      byte_offset += bucket_meta.size_bytes;
    }
    size_bytes = byte_offset;
    valid      = true;
  }

  /**
   * @brief Constructs metadata from a serialized 64-bit bitmap with bucket metadata
   *
   * @param bitmap Pointer to the beginning of the serialized bitmap
   * @param bucket_metadata Vector to store metadata for each bucket
   */
  __host__ roaring_bitmap_metadata(cuda::std::byte const* bitmap,
                                   std::vector<bucket_metadata>& bucket_metadata)
  {
    cuda::std::size_t byte_offset     = 0;
    cuda::std::byte const* bitmap_ptr = bitmap;
    cuda::std::memcpy(&num_buckets, bitmap_ptr, sizeof(cuda::std::uint64_t));
    byte_offset += sizeof(cuda::std::uint64_t);  // skip num_buckets

    bucket_metadata.clear();
    bucket_metadata.reserve(num_buckets);

    for (cuda::std::size_t i = 0; i < num_buckets; ++i) {
      cuda::std::uint32_t bucket_key;
      cuda::std::memcpy(&bucket_key, bitmap_ptr + byte_offset, sizeof(cuda::std::uint32_t));
      byte_offset += sizeof(cuda::std::uint32_t);  // skip bucket key
      roaring_bitmap_metadata<cuda::std::uint32_t> bucket_meta{bitmap_ptr + byte_offset};
      if (!bucket_meta.valid) {
        valid = false;
        return;
      }
      bucket_metadata.emplace_back(byte_offset, bucket_key, bucket_meta);
      num_keys += bucket_meta.num_keys;
      byte_offset += bucket_meta.size_bytes;  // skip bucket
    }
    size_bytes = byte_offset;
    valid      = true;
  }

  /**
   * @brief Constructs metadata from a serialized 64-bit bitmap
   *
   * @param bitmap Pointer to the beginning of the serialized bitmap
   */
  __host__ __device__ roaring_bitmap_metadata(cuda::std::byte const* bitmap)
  {
    cuda::std::size_t byte_offset     = 0;
    cuda::std::byte const* bitmap_ptr = bitmap;
    cuda::std::memcpy(&num_buckets, bitmap_ptr, sizeof(cuda::std::uint64_t));
    byte_offset += sizeof(cuda::std::uint64_t);  // skip num_buckets

    for (cuda::std::size_t i = 0; i < num_buckets; ++i) {
      byte_offset += sizeof(cuda::std::uint32_t);  // skip bucket key
      roaring_bitmap_metadata<cuda::std::uint32_t> bucket_meta{bitmap_ptr + byte_offset};
      if (!bucket_meta.valid) {
        valid = false;
        return;
      }
      num_keys += bucket_meta.num_keys;
      byte_offset += bucket_meta.size_bytes;  // skip bucket
    }
    size_bytes = byte_offset;
    valid      = true;
  }
};
}  // namespace cuco::experimental::detail
