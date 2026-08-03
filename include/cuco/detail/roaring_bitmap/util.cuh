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

/**
 * @brief Non-owning view of serialized bitmap data with optional bounds information
 *
 * Pointer-backed views preserve the unchecked behavior of the legacy API, while span-backed views
 * validate accesses against the serialized data size.
 */
class serialized_bitmap_view {
 public:
  /**
   * @brief Constructs an unbounded view from a pointer
   *
   * @param data Pointer to the beginning of the serialized bitmap
   */
  __host__ __device__ explicit serialized_bitmap_view(cuda::std::byte const* data)
    : data_{data}, size_{0}, bounded_{false}
  {
  }

  /**
   * @brief Constructs a bounded view from a span
   *
   * @param bitmap Serialized bitmap bytes
   */
  __host__ __device__ explicit serialized_bitmap_view(cuda::std::span<cuda::std::byte const> bitmap)
    : data_{bitmap.data()}, size_{bitmap.size()}, bounded_{true}
  {
  }

  /**
   * @brief Returns a pointer to the serialized bitmap data
   *
   * @return Pointer to the beginning of the serialized bitmap
   */
  [[nodiscard]] __host__ __device__ cuda::std::byte const* data() const noexcept { return data_; }

  /**
   * @brief Returns the size of a bounded view
   *
   * @return Serialized bitmap size in bytes, or zero for an unbounded view
   */
  [[nodiscard]] __host__ __device__ cuda::std::size_t size() const noexcept { return size_; }

  /**
   * @brief Indicates whether the view has bounds information
   *
   * @return true if the view was constructed from a span, otherwise false
   */
  [[nodiscard]] __host__ __device__ bool is_bounded() const noexcept { return bounded_; }

  /**
   * @brief Checks whether a byte range is contained in the view
   *
   * Unbounded views contain every range.
   *
   * @param offset Start of the range in bytes
   * @param size Size of the range in bytes
   * @return true if the range is contained in the view, otherwise false
   */
  [[nodiscard]] __host__ __device__ bool contains(cuda::std::size_t offset,
                                                  cuda::std::size_t size) const noexcept
  {
    return not bounded_ or (offset <= size_ and size <= size_ - offset);
  }

  /**
   * @brief Returns a view beginning at the specified byte offset
   *
   * @param offset Offset from the beginning of the serialized bitmap
   * @return View of the remaining serialized bitmap data
   */
  [[nodiscard]] __host__ __device__ serialized_bitmap_view subview(cuda::std::size_t offset) const
  {
    if (not bounded_) { return serialized_bitmap_view{data_ + offset}; }
    if (offset > size_) {
      return serialized_bitmap_view{cuda::std::span<cuda::std::byte const>{data_, 0}};
    }
    return serialized_bitmap_view{
      cuda::std::span<cuda::std::byte const>{data_ + offset, size_ - offset}};
  }

  /**
   * @brief Loads a value if its byte range is contained in the view
   *
   * @tparam T Type of value to load
   * @param offset Offset of the value in bytes
   * @param value Reference that receives the loaded value
   * @return true if the value was loaded, otherwise false
   */
  template <class T>
  __host__ __device__ bool try_load(cuda::std::size_t offset, T& value) const
  {
    if (not contains(offset, sizeof(T))) { return false; }
    value = misaligned_load<T>(data_ + offset);
    return true;
  }

 private:
  cuda::std::byte const* data_;
  cuda::std::size_t size_;
  bool bounded_;
};

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
    : roaring_bitmap_metadata{serialized_bitmap_view{bitmap}}
  {
  }

  /**
   * @brief Constructs metadata from a serialized bitmap
   *
   * @param bitmap Pointer to the beginning of the serialized bitmap
   */
  __host__ __device__ roaring_bitmap_metadata(cuda::std::byte const* bitmap)
    : roaring_bitmap_metadata{serialized_bitmap_view{bitmap}}
  {
  }

  /**
   * @brief Constructs metadata from an internal serialized bitmap view
   *
   * @param bitmap Serialized bitmap view
   */
  __host__ __device__ explicit roaring_bitmap_metadata(serialized_bitmap_view bitmap)
  {
    parse(bitmap);
  }

 private:
  template <class T>
  __host__ __device__ bool load(serialized_bitmap_view bitmap, cuda::std::size_t offset, T& value)
  {
    if (bitmap.try_load(offset, value)) { return true; }
    valid = false;
    NV_IF_TARGET(
      NV_IS_HOST,
      CUCO_FAIL("Invalid bitmap format: serialized data is truncated");)  // TODO device error
                                                                          // handling
    return false;
  }

  __host__ __device__ bool expect_range(serialized_bitmap_view bitmap,
                                        cuda::std::size_t offset,
                                        cuda::std::size_t size)
  {
    if (bitmap.contains(offset, size)) { return true; }
    valid = false;
    NV_IF_TARGET(
      NV_IS_HOST,
      CUCO_FAIL("Invalid bitmap format: serialized data is truncated");)  // TODO device error
                                                                          // handling
    return false;
  }

  __host__ __device__ bool get_container_size(serialized_bitmap_view bitmap,
                                              cuda::std::size_t container_offset,
                                              cuda::std::int32_t index,
                                              cuda::std::size_t& size)
  {
    bool const is_run_container =
      has_run and check_bit(bitmap.data() + run_container_bitmap, index);
    if (is_run_container) {
      cuda::std::uint16_t num_runs;
      if (not load(bitmap, container_offset, num_runs)) { return false; }
      size = sizeof(cuda::std::uint16_t) +
             static_cast<cuda::std::size_t>(num_runs) * 2 * sizeof(cuda::std::uint16_t);
      return true;
    }

    auto const card_offset =
      static_cast<cuda::std::size_t>(key_cards) +
      static_cast<cuda::std::size_t>(index * 2 + 1) * sizeof(cuda::std::uint16_t);
    cuda::std::uint16_t stored_card;
    if (not load(bitmap, card_offset, stored_card)) { return false; }
    auto const card = 1u + stored_card;
    size            = card <= max_array_container_card
                        ? static_cast<cuda::std::size_t>(card) * sizeof(cuda::std::uint16_t)
                        : static_cast<cuda::std::size_t>(bitset_container_bytes);
    return true;
  }

  __host__ __device__ void parse(serialized_bitmap_view bitmap)
  {
    constexpr cuda::std::uint32_t serial_cookie_no_runcontainer = 12346;
    constexpr cuda::std::uint32_t serial_cookie                 = 12347;
    constexpr cuda::std::uint32_t max_containers                = 1 << 16;
    constexpr cuda::std::uint32_t cookie_mask                   = 0xFFFF;
    constexpr cuda::std::uint32_t cookie_shift                  = 16;

    cuda::std::size_t offset = 0;
    cuda::std::uint32_t cookie;
    if (not load(bitmap, offset, cookie)) { return; }
    offset += sizeof(cuda::std::uint32_t);

    if ((cookie & cookie_mask) != serial_cookie && cookie != serial_cookie_no_runcontainer) {
      valid = false;
      NV_IF_TARGET(
        NV_IS_HOST,
        CUCO_FAIL(
          "Invalid bitmap format: cookie type invalid or not supported");)  // TODO device error
                                                                            // handling
      return;
    }

    cuda::std::uint32_t container_count;
    if ((cookie & cookie_mask) == serial_cookie) {
      container_count = (cookie >> cookie_shift) + 1;
    } else {
      if (not load(bitmap, offset, container_count)) { return; }
      offset += sizeof(cuda::std::uint32_t);
    }
    if (container_count > max_containers) {
      valid = false;
      NV_IF_TARGET(
        NV_IS_HOST,
        CUCO_FAIL(
          "Invalid bitmap format: num_containers out of range");)  // TODO device error handling
      return;
    }
    num_containers = static_cast<cuda::std::int32_t>(container_count);

    has_run = (cookie & cookie_mask) == serial_cookie;
    if (has_run) {
      auto const run_container_bitmap_size =
        static_cast<cuda::std::size_t>((num_containers + 7) / 8);
      if (not expect_range(bitmap, offset, run_container_bitmap_size)) { return; }
      run_container_bitmap = static_cast<cuda::std::uint32_t>(offset);
      offset += run_container_bitmap_size;
    }

    key_cards = static_cast<cuda::std::uint32_t>(offset);
    auto const key_cards_size =
      static_cast<cuda::std::size_t>(num_containers) * 2 * sizeof(cuda::std::uint16_t);
    if (not expect_range(bitmap, offset, key_cards_size)) { return; }
    offset += key_cards_size;

    if ((!has_run) || (num_containers >= no_offset_threshold)) {
      offsets_in_serialized_data = true;
      container_offsets          = static_cast<cuda::std::uint32_t>(offset);
      auto const container_offsets_size =
        static_cast<cuda::std::size_t>(num_containers) * sizeof(cuda::std::uint32_t);
      if (not expect_range(bitmap, offset, container_offsets_size)) { return; }
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
      cuda::std::uint16_t stored_card;
      if (not load(bitmap, card_offset, stored_card)) { return; }
      auto const card = 1u + stored_card;
      if (card > cuda::std::numeric_limits<cuda::std::size_t>::max() - num_keys) {
        valid = false;
        NV_IF_TARGET(
          NV_IS_HOST,
          CUCO_FAIL("Invalid bitmap format: cardinality overflow");)  // TODO device error handling
        return;
      }
      num_keys += card;
    }

    if (offsets_in_serialized_data) {
      auto const containers_start = offset;
      auto previous_end           = containers_start;
      for (cuda::std::int32_t i = 0; i < num_containers; ++i) {
        auto const offset_offset = static_cast<cuda::std::size_t>(container_offsets) +
                                   static_cast<cuda::std::size_t>(i) * sizeof(cuda::std::uint32_t);
        cuda::std::uint32_t stored_offset;
        if (not load(bitmap, offset_offset, stored_offset)) { return; }
        auto const container_offset = static_cast<cuda::std::size_t>(stored_offset);
        if (container_offset < containers_start or container_offset < previous_end) {
          valid = false;
          NV_IF_TARGET(
            NV_IS_HOST,
            CUCO_FAIL("Invalid bitmap format: container offsets are invalid");)  // TODO device
                                                                                 // error handling
          return;
        }
        cuda::std::size_t size;
        if (not get_container_size(bitmap, container_offset, i, size)) { return; }
        if (not expect_range(bitmap, container_offset, size)) { return; }
        previous_end = container_offset + size;
      }
      size_bytes = previous_end;
    } else {
      for (cuda::std::int32_t i = 0; i < num_containers; ++i) {
        if (offset > cuda::std::numeric_limits<cuda::std::uint32_t>::max()) {
          valid = false;
          NV_IF_TARGET(
            NV_IS_HOST,
            CUCO_FAIL(
              "Invalid bitmap format: container offset is out of range");)  // TODO device error
                                                                            // handling
          return;
        }
        computed_offsets[i] = static_cast<cuda::std::uint32_t>(offset);
        cuda::std::size_t size;
        if (not get_container_size(bitmap, offset, i, size)) { return; }
        if (not expect_range(bitmap, offset, size)) { return; }
        offset += size;
      }
      size_bytes = offset;
    }

    valid = true;
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
    parse(serialized_bitmap_view{bitmap}, bucket_metadata);
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
    parse(serialized_bitmap_view{bitmap}, bucket_metadata);
  }

  /**
   * @brief Constructs metadata from a serialized 64-bit bitmap
   *
   * @param bitmap Pointer to the beginning of the serialized bitmap
   */
  __host__ __device__ roaring_bitmap_metadata(cuda::std::byte const* bitmap)
  {
    parse(serialized_bitmap_view{bitmap});
  }

 private:
  template <class T>
  __host__ __device__ bool load(serialized_bitmap_view bitmap, cuda::std::size_t offset, T& value)
  {
    if (bitmap.try_load(offset, value)) { return true; }
    valid = false;
    NV_IF_TARGET(
      NV_IS_HOST,
      CUCO_FAIL("Invalid bitmap format: serialized data is truncated");)  // TODO device error
                                                                          // handling
    return false;
  }

  __host__ void parse(serialized_bitmap_view bitmap, std::vector<bucket_metadata>& bucket_metadata)
  {
    cuda::std::size_t byte_offset = 0;
    cuda::std::uint64_t serialized_num_buckets;
    if (not load(bitmap, byte_offset, serialized_num_buckets)) { return; }
    byte_offset += sizeof(cuda::std::uint64_t);

    CUCO_EXPECTS(serialized_num_buckets <= cuda::std::numeric_limits<cuda::std::size_t>::max(),
                 "Invalid bitmap format: num_buckets out of range");
    num_buckets = static_cast<cuda::std::size_t>(serialized_num_buckets);

    constexpr cuda::std::size_t minimum_bucket_prefix_size =
      sizeof(cuda::std::uint32_t) + sizeof(cuda::std::uint32_t);
    if (bitmap.is_bounded()) {
      CUCO_EXPECTS(num_buckets <= (bitmap.size() - byte_offset) / minimum_bucket_prefix_size,
                   "Invalid bitmap format: num_buckets exceeds the serialized data size");
    }

    bucket_metadata.clear();
    if (not bitmap.is_bounded()) { bucket_metadata.reserve(num_buckets); }

    for (cuda::std::size_t i = 0; i < num_buckets; ++i) {
      cuda::std::uint32_t bucket_key;
      if (not load(bitmap, byte_offset, bucket_key)) { return; }
      byte_offset += sizeof(cuda::std::uint32_t);

      roaring_bitmap_metadata<cuda::std::uint32_t> bucket_meta{bitmap.subview(byte_offset)};
      CUCO_EXPECTS(bucket_meta.valid and bucket_meta.size_bytes > 0,
                   "Invalid bitmap format: bucket metadata is invalid");
      CUCO_EXPECTS(not bitmap.is_bounded() or bitmap.contains(byte_offset, bucket_meta.size_bytes),
                   "Invalid bitmap format: serialized data is truncated");
      CUCO_EXPECTS(
        bucket_meta.num_keys <= cuda::std::numeric_limits<cuda::std::size_t>::max() - num_keys,
        "Invalid bitmap format: cardinality overflow");
      CUCO_EXPECTS(
        bucket_meta.size_bytes <= cuda::std::numeric_limits<cuda::std::size_t>::max() - byte_offset,
        "Invalid bitmap format: bitmap size overflow");

      bucket_metadata.emplace_back(byte_offset, bucket_key, bucket_meta);
      num_keys += bucket_meta.num_keys;
      byte_offset += bucket_meta.size_bytes;
    }
    size_bytes = byte_offset;
    valid      = true;
  }

  __host__ __device__ void parse(serialized_bitmap_view bitmap)
  {
    cuda::std::size_t byte_offset = 0;
    cuda::std::uint64_t serialized_num_buckets;
    if (not load(bitmap, byte_offset, serialized_num_buckets)) { return; }
    if (serialized_num_buckets > cuda::std::numeric_limits<cuda::std::size_t>::max()) {
      valid = false;
      NV_IF_TARGET(NV_IS_HOST,
                   CUCO_FAIL("Invalid bitmap format: num_buckets out of range");)  // TODO device
                                                                                   // error handling
      return;
    }
    num_buckets = static_cast<cuda::std::size_t>(serialized_num_buckets);
    byte_offset += sizeof(cuda::std::uint64_t);

    for (cuda::std::size_t i = 0; i < num_buckets; ++i) {
      cuda::std::uint32_t bucket_key;
      if (not load(bitmap, byte_offset, bucket_key)) { return; }
      byte_offset += sizeof(cuda::std::uint32_t);

      roaring_bitmap_metadata<cuda::std::uint32_t> bucket_meta{bitmap.subview(byte_offset)};
      if (not bucket_meta.valid) {
        valid = false;
        return;
      }
      if (bucket_meta.num_keys > cuda::std::numeric_limits<cuda::std::size_t>::max() - num_keys or
          bucket_meta.size_bytes >
            cuda::std::numeric_limits<cuda::std::size_t>::max() - byte_offset) {
        valid = false;
        NV_IF_TARGET(
          NV_IS_HOST,
          CUCO_FAIL("Invalid bitmap format: bitmap size overflow");)  // TODO device error handling
        return;
      }
      num_keys += bucket_meta.num_keys;
      byte_offset += bucket_meta.size_bytes;
    }
    size_bytes = byte_offset;
    valid      = true;
  }
};
}  // namespace cuco::experimental::detail
