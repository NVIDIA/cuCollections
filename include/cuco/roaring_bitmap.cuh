/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/roaring_bitmap/roaring_bitmap_storage.cuh>
#include <cuco/roaring_bitmap_ref.cuh>
#include <cuco/utility/allocator.hpp>

#include <cuda/std/cstddef>
#include <cuda/stream_ref>

namespace cuco::experimental {

/**
 * @brief GPU-accelerated container that owns a serialized Roaring bitmap.
 *
 * The `roaring_bitmap` provides host-side bulk membership queries over a bitmap stored in the
 * [Roaring bitmap format specification](https://github.com/RoaringBitmap/RoaringFormatSpec).
 * It can be constructed by copying an existing serialized bitmap to device-accessible storage or
 * built on the GPU from an unordered sequence of potentially duplicate indices. Queries are
 * executed on the GPU.
 *
 * In addition to bulk host APIs such as `contains`/`contains_async`, this container exposes a
 * non-owning reference object via `ref()` that can be used for device-side per-thread queries.
 *
 * @tparam T Index type. Must be `cuda::std::uint32_t` or `cuda::std::uint64_t`.
 * @tparam Allocator Allocator type used to manage device-accessible storage for the serialized
 *                   bytes.
 */
template <class T, class Allocator = cuco::cuda_allocator<cuda::std::byte>>
class roaring_bitmap {
 public:
  using value_type     = T;                                             ///< Index type
  using storage_type   = detail::roaring_bitmap_storage<T, Allocator>;  ///< Storage implementation
  using allocator_type = typename storage_type::allocator_type;         ///< Allocator type
  using ref_type       = roaring_bitmap_ref<value_type>;  ///< Non-owning reference type

  /**
   * @brief Constructs a `roaring_bitmap` by copying the serialized bytes to device-accessible
   *        storage.
   *
   * @note This constructor is deprecated. Use `from_serialized` instead. The constructor remains
   *       available without a compiler deprecation attribute to avoid breaking existing code.
   * @note `bitmap` must remain valid until `stream` completes the copy. The bitmap can be used
   *       immediately by work submitted to the same stream; use an explicit dependency before
   *       accessing it from another stream.
   *
   * @param bitmap Pointer to the beginning of the serialized bitmap in host memory
   * @param alloc Allocator used to allocate device-accessible storage
   * @param stream CUDA stream used for device memory operations during construction
   *
   * @throw cuco::logic_error If the serialized bitmap header is invalid or unsupported
   */
  roaring_bitmap(cuda::std::byte const* bitmap,
                 Allocator const& alloc  = {},
                 cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}});

  /**
   * @brief Creates a `roaring_bitmap` by copying serialized bytes to device-accessible storage.
   *
   * @note `bitmap` must remain valid until `stream` completes the copy. The bitmap can be used
   *       immediately by work submitted to the same stream; use an explicit dependency before
   *       accessing it from another stream.
   *
   * @param bitmap Pointer to the beginning of the serialized bitmap in host memory
   * @param alloc Allocator used to allocate device-accessible storage
   * @param stream CUDA stream used for device memory operations during construction
   *
   * @return Bitmap containing a copy of the serialized input
   *
   * @throw cuco::logic_error If the serialized bitmap header is invalid or unsupported
   */
  [[nodiscard]] static roaring_bitmap from_serialized(cuda::std::byte const* bitmap,
                                                      Allocator const& alloc  = {},
                                                      cuda::stream_ref stream = cuda::stream_ref{
                                                        cudaStream_t{nullptr}});

  /**
   * @brief Creates a `roaring_bitmap` from an unordered sequence of indices.
   *
   * @note The input must remain valid until the construction stream completes.
   * @note This function may synchronize `stream` once to determine the exact serialized allocation
   *       size. Serialization remains stream-ordered and may still be in progress when the function
   *       returns. The object can be used immediately by work submitted to the same stream; use an
   *       explicit dependency before accessing it from another stream.
   *
   * @tparam InputIt Device-accessible random access input iterator whose value type is
   *                 `cuda::std::uint32_t`
   *
   * @param first Beginning of the sequence of indices
   * @param last End of the sequence of indices
   * @param alloc Allocator used for permanent and temporary device storage
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Bitmap containing the unique input indices
   *
   * @throw cuco::logic_error If `[first, last)` is not a valid range
   */
  template <class InputIt>
  [[nodiscard]] static roaring_bitmap from_indices(InputIt first,
                                                   InputIt last,
                                                   Allocator const& alloc  = {},
                                                   cuda::stream_ref stream = cuda::stream_ref{
                                                     cudaStream_t{nullptr}});

  /**
   * @brief Creates a `roaring_bitmap` from sorted indices that may contain duplicates.
   *
   * @note The input range must be nondecreasing. This precondition is not checked.
   * @note The input must remain valid until the construction stream completes.
   * @note This function may synchronize `stream` once to determine the exact serialized allocation
   *       size. Final serialization remains stream ordered.
   *
   * @tparam InputIt Device-accessible random access input iterator whose value type is
   *                 `cuda::std::uint32_t`
   *
   * @param first Beginning of the sorted sequence
   * @param last End of the sorted sequence
   * @param alloc Allocator used for permanent and temporary device storage
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Bitmap containing the unique input indices
   *
   * @throw cuco::logic_error If `[first, last)` is not a valid range
   */
  template <class InputIt>
  [[nodiscard]] static roaring_bitmap from_sorted_indices(
    InputIt first,
    InputIt last,
    Allocator const& alloc  = {},
    cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}});

  /**
   * @brief Creates a `roaring_bitmap` from sorted unique indices.
   *
   * @note The input range must be strictly increasing. This precondition is not checked.
   * @note The input must remain valid until the construction stream completes.
   * @note This function may synchronize `stream` once to determine the exact serialized allocation
   *       size. Final serialization remains stream ordered.
   *
   * @tparam InputIt Device-accessible random access input iterator whose value type is
   *                 `cuda::std::uint32_t`
   *
   * @param first Beginning of the sorted unique sequence
   * @param last End of the sorted unique sequence
   * @param alloc Allocator used for permanent and temporary device storage
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Bitmap containing the input indices
   *
   * @throw cuco::logic_error If `[first, last)` is not a valid range
   */
  template <class InputIt>
  [[nodiscard]] static roaring_bitmap from_sorted_unique_indices(
    InputIt first,
    InputIt last,
    Allocator const& alloc  = {},
    cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}});

  /**
   * @brief Copy constructor
   *
   * @param other The roaring_bitmap to copy from
   */
  roaring_bitmap(roaring_bitmap const& other) = default;

  /**
   * @brief Move constructor
   *
   * @param other The roaring_bitmap to move from
   */
  roaring_bitmap(roaring_bitmap&& other) = default;

  /**
   * @brief Copy assignment operator
   *
   * @param other The roaring_bitmap to copy from
   * @return Reference to this roaring_bitmap
   */
  roaring_bitmap& operator=(roaring_bitmap const& other) = default;

  /**
   * @brief Move assignment operator
   *
   * @param other The roaring_bitmap to move from
   * @return Reference to this roaring_bitmap
   */
  roaring_bitmap& operator=(roaring_bitmap&& other) = default;

  ~roaring_bitmap() = default;  ///< Destructor

  /**
   * @brief Bulk membership query for indices in `[first, last)`.
   *
   * @note This function synchronizes the given stream. For asynchronous execution use
   *       `contains_async`.
   *
   * @tparam InputIt  Device-accessible random access input iterator of indices convertible to `T`
   * @tparam OutputIt Device-accessible random access output iterator whose `value_type` is
   * constructible from `bool`
   *
   * @param first Beginning of the sequence of indices
   * @param last  End of the sequence of indices
   * @param contained Output iterator where results are written; `true` iff the corresponding index
   *                  is present in the bitmap
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  template <class InputIt, class OutputIt>
  void contains(InputIt first,
                InputIt last,
                OutputIt contained,
                cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) const;

  /**
   * @brief Asynchronously performs a bulk membership query for indices in `[first, last)`.
   *
   * @tparam InputIt  Device-accessible random access input iterator of indices convertible to `T`
   * @tparam OutputIt Device-accessible random access output iterator to `bool`
   *
   * @param first Beginning of the sequence of indices
   * @param last  End of the sequence of indices
   * @param contained Output iterator where results are written; `true` iff the corresponding index
   *                  is present in the bitmap
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  template <class InputIt, class OutputIt>
  void contains_async(InputIt first,
                      InputIt last,
                      OutputIt contained,
                      cuda::stream_ref stream = cuda::stream_ref{
                        cudaStream_t{nullptr}}) const noexcept;

  /**
   * @brief Number of indices stored in the bitmap.
   *
   * @return Count of indices in the bitmap
   */
  [[nodiscard]] cuda::std::size_t size() const noexcept;

  /**
   * @brief Checks whether the bitmap contains no indices.
   *
   * @return `true` iff `size() == 0`
   */
  [[nodiscard]] bool empty() const noexcept;

  /**
   * @brief Returns a pointer to the beginning of the serialized bitmap bytes in device-accessible
   *        storage.
   *
   * @return Pointer to the serialized storage
   */
  [[nodiscard]] cuda::std::byte const* data() const noexcept;

  /**
   * @brief Size in bytes of the serialized bitmap storage.
   *
   * @return Number of bytes occupied by the serialized bitmap
   */
  [[nodiscard]] cuda::std::size_t size_bytes() const noexcept;

  /**
   * @brief Returns the allocator used to manage device-accessible storage.
   *
   * @return Allocator instance
   */
  [[nodiscard]] allocator_type allocator() const noexcept;

  /**
   * @brief Returns a non-owning reference to the underlying bitmap suitable for device-side use.
   *
   * The returned reference type provides device functions such as `contains(T)` for per-thread
   * membership testing.
   *
   * @return Non-owning reference to the underlying bitmap
   */
  [[nodiscard]] ref_type ref() const noexcept;

 private:
  explicit roaring_bitmap(storage_type&& storage);

  storage_type storage_;  ///< Storage type
};

}  // namespace cuco::experimental

#include <cuco/detail/roaring_bitmap/roaring_bitmap.inl>
