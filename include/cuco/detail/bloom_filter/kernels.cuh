/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuco/detail/__config>
#include <cuco/detail/utility/cuda.cuh>
#include <cuco/detail/utility/math.cuh>

#include <cuda/std/iterator>

#include <cstdint>

namespace cuco::detail::bloom_filter_ns {

/**
 * @brief `Size`-thread tile of a 1-D thread block whose collectives use the full-warp mask.
 *
 * Unlike a cooperative-groups tile, which syncs with a per-tile mask, the warp-uniform mask lets
 * the compiler skip the per-call convergence check, and a tile vote becomes one warp ballot.
 * This requires every thread of the warp to reach each collective, so the bulk kernels never
 * branch around them.
 *
 * @tparam Size Threads per tile, a power of two up to 32
 */
template <uint32_t Size>
struct warp_tile {
  static_assert(Size > 1 && Size <= 32 && (Size & (Size - 1)) == 0);

  static constexpr uint32_t full_mask = 0xffffffffu;  ///< All lanes of the warp
  static constexpr uint32_t tile_mask =
    Size == 32 ? full_mask : (1u << Size) - 1;  ///< Ballot bits of one tile

  /// @return Rank of the calling thread in its tile
  [[nodiscard]] __device__ uint32_t thread_rank() const noexcept { return threadIdx.x % Size; }

  /**
   * @brief Broadcasts `value` from tile rank `src` to the tile.
   *
   * @tparam T Shuffled type
   * @param value Value of the calling thread
   * @param src Source rank in the tile
   * @return `value` of thread `src`
   */
  template <class T>
  [[nodiscard]] __device__ T shfl(T value, uint32_t src) const noexcept
  {
    return __shfl_sync(full_mask, value, src, Size);
  }

  /**
   * @brief Evaluates `predicate` for all threads of the tile.
   *
   * @param predicate Predicate of the calling thread
   * @return Whether `predicate` holds for every thread of the tile
   */
  [[nodiscard]] __device__ bool all(bool predicate) const noexcept
  {
    if constexpr (Size == 32) {
      return __all_sync(full_mask, predicate);
    } else {
      auto const tile_base = threadIdx.x % 32 / Size * Size;
      return ((__ballot_sync(full_mask, predicate) >> tile_base) & tile_mask) == tile_mask;
    }
  }
};

CUCO_SUPPRESS_KERNEL_WARNINGS
template <bool ConditionalAdd, int32_t CGSize, int32_t BlockSize, class InputIt, class Ref>
__device__ void add_n_impl(InputIt first, cuco::detail::index_type n, Ref ref)
{
  using key_type = typename cuda::std::iterator_traits<InputIt>::value_type;

  if constexpr (CGSize > 1) {
    auto const idx   = cuco::detail::global_thread_id();
    auto const group = warp_tile<CGSize>{};
    auto const is_full_tile =
      static_cast<cuco::detail::index_type>(blockIdx.x + 1) * BlockSize <= n;
    if (is_full_tile) {
      key_type const& key = *(first + idx);
      ref.add_coop<ConditionalAdd>(group, key);
    } else {
      auto const is_valid = idx < n;
      ref.add_coop<ConditionalAdd>(group, first, idx, is_valid);
    }
  } else {
    auto const idx = cuco::detail::global_thread_id();
    if (idx < n) {
      key_type const& key = *(first + idx);
      ref.add<ConditionalAdd>(key);
    }
  }
}

template <bool ConditionalAdd, int32_t CGSize, int32_t BlockSize, class InputIt, class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void add_n(InputIt first,
                                                    cuco::detail::index_type n,
                                                    Ref ref)
{
  add_n_impl<ConditionalAdd, CGSize, BlockSize>(first, n, ref);
}

template <int32_t CGSize, int32_t BlockSize, class InputIt, class OutputIt, class Ref>
__device__ void contains_n_impl(InputIt first,
                                cuco::detail::index_type n,
                                OutputIt output_begin,
                                Ref ref)
{
  using key_type = typename cuda::std::iterator_traits<InputIt>::value_type;

  if constexpr (CGSize > 1) {
    auto const idx   = cuco::detail::global_thread_id();
    auto const group = warp_tile<CGSize>{};
    auto const is_full_tile =
      static_cast<cuco::detail::index_type>(blockIdx.x + 1) * BlockSize <= n;
    if (is_full_tile) {
      key_type const& key   = *(first + idx);
      *(output_begin + idx) = ref.contains_coop(group, key);
    } else {
      auto const is_valid = idx < n;
      auto const result   = ref.contains_coop(group, first, idx, is_valid);
      if (is_valid) { *(output_begin + idx) = result; }
    }
  } else {
    auto const idx = cuco::detail::global_thread_id();
    if (idx < n) {
      key_type const& key   = *(first + idx);
      *(output_begin + idx) = ref.contains(key);
    }
  }
}

template <int32_t CGSize, int32_t BlockSize, class InputIt, class OutputIt, class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void contains_n(InputIt first,
                                                         cuco::detail::index_type n,
                                                         OutputIt output_begin,
                                                         Ref ref)
{
  contains_n_impl<CGSize, BlockSize>(first, n, output_begin, ref);
}

template <bool ConditionalAdd,
          int32_t CGSize,
          int32_t BlockSize,
          class InputIt,
          class StencilIt,
          class Predicate,
          class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void add_if_n(
  InputIt first, cuco::detail::index_type n, StencilIt stencil, Predicate pred, Ref ref)
{
  using key_type = typename cuda::std::iterator_traits<InputIt>::value_type;

  if constexpr (CGSize > 1) {
    auto const idx      = cuco::detail::global_thread_id();
    auto const group    = warp_tile<CGSize>{};
    auto const in_range = idx < n;
    auto const is_valid = in_range && pred(*(stencil + idx));
    ref.template add_coop<ConditionalAdd>(group, first, idx, is_valid);
  } else {
    auto const idx = cuco::detail::global_thread_id();
    if (idx < n && pred(*(stencil + idx))) {
      key_type const& key = *(first + idx);
      ref.template add<ConditionalAdd>(key);
    }
  }
}

template <int32_t CGSize,
          int32_t BlockSize,
          class InputIt,
          class StencilIt,
          class Predicate,
          class OutputIt,
          class Ref>
CUCO_KERNEL __launch_bounds__(BlockSize) void contains_if_n(InputIt first,
                                                            cuco::detail::index_type n,
                                                            StencilIt stencil,
                                                            Predicate pred,
                                                            OutputIt output_begin,
                                                            Ref ref)
{
  using key_type = typename cuda::std::iterator_traits<InputIt>::value_type;

  if constexpr (CGSize > 1) {
    auto const idx      = cuco::detail::global_thread_id();
    auto const group    = warp_tile<CGSize>{};
    auto const in_range = idx < n;
    auto const is_valid = in_range && pred(*(stencil + idx));
    auto const result   = ref.contains_coop(group, first, idx, is_valid);
    if (in_range) { *(output_begin + idx) = is_valid ? result : false; }
  } else {
    auto const idx = cuco::detail::global_thread_id();
    if (idx < n) {
      if (pred(*(stencil + idx))) {
        key_type const& key   = *(first + idx);
        *(output_begin + idx) = ref.contains(key);
      } else {
        *(output_begin + idx) = false;
      }
    }
  }
}

}  // namespace cuco::detail::bloom_filter_ns

namespace cuco::detail {

/**
 * @brief Specialization of `cuco::detail::tile_size` for `bloom_filter_ns::warp_tile`.
 *
 * @tparam Size Threads per tile
 */
template <uint32_t Size>
struct tile_size<bloom_filter_ns::warp_tile<Size>> {
  static constexpr uint32_t value = Size;  ///< Threads per tile
};

}  // namespace cuco::detail
