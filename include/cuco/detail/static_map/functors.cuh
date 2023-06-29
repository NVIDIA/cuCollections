/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
 */

#pragma once

#include <cuco/detail/bitwise_compare.cuh>

#include <thrust/tuple.h>

namespace cuco {
namespace experimental {
namespace static_map_ns {
namespace detail {

/**
 * @brief Device functor returning the content of the slot indexed by `idx`.
 *
 * @tparam StorageRef Storage ref type
 */
template <typename StorageRef>
struct get_slot {
  StorageRef storage_;  ///< Storage ref

  /**
   * @brief Constructs `get_slot` functor with the given storage ref.
   *
   * @param s Input storage ref
   */
  explicit constexpr get_slot(StorageRef s) noexcept : storage_{s} {}

  /**
   * @brief Accesses the slot content with the given index.
   *
   * @param idx The slot index
   * @return The slot content
   */
  __device__ constexpr auto operator()(typename StorageRef::size_type idx) const noexcept
  {
    auto const window_idx      = idx / StorageRef::window_size;
    auto const intra_idx       = idx % StorageRef::window_size;
    auto const [first, second] = storage_[window_idx][intra_idx];
    return thrust::make_tuple(first, second);
  }
};

/**
 * @brief Device functor returning whether the input slot indexed by `idx` is filled.
 *
 * @tparam T The slot key type
 * @tparam U The slot value type
 */
template <typename T, typename U>
struct slot_is_filled {
  T empty_sentinel_;  ///< The value of the empty key sentinel

  /**
   * @brief Constructs `slot_is_filled` functor with the given empty sentinel.
   *
   * @param s Sentinel indicating empty slot
   */
  explicit constexpr slot_is_filled(T const& s) noexcept : empty_sentinel_{s} {}

  /**
   * @brief Indicates if the target slot `slot` is filled.
   *
   * @tparam U Slot content type
   *
   * @param slot The slot
   *
   * @return `true` if slot is filled
   */
  template <typename Slot>
  __device__ constexpr bool operator()(Slot const& slot) const noexcept
  {
    return not cuco::detail::bitwise_compare(empty_sentinel_, thrust::get<0>(slot));
  }

  /**
   * @brief Indicates if the target slot `slot` is filled.
   *
   * @param slot The slot
   *
   * @return `true` if slot is filled
   */
  __device__ constexpr bool operator()(cuco::pair<T, U> const& slot) const noexcept
  {
    return not cuco::detail::bitwise_compare(empty_sentinel_, slot.first);
  }
};

}  // namespace detail
}  // namespace static_map_ns
}  // namespace experimental
}  // namespace cuco
