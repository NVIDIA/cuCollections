/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cuco/detail/traits.hpp>

namespace cuco::open_addressing_ns::detail {

/**
 * @brief Device functor returning the content of the slot indexed by `idx`
 *
 * @tparam HasPayload Flag indicating whether the slot contains a payload
 * @tparam StorageRef Storage ref type
 */
template <bool HasPayload, typename StorageRef>
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
    auto const window_idx = idx / StorageRef::window_size;
    auto const intra_idx  = idx % StorageRef::window_size;
    if constexpr (HasPayload) {
      auto const& [first, second] = storage_[window_idx][intra_idx];
      return thrust::make_tuple(first, second);
    } else {
      return storage_[window_idx][intra_idx];
    }
  }
};

/**
 * @brief Device functor returning whether the given slot is filled
 *
 * @tparam HasPayload Flag indicating whether the slot contains a payload
 * @tparam T The slot key type
 */
template <bool HasPayload, typename T>
struct slot_is_filled {
  T empty_sentinel_;   ///< The value of the empty key sentinel
  T erased_sentinel_;  ///< Key value that represents an erased slot

  /**
   * @brief Constructs `slot_is_filled` functor with the given sentinels
   *
   * @param empty_sentinel Key sentinel indicating an empty slot
   * @param erased_sentinel Key sentinel indicating an erased slot
   */
  explicit constexpr slot_is_filled(T const& empty_sentinel, T const& erased_sentinel) noexcept
    : empty_sentinel_{empty_sentinel}, erased_sentinel_{erased_sentinel}
  {
  }

  /**
   * @brief Indicates if the target slot `slot` is filled.
   *
   * @tparam S The slot type
   *
   * @param slot The slot
   *
   * @return `true` if slot is filled
   */
  template <typename S>
  __device__ constexpr bool operator()(S const& slot) const noexcept
  {
    auto const key = [&]() {
      if constexpr (HasPayload) {
        // required by thrust zip iterator in `retrieve_all`
        if constexpr (cuco::detail::is_thrust_pair_like<S>::value) {
          return thrust::get<0>(slot);
        } else {
          return slot.first;
        }
      } else {
        return slot;
      }
    }();
    return not(cuco::detail::bitwise_compare(empty_sentinel_, key) or
               cuco::detail::bitwise_compare(erased_sentinel_, key));
  }
};

}  // namespace cuco::open_addressing_ns::detail
