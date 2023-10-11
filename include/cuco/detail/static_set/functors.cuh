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

namespace cuco {
namespace experimental {
namespace static_set_ns {
namespace detail {

/**
 * @brief Device functor returning whether the input slot indexed by `idx` is filled.
 *
 * @tparam T The slot content type
 */
template <typename T>
struct slot_is_filled {
  T empty_sentinel_;   ///< The value of the empty key sentinel
  T erased_sentinel_;  ///< Key value that represents an erased slot

  /**
   * @brief Constructs `slot_is_filled` functor with the given sentinels.
   *
   * @param empty_sentinel Sentinel indicating empty slot
   * @param erased_sentinel Sentinel indicating erased slot
   */
  explicit constexpr slot_is_filled(T const& empty_sentinel, T const& erased_sentinel) noexcept
    : empty_sentinel_{empty_sentinel}, erased_sentinel_{erased_sentinel}
  {
  }

  /**
   * @brief Indicates if the target slot `slot` is filled.
   *
   * @tparam T Slot content type
   *
   * @param slot The slot
   *
   * @return `true` if slot is filled
   */
  __device__ constexpr bool operator()(T const& slot) const noexcept
  {
    return not(cuco::detail::bitwise_compare(empty_sentinel_, slot) or
               cuco::detail::bitwise_compare(erased_sentinel_, slot));
  }
};

}  // namespace detail
}  // namespace static_set_ns
}  // namespace experimental
}  // namespace cuco
