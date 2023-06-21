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

namespace cuco {
namespace experimental {
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
    auto const window_idx = idx / StorageRef::window_size;
    auto const intra_idx  = idx % StorageRef::window_size;
    return storage_[window_idx][intra_idx];
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
