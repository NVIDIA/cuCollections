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
  get_slot(StorageRef s) : storage_{s} {}

  /**
   * @brief Indicates if the target slot `s` is filled.
   *
   * @param idx The slot index
   * @return The slot content
   */
  __device__ typename StorageRef::value_type operator()(typename StorageRef::size_type idx) const
  {
    auto const window_idx = idx / StorageRef::window_size;
    auto const intra_idx  = idx % StorageRef::window_size;
    return storage_[window_idx][intra_idx];
  }
};

/**
 * @brief Device functor returning whether the input slot indexed by `idx` is filled.
 *
 * @tparam T The slot content type
 */
template <typename T>
struct slot_is_filled {
  T empty_sentinel_;  ///< The value of the empty key sentinel

  /**
   * @brief Constructs `slot_is_filled` functor with the given empty sentinel.
   *
   * @param s Sentinel indicating empty slot
   */
  slot_is_filled(T s) : empty_sentinel_{s} {}

  /**
   * @brief Indicates if the target slot `slot` is filled.
   *
   * @tparam T Slot content type
   *
   * @param slot The slot
   * @return `true` if slot is filled
   */
  __device__ bool operator()(T slot) const
  {
    return not cuco::detail::bitwise_compare(empty_sentinel_, slot);
  }
};

/**
 * @brief Device functor returning the number of filled elements per window.
 *
 * @tparam Sentinel Empty sentinel type
 */
template <typename Sentinel>
struct elements_per_window {
  Sentinel empty_key_sentinel_;  ///< The value of the empty key sentinel

  /**
   * @brief Computes the number of filled elements per window.
   *
   * @tparam Window Window storage type
   *
   * @param window The window to query
   * @return Number of filled elements per window
   */
  template <typename Window>
  __device__ int32_t operator()(Window const& window) const
  {
    int32_t num = 0;
    for (auto const& element : window) {
      num += not cuco::detail::bitwise_compare(element, empty_key_sentinel_);
    }
    return num;
  }
};

/**
 * @brief A strong type wrapper.
 *
 * @tparam T Type of the mapped values
 */
template <typename T>
struct strong_type {
  /**
   * @brief Constructs a strong type.
   *
   * @param v Value to be wrapped as a strong type
   */
  __host__ __device__ explicit constexpr strong_type(T v) : value{v} {}

  /**
   * @brief Implicit conversion operator to the underlying value.
   *
   * @return Underlying value
   */
  __host__ __device__ constexpr operator T() const noexcept { return value; }

  T value;  ///< Underlying value
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
