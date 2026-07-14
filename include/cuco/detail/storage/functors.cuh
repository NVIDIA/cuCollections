/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cuco::detail {
/**
 * @brief Functor for initializing device memory with a given value
 *
 * @tparam SizeType Type used for indexing
 * @tparam T Type of value being initialized
 */
template <typename SizeType, typename T>
struct initialize_functor {
  T* const _d_ptr;  ///< Pointer to device memory
  T const _key;     ///< Value to initialize memory with

  /**
   * @brief Constructs functor for initializing device memory
   *
   * @param d_ptr Pointer to device memory to initialize
   * @param key Value to initialize memory with
   */
  __host__ __device__ initialize_functor(T* d_ptr, T key) noexcept : _d_ptr{d_ptr}, _key{key} {}

  /**
   * @brief Device function to initialize memory at given index
   *
   * @param idx Index into device memory
   */
  __device__ __forceinline__ void operator()(SizeType idx) const noexcept { _d_ptr[idx] = _key; }
};
}  // namespace cuco::detail
