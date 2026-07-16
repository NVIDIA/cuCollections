/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/atomic>

namespace cuco::reduce {

/**
 * @brief Device functor performing sum reduction, used with `insert-or-apply`
 */
struct plus {
  /**
   * @brief Performs atomic fetch_add on payload and the new value to be inserted
   *
   * @tparam T The payload type
   * @tparam Scope The cuda::thread_scope used for atomic_ref
   *
   * @param payload_ref The atomic_ref pointing to payload part of the slot
   * @param val The new value to be applied as reduction to the current value
   * in the payload.
   */
  template <typename T, cuda::thread_scope Scope>
  __device__ void operator()(cuda::atomic_ref<T, Scope> payload_ref, T const& val)
  {
    payload_ref.fetch_add(val, cuda::memory_order_relaxed);
  }
};

/**
 * @brief Device functor performing max reduction, used with `insert-or-apply`
 */
struct max {
  /**
   * @brief Performs atomic fetch_max on payload and the new value to be inserted
   *
   * @tparam T The payload type
   * @tparam Scope The cuda::thread_scope used for atomic_ref
   *
   * @param payload_ref The atomic_ref pointing to payload part of the slot
   * @param val The new value to be applied as reduction to the current value
   * in the payload.
   */
  template <typename T, cuda::thread_scope Scope>
  __device__ void operator()(cuda::atomic_ref<T, Scope> payload_ref, T const& val)
  {
    payload_ref.fetch_max(val, cuda::memory_order_relaxed);
  }
};

/**
 * @brief Device functor performing min reduction, used with `insert-or-apply`
 */
struct min {
  /**
   * @brief Performs atomic fetch_min on payload and the new value to be inserted
   *
   * @tparam T The payload type
   * @tparam Scope The cuda::thread_scope used for atomic_ref
   *
   * @param payload_ref The atomic_ref pointing to payload part of the slot
   * @param val The new value to be applied as reduction to the current value
   * in the payload.
   */
  template <typename T, cuda::thread_scope Scope>
  __device__ void operator()(cuda::atomic_ref<T, Scope> payload_ref, T const& val)
  {
    payload_ref.fetch_min(val, cuda::memory_order_relaxed);
  }
};

}  // namespace cuco::reduce