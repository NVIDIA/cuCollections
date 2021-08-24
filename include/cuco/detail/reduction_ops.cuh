/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
 * limitations under the License.
 */

#pragma once

namespace cuco {

/**
 * @brief `+` reduction functor that internally uses an atomic fetch-and-add
 * operation.
 *
 * @tparam T The data type used for reduction
 */
template <typename T>
struct reduce_add {
  using value_type            = T;
  static constexpr T identity = 0;

  template <cuda::thread_scope Scope, typename T2>
  __device__ T apply(cuda::atomic<T, Scope>& slot, T2 const& value) const
  {
    return slot.fetch_add(value, cuda::memory_order_relaxed);
  }
};

// remove this workaround once libcu++ extends FP atomics support
// https://github.com/NVIDIA/libcudacxx/issues/104
template <>
struct reduce_add<float> {
  using value_type                = float;
  static constexpr float identity = 0;

  template <cuda::thread_scope Scope, typename T2>
  __device__ float apply(cuda::atomic<float, Scope>& slot, T2 const& value) const
  {
    return atomicAdd(reinterpret_cast<float*>(&slot), value);
  }
};

template <>
struct reduce_add<double> {
  using value_type                 = double;
  static constexpr double identity = 0;

  template <cuda::thread_scope Scope, typename T2>
  __device__ double apply(cuda::atomic<double, Scope>& slot, T2 const& value) const
  {
    return atomicAdd(reinterpret_cast<double*>(&slot), value);
  }
};

/**
 * @brief `-` reduction functor that internally uses an atomic fetch-and-add
 * operation.
 *
 * @tparam T The data type used for reduction
 */
template <typename T>
struct reduce_sub {
  using value_type            = T;
  static constexpr T identity = 0;

  template <cuda::thread_scope Scope, typename T2>
  __device__ T apply(cuda::atomic<T, Scope>& slot, T2 const& value) const
  {
    return slot.fetch_sub(value, cuda::memory_order_relaxed);
  }
};

template <>
struct reduce_sub<float> {
  using value_type                = float;
  static constexpr float identity = 0;

  template <cuda::thread_scope Scope, typename T2>
  __device__ float apply(cuda::atomic<float, Scope>& slot, T2 const& value) const
  {
    return atomicSub(reinterpret_cast<float*>(&slot), value);
  }
};

template <>
struct reduce_sub<double> {
  using value_type                 = double;
  static constexpr double identity = 0;

  template <cuda::thread_scope Scope, typename T2>
  __device__ double apply(cuda::atomic<double, Scope>& slot, T2 const& value) const
  {
    return atomicSub(reinterpret_cast<double*>(&slot), value);
  }
};

/**
 * @brief `min` reduction functor that internally uses an atomic fetch-and-add
 * operation.
 *
 * @tparam T The data type used for reduction
 */
template <typename T>
struct reduce_min {
  using value_type            = T;
  static constexpr T identity = std::numeric_limits<T>::max();

  template <cuda::thread_scope Scope, typename T2>
  __device__ T apply(cuda::atomic<T, Scope>& slot, T2 const& value) const
  {
    return slot.fetch_min(value, cuda::memory_order_relaxed);
  }
};

template <>
struct reduce_min<float> {
  using value_type                = float;
  static constexpr float identity = std::numeric_limits<float>::max();

  template <cuda::thread_scope Scope, typename T2>
  __device__ float apply(cuda::atomic<float, Scope>& slot, T2 const& value) const
  {
    return atomicMin(reinterpret_cast<float*>(&slot), value);
  }
};

template <>
struct reduce_min<double> {
  using value_type                 = double;
  static constexpr double identity = std::numeric_limits<double>::max();

  template <cuda::thread_scope Scope, typename T2>
  __device__ double apply(cuda::atomic<double, Scope>& slot, T2 const& value) const
  {
    return atomicMin(reinterpret_cast<double*>(&slot), value);
  }
};

/**
 * @brief `max` reduction functor that internally uses an atomic fetch-and-add
 * operation.
 *
 * @tparam T The data type used for reduction
 */
template <typename T>
struct reduce_max {
  using value_type            = T;
  static constexpr T identity = std::numeric_limits<T>::lowest();

  template <cuda::thread_scope Scope, typename T2>
  __device__ T apply(cuda::atomic<T, Scope>& slot, T2 const& value) const
  {
    return slot.fetch_max(value, cuda::memory_order_relaxed);
  }
};

template <>
struct reduce_max<float> {
  using value_type                = float;
  static constexpr float identity = std::numeric_limits<float>::lowest();

  template <cuda::thread_scope Scope, typename T2>
  __device__ float apply(cuda::atomic<float, Scope>& slot, T2 const& value) const
  {
    return atomicMax(reinterpret_cast<float*>(&slot), value);
  }
};

template <>
struct reduce_max<double> {
  using value_type                 = double;
  static constexpr double identity = std::numeric_limits<double>::lowest();

  template <cuda::thread_scope Scope, typename T2>
  __device__ double apply(cuda::atomic<double, Scope>& slot, T2 const& value) const
  {
    return atomicMax(reinterpret_cast<double*>(&slot), value);
  }
};

/**
 * @brief Wrapper for a user-defined custom reduction operator.
 * @brief Internally uses an atomic compare-and-swap loop.
 *
 * @tparam T The data type used for reduction
 * @tparam Identity Neutral element under the given reduction group
 * @tparam Op Commutative and associative binary operator
 */
template <typename T,
          T Identity,
          typename Op,
          std::uint32_t BackoffBaseDelay = 8,
          std::uint32_t BackoffMaxDelay  = 256>
struct custom_op {
  using value_type            = T;
  static constexpr T identity = Identity;

  Op op;

  template <cuda::thread_scope Scope, typename T2>
  __device__ T apply(cuda::atomic<T, Scope>& slot, T2 const& value) const
  {
    [[maybe_unused]] unsigned ns = BackoffBaseDelay;

    auto old = slot.load(cuda::memory_order_relaxed);
    while (not slot.compare_exchange_strong(old, op(old, value), cuda::memory_order_relaxed)) {
#if __CUDA_ARCH__ >= 700
      // exponential backoff strategy to reduce atomic contention
      if (true) {
        asm volatile("nanosleep.u32 %0;" ::"r"((unsigned)ns) :);
        if (ns < BackoffMaxDelay) { ns *= 2; }
      }
#endif
    }
    return old;
  }
};

}  // namespace cuco