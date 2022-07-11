/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cuda/atomic>
#include <type_traits>

namespace cuco {
namespace detail {

/**
 * @brief Base class of all reduction functors.
 *
 * @warning This class should not be used directly.
 *
 */
class reduction_functor_base {
};

template <typename T, typename Enable = void>
struct reduce_add_impl {
  template <cuda::thread_scope Scope>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T const& rhs) const noexcept
  {
    return lhs.fetch_add(rhs) + rhs;
  }
};

template <typename T, typename Enable = void>
struct reduce_min_impl {
  template <cuda::thread_scope Scope>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T const& rhs) const noexcept
  {
    return min(lhs.fetch_min(rhs), rhs);
  }
};

template <typename T, typename Enable = void>
struct reduce_max_impl {
  template <cuda::thread_scope Scope>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T const& rhs) const noexcept
  {
    return max(lhs.fetch_max(rhs), rhs);
  }
};

template <typename T, typename Enable = void>
struct reduce_count_impl {
  template <cuda::thread_scope Scope>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T const& /* rhs */) const noexcept
  {
    return ++lhs;
  }
};

// remove the following WAR once libcu++ extends FP atomics support and fixes signed integer atomics
// https://github.com/NVIDIA/libcudacxx/pull/286
template <typename T>
struct reduce_add_impl<
  T,
  typename cuda::std::enable_if<cuda::std::is_floating_point<T>::value>::type> {
  template <cuda::thread_scope Scope>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T rhs) const noexcept
  {
    if constexpr (Scope == cuda::thread_scope_system)
      return atomicAdd_system(reinterpret_cast<T*>(&lhs), rhs) + rhs;
    else if constexpr (Scope == cuda::thread_scope_device)
      return atomicAdd(reinterpret_cast<T*>(&lhs), rhs) + rhs;
    else
      return atomicAdd_block(reinterpret_cast<T*>(&lhs), rhs) + rhs;
  }
};

template <typename T>
struct reduce_min_impl<T,
                       typename cuda::std::enable_if<cuda::std::is_integral<T>::value &&
                                                     cuda::std::is_signed<T>::value>::type> {
  template <cuda::thread_scope Scope>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T const& rhs) const noexcept
  {
    using InternalT = typename cuda::std::conditional<sizeof(T) == 8, long long int, int>::type;
    InternalT* ptr  = reinterpret_cast<InternalT*>(&lhs);
    InternalT value = rhs;
    if constexpr (Scope == cuda::thread_scope_system)
      return min(atomicMin_system(ptr, value), value);
    else if constexpr (Scope == cuda::thread_scope_device)
      return min(atomicMin(ptr, value), value);
    else
      return min(atomicMin_block(ptr, value), value);
  }
};

template <typename T>
struct reduce_max_impl<T,
                       typename cuda::std::enable_if<cuda::std::is_integral<T>::value &&
                                                     cuda::std::is_signed<T>::value>::type> {
  template <cuda::thread_scope Scope>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T const& rhs) const noexcept
  {
    using InternalT = typename cuda::std::conditional<sizeof(T) == 8, long long int, int>::type;
    InternalT* ptr  = reinterpret_cast<InternalT*>(&lhs);
    InternalT value = rhs;
    if constexpr (Scope == cuda::thread_scope_system)
      return max(atomicMax_system(ptr, value), value);
    else if constexpr (Scope == cuda::thread_scope_device)
      return max(atomicMax(ptr, value), value);
    else
      return max(atomicMax_block(ptr, value), value);
  }
};

template <typename T>
struct reduce_min_impl<
  T,
  typename cuda::std::enable_if<cuda::std::is_floating_point<T>::value>::type> {
  __device__ T operator()(T lhs, T rhs) const noexcept { return min(lhs, rhs); }
};

template <typename T>
struct reduce_max_impl<
  T,
  typename cuda::std::enable_if<cuda::std::is_floating_point<T>::value>::type> {
  __device__ T operator()(T lhs, T rhs) const noexcept { return max(lhs, rhs); }
};

}  // namespace detail
}  // namespace cuco