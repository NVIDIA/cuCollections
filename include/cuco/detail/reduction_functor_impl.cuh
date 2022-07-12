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
#include <cuda/std/type_traits>

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
struct reduce_add_impl<T, typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<T>>> {
  template <cuda::thread_scope Scope,
            cuda::std::enable_if_t<Scope == cuda::thread_scope_system, bool> = true>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T rhs) const noexcept
  {
    return atomicAdd_system(reinterpret_cast<T*>(&lhs), rhs) + rhs;
  }

  template <cuda::thread_scope Scope,
            cuda::std::enable_if_t<Scope == cuda::thread_scope_device, bool> = true>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T rhs) const noexcept
  {
    return atomicAdd(reinterpret_cast<T*>(&lhs), rhs) + rhs;
  }

  template <
    cuda::thread_scope Scope,
    cuda::std::enable_if_t<Scope != cuda::thread_scope_system && Scope != cuda::thread_scope_device,
                           bool> = true>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T rhs) const noexcept
  {
    return atomicAdd_block(reinterpret_cast<T*>(&lhs), rhs) + rhs;
  }
};

template <typename T>
struct reduce_min_impl<
  T,
  typename cuda::std::enable_if_t<cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>>> {
 private:
  using internal_type = typename cuda::std::conditional_t<sizeof(T) == 8, long long int, int>;

 public:
  template <cuda::thread_scope Scope,
            cuda::std::enable_if_t<Scope == cuda::thread_scope_system, bool> = true>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T rhs) const noexcept
  {
    return min(atomicMin_system(reinterpret_cast<internal_type*>(&lhs), rhs),
               static_cast<internal_type>(rhs));
  }

  template <cuda::thread_scope Scope,
            cuda::std::enable_if_t<Scope == cuda::thread_scope_device, bool> = true>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T rhs) const noexcept
  {
    return min(atomicMin(reinterpret_cast<internal_type*>(&lhs), rhs),
               static_cast<internal_type>(rhs));
  }

  template <
    cuda::thread_scope Scope,
    cuda::std::enable_if_t<Scope != cuda::thread_scope_system && Scope != cuda::thread_scope_device,
                           bool> = true>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T rhs) const noexcept
  {
    return min(atomicMin_block(reinterpret_cast<internal_type*>(&lhs), rhs),
               static_cast<internal_type>(rhs));
  }
};

template <typename T>
struct reduce_max_impl<
  T,
  typename cuda::std::enable_if_t<cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>>> {
 private:
  using internal_type = typename cuda::std::conditional_t<sizeof(T) == 8, long long int, int>;

 public:
  template <cuda::thread_scope Scope,
            cuda::std::enable_if_t<Scope == cuda::thread_scope_system, bool> = true>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T rhs) const noexcept
  {
    return max(atomicMax_system(reinterpret_cast<internal_type*>(&lhs), rhs),
               static_cast<internal_type>(rhs));
  }

  template <cuda::thread_scope Scope,
            cuda::std::enable_if_t<Scope == cuda::thread_scope_device, bool> = true>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T rhs) const noexcept
  {
    return max(atomicMax(reinterpret_cast<internal_type*>(&lhs), rhs),
               static_cast<internal_type>(rhs));
  }

  template <
    cuda::thread_scope Scope,
    cuda::std::enable_if_t<Scope != cuda::thread_scope_system && Scope != cuda::thread_scope_device,
                           bool> = true>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T rhs) const noexcept
  {
    return max(atomicMax_block(reinterpret_cast<internal_type*>(&lhs), rhs),
               static_cast<internal_type>(rhs));
  }
};

template <typename T>
struct reduce_min_impl<T, typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<T>>> {
  __device__ T operator()(T lhs, T rhs) const noexcept { return min(lhs, rhs); }
};

template <typename T>
struct reduce_max_impl<T, typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<T>>> {
  __device__ T operator()(T lhs, T rhs) const noexcept { return max(lhs, rhs); }
};

}  // namespace detail
}  // namespace cuco