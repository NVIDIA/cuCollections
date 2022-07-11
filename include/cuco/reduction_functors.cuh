/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <atomic>
#include <cuco/detail/reduction_functor_impl.cuh>

#include <cuda/atomic>
#include <limits>
#include <type_traits>

namespace cuco {

/**
 * @brief Wrapper for reduction identity value.
 *
 * @tparam T The underlying value type used for reduction
 */
template <typename T>
class identity_value {
 public:
  using type = T;
  constexpr identity_value(T const& identity) noexcept : identity_(identity) {}
  constexpr T value() const noexcept { return identity_; }
 private:
  T identity_;
};

/**
 * @brief Wrapper for a user-defined custom reduction operator.
 *
 * External synchronization, if required,
 * is established via an atomic compare-and-swap loop.
 *
 * Example:
 * \code{.cpp}
 * template <typename T>
 * struct custom_plus {
 *   __device__ T operator()(T const& lhs, T const& rhs) const noexcept {
 *     return lhs + rhs;
 *   }
 * };
 *
 * template <typename T>
 * struct custom_plus_sync {
 *   template <cuda::thread_scope Scope>
 *   __device__ T operator()(cuda::atomic<T, Scope>& lhs, T const& rhs) const noexcept {
 *     return lhs.fetch_add(rhs) + rhs;
 *   }
 * };
 *
 * int main() {
 *   cuco::identity_value<int> identity{0}; // define the identity value for the given reduction operation, i.e., op(identity, x) == x
 *
 *   auto f1 = cuco::reduction_functor<custom_plus<int>, int>(identity); // synchronized via CAS-loop
 *   auto f2 = cuco::reduction_functor<custom_plus_sync<int>, int>(identity); // implicitly synchronized
 *
 *   auto custom_plus_lambda = [] __device__ (int lhs, int rhs) noexcept { return lhs + rhs; };
 *   auto f3 = cuco::reduction_functor<decltype(custom_plus_lambda), int>(identity, custom_plus_lambda);
 * }
 * \endcode
 *
 * @tparam Func The user-defined reduction functor
 * @tparam Value The value type used for reduction
 */
template <typename Func, typename Value>
class reduction_functor : detail::reduction_functor_base {
 public:
  using value_type = Value;

  reduction_functor(cuco::identity_value<Value> identity, Func functor = Func{}) noexcept : identity_(identity), functor_(functor) {}

  template <cuda::thread_scope Scope>
  __device__ value_type operator()(cuda::atomic<value_type, Scope>& lhs, value_type const& rhs) const noexcept
  {
    if constexpr (uses_external_sync()) {
      value_type old = lhs.load(cuda::memory_order_relaxed);
      value_type desired;

      do {
        desired = functor_(old, rhs);
      } while (!lhs.compare_exchange_weak(old, desired, cuda::memory_order_release, cuda::memory_order_relaxed));

      return desired;
    } else {
      return functor_(lhs, rhs);
    }
  }

  __host__ __device__ value_type identity() const noexcept {
    return identity_.value();
  }

  __host__ __device__ static constexpr bool uses_external_sync() noexcept {
    return !atomic_invocable_ || naive_invocable_;
  }

 private:
  cuco::identity_value<value_type> identity_;
  Func functor_;
  static constexpr bool naive_invocable_ = std::is_invocable_r<value_type, Func, value_type, value_type>::value;
  static constexpr bool atomic_invocable_ =
    std::is_invocable_r<value_type, Func, cuda::atomic<value_type, cuda::thread_scope_system>&, value_type>::value ||
    std::is_invocable_r<value_type, Func, cuda::atomic<value_type, cuda::thread_scope_device>&, value_type>::value ||
    std::is_invocable_r<value_type, Func, cuda::atomic<value_type, cuda::thread_scope_block>&,  value_type>::value ||
    std::is_invocable_r<value_type, Func, cuda::atomic<value_type, cuda::thread_scope_thread>&, value_type>::value;

  static_assert(atomic_invocable_ || naive_invocable_, "Invalid operator signature.");
};

/**
 * @brief Synchronized `+` reduction functor.
 *
 * @tparam T The value type used for reduction
 */
template <typename T>
auto reduce_add() { return reduction_functor(identity_value<T>{0}, detail::reduce_add_impl<T>{}); };

/**
 * @brief Synchronized `min` reduction functor.
 *
 * @tparam T The value type used for reduction
 */
template <typename T>
auto reduce_min() { return reduction_functor(identity_value{cuda::std::numeric_limits<T>::max()}, detail::reduce_min_impl<T>{}); };

/**
 * @brief Synchronized `max` reduction functor.
 *
 * @tparam T The value type used for reduction
 */
template <typename T>
auto reduce_max() { return reduction_functor(identity_value{cuda::std::numeric_limits<T>::lowest()}, detail::reduce_max_impl<T>{}); };

/**
 * @brief Synchronized `count` reduction functor.
 *
 * @tparam T The value type used for reduction
 */
template <typename T>
auto reduce_count() { return reduction_functor(identity_value<T>{0}, detail::reduce_count_impl<T>{}); };

}  // namespace cuco
