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

#include <limits>
#include <type_traits>
#include <utils.hpp>

#include <cuco/reduction_functors.cuh>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <catch2/catch.hpp>

template <typename InputIt, typename OutputIt, typename Func>
__global__ void reduce_kernel(InputIt first, InputIt last, OutputIt out, Func func)
{
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto it  = first + tid;

  if constexpr (cuda::std::is_base_of_v<cuco::detail::reduction_functor_base, Func>) {
    while (it < last) {
      func(*reinterpret_cast<cuda::atomic<typename Func::value_type, cuda::thread_scope_device>*>(
             thrust::raw_pointer_cast(out)),
           *it);
      it += gridDim.x * blockDim.x;
    }
  } else {
    while (it < last) {
      *out = func(*out, *it);
      it += gridDim.x * blockDim.x;
    }
  }
}

template <typename InputIt, typename OutputIt, typename Func>
void reduce_seq(InputIt first, InputIt last, OutputIt out, Func func)
{
  reduce_kernel<<<1, 1>>>(first, last, out, func);
  cudaDeviceSynchronize();
}

template <typename InputIt, typename OutputIt, typename Func>
void reduce_par(InputIt first, InputIt last, OutputIt out, Func func)
{
  reduce_kernel<<<1, 1024>>>(first, last, out, func);
  cudaDeviceSynchronize();
}

template <typename Func, typename EquivFunc>
void test_case_impl(Func func, EquivFunc equiv, bool uses_external_sync)
{
  using Value = typename Func::value_type;
  CHECK(cuda::std::is_base_of_v<cuco::detail::reduction_functor_base, decltype(func)>);
  CHECK(cuda::std::is_same_v<typename decltype(func)::value_type, Value>);
  CHECK(func.uses_external_sync() == uses_external_sync);

  constexpr std::size_t num_items{100};

  thrust::device_vector<Value> values(num_items);
  thrust::sequence(values.begin(), values.end(), 1);

  thrust::device_vector<Value> results_d(3, func.identity());

  reduce_seq(values.begin(), values.end(), results_d.data() + 0, func);
  reduce_par(values.begin(), values.end(), results_d.data() + 1, func);
  reduce_seq(values.begin(), values.end(), results_d.data() + 2, equiv);

  thrust::host_vector<Value> results_h = results_d;
  auto sequential_result               = results_h[0];
  auto parallel_result                 = results_h[1];
  auto correct_result                  = results_h[2];

  CHECK(sequential_result == correct_result);
  CHECK(parallel_result == correct_result);
  CHECK(parallel_result == sequential_result);
}

template <typename T>
struct custom_plus {
  __device__ T operator()(T lhs, T rhs) const noexcept { return lhs + rhs; }
};

template <typename T>
struct custom_plus_constref {
  __device__ T operator()(T const& lhs, T const& rhs) const noexcept { return lhs + rhs; }
};

template <typename T>
struct custom_plus_sync {
  template <cuda::thread_scope Scope>
  __device__ T operator()(cuda::atomic<T, Scope>& lhs, T const& rhs) const noexcept
  {
    return lhs.fetch_add(rhs) + rhs;
  }
};

template <typename T>
struct equiv_count {
  __device__ T operator()(T const& lhs, T const& /* rhs */) const noexcept { return lhs + 1; }
};

TEMPLATE_TEST_CASE_SIG(
  "Test '+' reduction functors",
  "",
  ((typename Value, typename Func, bool UsesExternalSync), Value, Func, UsesExternalSync),
  (int32_t, cuco::detail::reduce_add_impl<int32_t>, false),
  (int64_t, cuco::detail::reduce_add_impl<int64_t>, false),
  (uint32_t, cuco::detail::reduce_add_impl<uint32_t>, false),
  (uint64_t, cuco::detail::reduce_add_impl<uint64_t>, false),
  (float, cuco::detail::reduce_add_impl<float>, false),
  (double, cuco::detail::reduce_add_impl<double>, false),
  (int32_t, thrust::plus<int32_t>, true),
  (int32_t, custom_plus_sync<int32_t>, false),
  (int32_t, custom_plus<int32_t>, true),
  (int32_t, custom_plus_constref<int32_t>, true))
{
  test_case_impl(cuco::reduction_functor<Func, Value>(cuco::identity_value<Value>(0)),
                 thrust::plus<Value>(),
                 UsesExternalSync);
}

TEMPLATE_TEST_CASE_SIG(
  "Test 'min' reduction functors",
  "",
  ((typename Value, typename Func, bool UsesExternalSync), Value, Func, UsesExternalSync),
  (int32_t, cuco::detail::reduce_min_impl<int32_t>, false),
  (int64_t, cuco::detail::reduce_min_impl<int64_t>, false),
  (uint32_t, cuco::detail::reduce_min_impl<uint32_t>, false),
  (uint64_t, cuco::detail::reduce_min_impl<uint64_t>, false),
  (float, cuco::detail::reduce_min_impl<float>, true),
  (double, cuco::detail::reduce_min_impl<double>, true),
  (int32_t, thrust::minimum<int32_t>, true))
{
  test_case_impl(cuco::reduction_functor<Func, Value>(
                   cuco::identity_value<Value>(std::numeric_limits<Value>::max())),
                 thrust::minimum<Value>(),
                 UsesExternalSync);
}

TEMPLATE_TEST_CASE_SIG(
  "Test 'max' reduction functors",
  "",
  ((typename Value, typename Func, bool UsesExternalSync), Value, Func, UsesExternalSync),
  (int32_t, cuco::detail::reduce_max_impl<int32_t>, false),
  (int64_t, cuco::detail::reduce_max_impl<int64_t>, false),
  (uint32_t, cuco::detail::reduce_max_impl<uint32_t>, false),
  (uint64_t, cuco::detail::reduce_max_impl<uint64_t>, false),
  (float, cuco::detail::reduce_max_impl<float>, true),
  (double, cuco::detail::reduce_max_impl<double>, true),
  (int32_t, thrust::maximum<int32_t>, true))
{
  test_case_impl(cuco::reduction_functor<Func, Value>(
                   cuco::identity_value<Value>(std::numeric_limits<Value>::min())),
                 thrust::maximum<Value>(),
                 UsesExternalSync);
}

TEMPLATE_TEST_CASE_SIG(
  "Test 'count' reduction functors",
  "",
  ((typename Value, typename Func, bool UsesExternalSync), Value, Func, UsesExternalSync),
  (int32_t, cuco::detail::reduce_count_impl<int32_t>, false),
  (int64_t, cuco::detail::reduce_count_impl<int64_t>, false),
  (uint32_t, cuco::detail::reduce_count_impl<uint32_t>, false),
  (uint64_t, cuco::detail::reduce_count_impl<uint64_t>, false))
{
  test_case_impl(
    cuco::reduction_functor<Func, Value>(cuco::identity_value<Value>(0)),
    equiv_count<Value>(),
    UsesExternalSync);
}