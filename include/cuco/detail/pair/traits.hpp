/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <thrust/device_reference.h>

#include <tuple>

namespace cuco::detail {

template <typename T, typename = void>
struct is_std_pair_like : cuda::std::false_type {};

template <typename T>
struct is_std_pair_like<T,
                        cuda::std::void_t<decltype(std::get<0>(cuda::std::declval<T>())),
                                          decltype(std::get<1>(cuda::std::declval<T>()))>>
  : cuda::std::
      conditional_t<std::tuple_size<T>::value == 2, cuda::std::true_type, cuda::std::false_type> {};

template <typename T, typename = void>
struct is_cuda_std_pair_like_impl : cuda::std::false_type {};

template <typename T>
struct is_cuda_std_pair_like_impl<
  T,
  cuda::std::void_t<decltype(cuda::std::get<0>(cuda::std::declval<T>())),
                    decltype(cuda::std::get<1>(cuda::std::declval<T>())),
                    decltype(cuda::std::tuple_size<T>::value)>>
  : cuda::std::conditional_t<cuda::std::tuple_size<T>::value == 2,
                             cuda::std::true_type,
                             cuda::std::false_type> {};

template <typename T>
struct is_cuda_std_pair_like
  : is_cuda_std_pair_like_impl<cuda::std::remove_reference_t<decltype(thrust::raw_reference_cast(
      cuda::std::declval<T>()))>> {};

}  // namespace cuco::detail
