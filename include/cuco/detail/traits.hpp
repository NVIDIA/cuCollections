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

#include <thrust/device_reference.h>
#include <thrust/tuple.h>

#include <tuple>
#include <type_traits>

namespace cuco::detail {

template <typename T, typename = void>
struct is_std_pair_like : std::false_type {
};

template <typename T>
struct is_std_pair_like<
  T,
  std::void_t<decltype(std::get<0>(std::declval<T>())), decltype(std::get<1>(std::declval<T>()))>>
  : std::conditional_t<std::tuple_size<T>::value == 2, std::true_type, std::false_type> {
};

template <typename T, typename = void>
struct is_thrust_pair_like_impl : std::false_type {
};

template <typename T>
struct is_thrust_pair_like_impl<T,
                                std::void_t<decltype(thrust::get<0>(std::declval<T>())),
                                            decltype(thrust::get<1>(std::declval<T>()))>>
  : std::conditional_t<thrust::tuple_size<T>::value == 2, std::true_type, std::false_type> {
};

template <typename T>
struct is_thrust_pair_like
  : is_thrust_pair_like_impl<
      std::remove_reference_t<decltype(thrust::raw_reference_cast(std::declval<T>()))>> {
};

}  // namespace cuco::detail
