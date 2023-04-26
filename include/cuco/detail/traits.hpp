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
 * limitations under the License.
 */

#pragma once

#include <cuco/detail/extent_base.cuh>
#include <cuco/detail/probing_scheme_base.cuh>
// #include <cuco/detail/storage/storage_base.cuh> // FIXME check inheritance
#include <cuco/storage.cuh>

#include <memory>
#include <type_traits>

namespace cuco::experimental::detail {

// Trait to check if T is a valid probing scheme type
template <typename T, typename Enable = void>
struct is_probing_scheme : std::false_type {
};

template <typename T>
struct is_probing_scheme<
  T,
  std::enable_if_t<
    std::is_base_of_v<cuco::experimental::detail::probing_scheme_base<T::cg_size>, T>>>
  : std::true_type {
};

template <typename T>
inline constexpr bool is_probing_scheme_v = is_probing_scheme<T>::value;

template <typename T, typename Enable = void>
struct is_storage : std::false_type {
};

template <typename T>
struct is_storage<
  T,
  std::enable_if_t<std::is_same_v<cuco::experimental::aow_storage<T::window_size>, T>>>
  : std::true_type {
};

/* FIXME
std::enable_if_t<std::is_base_of_v<
  typename cuco::experimental::detail::storage_base<typename T::extent_type>,
  T>>> : std::true_type {};
*/

template <typename T>
inline constexpr bool is_storage_v = is_storage<T>::value;

// Trait to check if T is a `cuco::extent`
template <typename T, typename Enable = void>
struct is_extent : std::false_type {
};

template <typename T>
struct is_extent<
  T,
  std::enable_if_t<
    std::is_base_of_v<typename cuco::experimental::detail::extent_base<typename T::value_type>, T>>>
  : std::true_type {
};

template <typename T>
inline constexpr bool is_extent_v = is_extent<T>::value;

// Trait to check if T is allocator-like
template <typename T, typename Enable = std::void_t<>>
struct is_allocator : std::false_type {
};

template <typename T>
struct is_allocator<T,
                    std::void_t<typename T::value_type,
                                decltype(std::declval<T&>().allocate(std::size_t{})),
                                decltype(std::declval<T&>().deallocate(
                                  std::declval<typename T::value_type*>(), std::size_t{}))>>
  : std::true_type {
};

template <typename T>
inline constexpr bool is_allocator_v = is_allocator<T>::value;

template <typename Key>
struct key_equal_traits {
  template <typename T, typename = std::void_t<>>
  struct is_equal_functor : std::false_type {
  };

  template <typename T>
  struct is_equal_functor<
    T,
    std::void_t<std::enable_if_t<std::is_invocable_r_v<bool, T, Key const&, Key const&>>>>
    : std::true_type {
  };

  template <typename T>
  using is_equal_functor_t = typename is_equal_functor<T>::type;

  template <typename T>
  static constexpr bool is_equal_functor_v = is_equal_functor<T>::value;
};

}  // namespace cuco::experimental::detail