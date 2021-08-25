/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cstdint>
#include <type_traits>

namespace cuco {
namespace detail {
__host__ __device__ inline int cuda_memcmp(void const* __lhs, void const* __rhs, size_t __count)
{
  auto __lhs_c = reinterpret_cast<unsigned char const*>(__lhs);
  auto __rhs_c = reinterpret_cast<unsigned char const*>(__rhs);
  while (__count--) {
    auto const __lhs_v = *__lhs_c++;
    auto const __rhs_v = *__rhs_c++;
    if (__lhs_v < __rhs_v) { return -1; }
    if (__lhs_v > __rhs_v) { return 1; }
  }
  return 0;
}

template <std::size_t TypeSize>
struct bitwise_compare_impl {
  __host__ __device__ static bool compare(char const* lhs, char const* rhs)
  {
    return cuda_memcmp(lhs, rhs, TypeSize) == 0;
  }
};

template <>
struct bitwise_compare_impl<4> {
  __host__ __device__ inline static bool compare(char const* lhs, char const* rhs)
  {
    return *reinterpret_cast<uint32_t const*>(lhs) == *reinterpret_cast<uint32_t const*>(rhs);
  }
};

template <>
struct bitwise_compare_impl<8> {
  __host__ __device__ inline static bool compare(char const* lhs, char const* rhs)
  {
    return *reinterpret_cast<uint64_t const*>(lhs) == *reinterpret_cast<uint64_t const*>(rhs);
  }
};

/**
 * @brief Performs a bitwise equality comparison between the two specified objects
 *
 * @tparam T Type with unique object representations
 * @param lhs The first object
 * @param rhs The second object
 * @return If the bits in the object representations of lhs and rhs are identical.
 */
template <typename T>
__host__ __device__ bool bitwise_compare(T const& lhs, T const& rhs)
{
  static_assert(std::has_unique_object_representations_v<T>,
                "Bitwise compared objects must have unique object representation.");
  return detail::bitwise_compare_impl<sizeof(T)>::compare(reinterpret_cast<char const*>(&lhs),
                                                          reinterpret_cast<char const*>(&rhs));
}

/**
 * @brief Customization point that can be specialized to indicate that it is safe to perform bitwise
 * equality comparisons on objects of type `T`.
 *
 * By default, only types where `std::has_unique_object_representations_v<T>` is true are safe for
 * bitwise equality. However, this can be too restrictive for some types, e.g., floating point
 * types.
 *
 * User-defined specializations of `is_bitwise_comparable` are allowed, but it is the users
 * responsibility to ensure values do not occur that would lead to unexpected behavior. For example,
 * if a `NaN` bit pattern were used as the empty sentinel value, it may not compare bitwise equal to
 * other `NaN` bit patterns.
 *
 */
template <typename T, typename = void>
struct is_bitwise_comparable : std::false_type {
};

/// By default, only types with unique object representations are allowed
template <typename T>
struct is_bitwise_comparable<T, std::enable_if_t<std::has_unique_object_representations_v<T>>>
  : std::true_type {
};

/**
 * @brief Declares that a type `Type` is bitwise comparable.
 *
 */
#define CUCO_DECLARE_BITWISE_COMPARABLE(Type)           \
  namespace cuco {                                      \
  template <>                                           \
  struct is_bitwise_comparable<Type> : std::true_type { \
  };                                                    \
  }
}  // namespace detail
}  // namespace cuco
