/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <thrust/device_reference.h>
#include <thrust/memory.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <tuple>
#include <type_traits>

namespace cuco {
namespace detail {

/**
 * @brief Rounds `v` to the nearest power of 2 greater than or equal to `v`.
 *
 * @param v
 * @return The nearest power of 2 greater than or equal to `v`.
 */
constexpr std::size_t next_pow2(std::size_t v) noexcept
{
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return ++v;
}

/**
 * @brief Gives value to use as alignment for a pair type that is at least the
 * size of the sum of the size of the first type and second type, or 16,
 * whichever is smaller.
 */
template <typename First, typename Second>
constexpr std::size_t pair_alignment()
{
  return std::min(std::size_t{16}, next_pow2(sizeof(First) + sizeof(Second)));
}

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

/**
 * @brief Denotes the equivalent packed type based on the size of the object.
 *
 * @tparam N The size of the object
 */
template <std::size_t N>
struct packed {
  using type = void;  ///< `void` type by default
};
/**
 * @brief Denotes the packed type when the size of the object is 8.
 */
template <>
struct packed<sizeof(uint64_t)> {
  using type = uint64_t;  ///< Packed type as `uint64_t` if the size of the object is 8
};
/**
 * @brief Denotes the packed type when the size of the object is 4.
 */
template <>
struct packed<sizeof(uint32_t)> {
  using type = uint32_t;  ///< Packed type as `uint32_t` if the size of the object is 4
};
template <typename pair_type>
using packed_t = typename packed<sizeof(pair_type)>::type;

/**
 * @brief Indicates if a pair type can be packed.
 *
 * When the size of the key,value pair being inserted into the hash table is
 * equal in size to a type where atomicCAS is natively supported, it is more
 * efficient to "pack" the pair and insert it with a single atomicCAS.
 *
 * Pair types whose key and value have the same object representation may be
 * packed. Also, the `pair_type` must not contain any padding bits otherwise
 * accessing the packed value would be undefined.
 *
 * @tparam pair_type The pair type that will be packed
 *
 * @return true If the pair type can be packed
 * @return false  If the pair type cannot be packed
 */
template <typename pair_type,
          typename key_type   = typename pair_type::first_type,
          typename value_type = typename pair_type::second_type>
constexpr bool is_packable()
{
  return not std::is_void<packed_t<pair_type>>::value and
         std::has_unique_object_representations_v<pair_type>;
}

/**
 * @brief Allows viewing a pair in a packed representation.
 *
 * Used as an optimization for inserting when a pair can be inserted with a
 * single atomicCAS
 */
template <typename pair_type>
union pair_converter {
  using packed_type = packed_t<pair_type>;  ///< The packed pair type
  packed_type packed;                       ///< The pair in the packed representation
  pair_type pair;                           ///< The pair in the pair representation

  /**
   * @brief Constructs a pair converter by copying from `p`
   *
   * @tparam T Type that is convertible to `pair_type`
   *
   * @param p The pair to copy from
   */
  template <typename T>
  __device__ pair_converter(T&& p) : pair{p}
  {
  }

  /**
   * @brief Constructs a pair converter by copying from `p`
   *
   * @param p The packed data to copy from
   */
  __device__ pair_converter(packed_type p) : packed{p} {}
};

}  // namespace detail

/**
 * @brief Custom pair type
 *
 * This is necessary because `thrust::pair` is under aligned.
 *
 * @tparam First Type of the first value in the pair
 * @tparam Second Type of the second value in the pair
 */
template <typename First, typename Second>
struct alignas(detail::pair_alignment<First, Second>()) pair {
  using first_type  = First;   ///< Type of the first value in the pair
  using second_type = Second;  ///< Type of the second value in the pair

  pair()            = default;
  ~pair()           = default;
  pair(pair const&) = default;  ///< Copy constructor
  pair(pair&&)      = default;  ///< Move constructor

  /**
   * @brief Replaces the contents of the pair with another pair.
   *
   * @return Reference of the current pair object
   */
  pair& operator=(pair const&) = default;

  /**
   * @brief Replaces the contents of the pair with another pair.
   *
   * @return Reference of the current pair object
   */
  pair& operator=(pair&&) = default;

  /**
   * @brief Constructs a pair from objects `f` and `s`.
   *
   * @param f The object to copy into `first`
   * @param s The object to copy into `second`
   */
  __host__ __device__ constexpr pair(First const& f, Second const& s) : first{f}, second{s} {}

  /**
   * @brief Constructs a pair by copying from the given pair `p`.
   *
   * @tparam F Type of the first value of `p`
   * @tparam S Type of the second value of `p`
   *
   * @param p The pair to copy from
   */
  template <typename F, typename S>
  __host__ __device__ constexpr pair(pair<F, S> const& p) : first{p.first}, second{p.second}
  {
  }

  /**
   * @brief Constructs a pair from the given std::pair-like `p`.
   *
   * @tparam T Type of the pair to copy from
   *
   * @param p The input pair to copy from
   */
  template <typename T, std::enable_if_t<detail::is_std_pair_like<T>::value>* = nullptr>
  __host__ __device__ constexpr pair(T const& p)
    : pair{std::get<0>(thrust::raw_reference_cast(p)), std::get<1>(thrust::raw_reference_cast(p))}
  {
  }

  /**
   * @brief Constructs a pair from the given thrust::pair-like `p`.
   *
   * @tparam T Type of the pair to copy from
   *
   * @param p The input pair to copy from
   */
  template <typename T, std::enable_if_t<detail::is_thrust_pair_like<T>::value>* = nullptr>
  __host__ __device__ constexpr pair(T const& p)
    : pair{thrust::get<0>(thrust::raw_reference_cast(p)),
           thrust::get<1>(thrust::raw_reference_cast(p))}
  {
  }

  First first;    ///< The first value in the pair
  Second second;  ///< The second value in the pair
};

template <typename K, typename V>
using pair_type = cuco::pair<K, V>;

/**
 * @brief Creates a pair of type `pair_type`
 *
 * @tparam F
 * @tparam S
 *
 * @param f
 * @param s
 * @return pair_type with first element `f` and second element `s`.
 */
template <typename F, typename S>
__host__ __device__ pair_type<F, S> make_pair(F&& f, S&& s) noexcept
{
  return pair_type<F, S>{std::forward<F>(f), std::forward<S>(s)};
}

/**
 * @brief Tests if both elements of lhs and rhs are equal
 *
 * @tparam T1 Type of the first element of the left-hand side pair
 * @tparam T2 Type of the second element of the left-hand side pair
 * @tparam U1 Type of the first element of the right-hand side pair
 * @tparam U2 Type of the second element of the right-hand side pair
 *
 * @param lhs Left-hand side pair
 * @param rhs Right-hand side pair
 *
 * @return True if two pairs are equal. False otherwise
 */
template <class T1, class T2, class U1, class U2>
__host__ __device__ constexpr bool operator==(cuco::pair<T1, T2> const& lhs,
                                              cuco::pair<U1, U2> const& rhs) noexcept
{
  return lhs.first == rhs.first and lhs.second == rhs.second;
}

}  // namespace cuco
