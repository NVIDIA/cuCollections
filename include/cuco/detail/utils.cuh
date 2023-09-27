/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cuco/detail/bitwise_compare.cuh>

#include <thrust/tuple.h>

#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

namespace cuco {
namespace detail {

/**
 * @brief For the `n` least significant bits in the given unsigned 32-bit integer `x`,
 * returns the number of set bits.
 */
__device__ __forceinline__ int32_t count_least_significant_bits(uint32_t x, int32_t n)
{
  return __popc(x & (1 << n) - 1);
}

/**
 * @brief Converts pair to `thrust::tuple` to allow assigning to a zip iterator.
 *
 * @tparam Key The slot key type
 * @tparam Value The slot value type
 */
template <typename Key, typename Value>
struct slot_to_tuple {
  /**
   * @brief Converts a pair to a `thrust::tuple`.
   *
   * @tparam S The slot type
   *
   * @param s The slot to convert
   * @return A thrust::tuple containing `s.first` and `s.second`
   */
  template <typename S>
  __device__ thrust::tuple<Key, Value> operator()(S const& s)
  {
    return thrust::tuple<Key, Value>(s.first, s.second);
  }
};

/**
 * @brief Device functor returning whether the input slot `s` is filled.
 *
 * @tparam Key The slot key type
 */
template <typename Key>
struct slot_is_filled {
  Key empty_key_sentinel_;  ///< The value of the empty key sentinel

  /**
   * @brief Indicates if the target slot `s` is filled.
   *
   * @tparam S The slot type
   *
   * @param s The slot to query
   * @return `true` if slot `s` is filled
   */
  template <typename S>
  __device__ bool operator()(S const& s)
  {
    return not cuco::detail::bitwise_compare(thrust::get<0>(s), empty_key_sentinel_);
  }
};

/**
 * @brief A strong type wrapper.
 *
 * @tparam T Type of the mapped values
 */
template <typename T>
struct strong_type {
  /**
   * @brief Constructs a strong type.
   *
   * @param v Value to be wrapped as a strong type
   */
  __host__ __device__ explicit constexpr strong_type(T v) : value{v} {}

  /**
   * @brief Implicit conversion operator to the underlying value.
   *
   * @return Underlying value
   */
  __host__ __device__ constexpr operator T() const noexcept { return value; }

  T value;  ///< Underlying value
};

/**
 * @brief Converts a given hash value into a valid (positive) size type.
 *
 * @tparam SizeType The target type
 * @tparam HashType The input type
 *
 * @return Converted hash value
 */
template <typename SizeType, typename HashType>
__host__ __device__ constexpr SizeType sanitize_hash(HashType hash) noexcept
{
  if constexpr (cuda::std::is_signed_v<SizeType>) {
    return cuda::std::abs(static_cast<SizeType>(hash));
  } else {
    return static_cast<SizeType>(hash);
  }
}

/**
 * @brief Gives value to use as alignment for a pair type that is at least the
 * size of the sum of the size of the first type and second type, or 16,
 * whichever is smaller.
 */
template <typename First, typename Second>
constexpr std::size_t pair_alignment()
{
  return std::min(std::size_t{16}, cuda::std::bit_ceil(sizeof(First) + sizeof(Second)));
}

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

template <typename Pair>
using packed_t = typename packed<sizeof(Pair)>::type;

/**
 * @brief Indicates if a pair type can be packed.
 *
 * When the size of the key,value pair being inserted into the hash table is
 * equal in size to a type where atomicCAS is natively supported, it is more
 * efficient to "pack" the pair and insert it with a single atomicCAS.
 *
 * Pair types whose key and value have the same object representation may be
 * packed. Also, the `Pair` must not contain any padding bits otherwise
 * accessing the packed value would be undefined.
 *
 * @tparam Pair The pair type that will be packed
 *
 * @return true If the pair type can be packed
 * @return false  If the pair type cannot be packed
 */
template <typename Pair>
constexpr bool is_packable()
{
  return not std::is_void<packed_t<Pair>>::value and std::has_unique_object_representations_v<Pair>;
}

/**
 * @brief Allows viewing a pair in a packed representation.
 *
 * Used as an optimization for inserting when a pair can be inserted with a
 * single atomicCAS
 */
template <typename Pair>
union pair_converter {
  using packed_type = packed_t<Pair>;  ///< The packed pair type
  packed_type packed;                  ///< The pair in the packed representation
  Pair pair;                           ///< The pair in the pair representation

  /**
   * @brief Constructs a pair converter by copying from `p`
   *
   * @tparam T Type that is convertible to `Pair`
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
}  // namespace cuco
