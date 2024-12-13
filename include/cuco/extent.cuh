/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cstddef>
#include <cstdint>

namespace cuco {
static constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);

/**
 * @brief Static extent class.
 *
 * @tparam SizeType Size type
 * @tparam N Extent
 */
template <typename SizeType, std::size_t N = dynamic_extent>
struct extent {
  using value_type = SizeType;  ///< Extent value type

  constexpr extent() = default;

  /// Constructs from `SizeType`
  __host__ __device__ constexpr extent(SizeType) noexcept {}

  /**
   * @brief Conversion to value_type.
   *
   * @return Extent size
   */
  __host__ __device__ constexpr operator value_type() const noexcept { return N; }
};

/**
 * @brief Dynamic extent class.
 *
 * @tparam SizeType Size type
 */
template <typename SizeType>
struct extent<SizeType, dynamic_extent> {
  using value_type = SizeType;  ///< Extent value type

  /**
   * @brief Constructs extent from a given `size`.
   *
   * @param size The extent size
   */
  __host__ __device__ constexpr extent(SizeType size) noexcept : value_{size} {}

  /**
   * @brief Conversion to value_type.
   *
   * @return Extent size
   */
  __host__ __device__ constexpr operator value_type() const noexcept { return value_; }

 private:
  value_type value_;  ///< Extent value
};

/**
 * @brief Bucket extent strong type.
 *
 * @note This type is used internally and can only be constructed using the `make_bucket_extent'
 * factory method.
 *
 * @tparam SizeType Size type
 * @tparam N Extent
 *
 */
template <typename SizeType, std::size_t N = dynamic_extent>
struct bucket_extent;

/**
 * @brief Computes valid bucket extent based on given parameters.
 *
 * @note The actual capacity of a container (map/set) should be exclusively determined by the return
 * value of this utility since the output depends on the requested low-bound size, the probing
 * scheme, and the storage. This utility is used internally during container constructions while for
 * container ref constructions, it would be users' responsibility to use this function to determine
 * the input size of the ref.
 *
 * @tparam CGSize Number of elements handled per CG
 * @tparam BucketSize Number of elements handled per Bucket
 * @tparam SizeType Size type
 * @tparam N Extent
 *
 * @param ext The input extent
 *
 * @throw If the input extent is invalid
 *
 * @return Resulting valid extent
 */
template <int32_t CGSize, int32_t BucketSize, typename SizeType, std::size_t N>
[[nodiscard]] auto constexpr make_bucket_extent(extent<SizeType, N> ext);

/**
 * @brief Computes valid bucket extent/capacity based on given parameters.
 *
 * @note The actual capacity of a container (map/set) should be exclusively determined by the return
 * value of this utility since the output depends on the requested low-bound size, the probing
 * scheme, and the storage. This utility is used internally during container constructions while for
 * container ref constructions, it would be users' responsibility to use this function to determine
 * the capacity ctor argument for the container.
 *
 * @tparam CGSize Number of elements handled per CG
 * @tparam BucketSize Number of elements handled per Bucket
 * @tparam SizeType Size type
 *
 * @param size The input size
 *
 * @throw If the input size is invalid
 *
 * @return Resulting valid extent
 */
template <int32_t CGSize, int32_t BucketSize, typename SizeType>
[[nodiscard]] auto constexpr make_bucket_extent(SizeType size);

template <typename ProbingScheme, typename Storage, typename SizeType, std::size_t N>
[[nodiscard]] auto constexpr make_bucket_extent(cuco::extent<SizeType, N> ext);

template <typename ProbingScheme, typename Storage, typename SizeType>
[[nodiscard]] auto constexpr make_bucket_extent(SizeType ext);

/**
 * @brief Computes a valid bucket extent/capacity for a given container type.
 *
 * @note The actual capacity of a container (map/set) should be exclusively determined by the return
 * value of this utility since the output depends on the requested low-bound size, the probing
 * scheme, and the storage. This utility is used internally during container constructions while for
 * container ref constructions, it would be users' responsibility to use this function to determine
 * the capacity ctor argument for the container.
 *
 * @tparam Container Container type to compute the extent for
 * @tparam SizeType Size type
 * @tparam N Extent
 *
 * @param ext The input extent
 *
 * @throw If the input extent is invalid
 *
 * @return Resulting valid `bucket extent`
 */
template <typename Container, typename SizeType, std::size_t N>
[[nodiscard]] auto constexpr make_bucket_extent(extent<SizeType, N> ext);

/**
 * @brief Computes a valid capacity for a given container type.
 *
 * @note The actual capacity of a container (map/set) should be exclusively determined by the return
 * value of this utility since the output depends on the requested low-bound size, the probing
 * scheme, and the storage. This utility is used internally during container constructions while for
 * container ref constructions, it would be users' responsibility to use this function to determine
 * the capacity ctor argument for the container.
 *
 * @tparam Container Container type to compute the extent for
 * @tparam SizeType Size type
 *
 * @param size The input size
 *
 * @throw If the input size is invalid
 *
 * @return Resulting valid extent
 */
template <typename Container, typename SizeType>
[[nodiscard]] auto constexpr make_bucket_extent(SizeType size);

}  // namespace cuco

#include <cuco/detail/extent/extent.inl>
