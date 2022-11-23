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

#include <cstddef>

namespace cuco {
namespace experimental {
static constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);

/**
 * @brief Extent class.
 *
 * @tparam SizeType Size type
 * @tparam N Extent
 */
template <typename SizeType, std::size_t N = dynamic_extent>
struct extent {
  using value_type = SizeType;  ///< Extent value type

  constexpr extent() = default;

  /// Constructs from `SizeType`
  __host__ __device__ constexpr explicit extent(SizeType) noexcept {}

  /**
   * @brief Conversion to value_type.
   *
   * @return Extent size
   */
  __host__ __device__ constexpr operator value_type() const noexcept { return N; }
};

/**
 * @brief Extent class.
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
  value_type value_;  ///< Size of extent
};
}  // namespace experimental
}  // namespace cuco
