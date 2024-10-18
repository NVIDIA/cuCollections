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

namespace cuco {
inline namespace op {
// TODO enum class of int32_t instead of struct
// https://github.com/NVIDIA/cuCollections/issues/239
/**
 * @brief `insert` operator tag
 */
struct insert_tag {
} inline constexpr insert;  ///< `cuco::insert` operator

/**
 * @brief `insert_and_find` operator tag
 */
struct insert_and_find_tag {
} inline constexpr insert_and_find;  ///< `cuco::insert_and_find` operator

/**
 * @brief `insert_or_assign` operator tag
 */
struct insert_or_assign_tag {
} inline constexpr insert_or_assign;  ///< `cuco::insert_or_assign` operator

/**
 * @brief `insert_or_apply` operator tag
 */
struct insert_or_apply_tag {
} inline constexpr insert_or_apply;  ///< `cuco::insert_or_apply` operator

/**
 * @brief `erase` operator tag
 */
struct erase_tag {
} inline constexpr erase;  ///< `cuco::erase` operator

/**
 * @brief `contains` operator tag
 */
struct contains_tag {
} inline constexpr contains;  ///< `cuco::contains` operator

/**
 * @brief `count` operator tag
 */
struct count_tag {
} inline constexpr count;  ///< `cuco::contains` operator

/**
 * @brief `find` operator tag
 */
struct find_tag {
} inline constexpr find;  ///< `cuco::find` operator

/**
 * @brief `retrieve` operator tag
 */
struct retrieve_tag {
} inline constexpr retrieve;  ///< `cuco::retrieve` operator

/**
 * @brief `for_each` operator tag
 */
struct for_each_tag {
} inline constexpr for_each;  ///< `cuco::for_each` operator

}  // namespace op
}  // namespace cuco

#include <cuco/detail/operator.inl>
