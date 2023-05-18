/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
namespace experimental {
inline namespace op {
// TODO enum class of int32_t instead of struct
// https://github.com/NVIDIA/cuCollections/issues/239
/**
 * @brief `insert` operator tag
 */
struct insert_tag {
} inline constexpr insert;

/**
 * @brief `insert_and_find` operator tag
 */
struct insert_and_find_tag {
} inline constexpr insert_and_find;

/**
 * @brief `contains` operator tag
 */
struct contains_tag {
} inline constexpr contains;

/**
 * @brief `find` operator tag
 */
struct find_tag {
} inline constexpr find;

/**
 * @brief `get` operator tag
 */
struct get_tag {
} inline constexpr get;

/**
 * @brief `rank` operator tag
 */
struct rank_tag {
} inline constexpr rank;

/**
 * @brief `select` operator tag
 */
struct select_tag {
} inline constexpr select;

/**
 * @brief `find_next_set` operator tag
 */
struct find_next_set_tag {
} inline constexpr find_next_set;

}  // namespace op
}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/operator.inl>
