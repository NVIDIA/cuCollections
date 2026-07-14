/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/utility/traits.hpp>

#include <type_traits>

namespace cuco {
namespace detail {

/**
 * @brief CRTP mixin which augments a given `Reference` with an `Operator`.
 *
 * @throw If the operator is not defined in `include/cuco/operator.hpp`
 *
 * @tparam Operator Operator type, i.e., `cuco::op::*_tag`
 * @tparam Reference The reference type.
 *
 * @note This primary template should never be instantiated.
 */
template <typename Operator, typename Reference>
class operator_impl {
  static_assert(cuco::dependent_false<Operator, Reference>,
                "Operator type is not supported by reference type.");
};

/**
 * @brief Checks if the given `Operator` is contained in a list of `Operators`.
 *
 * @tparam Operator Operator type, i.e., `cuco::op::*_tag`
 * @tparam Operators List of operators to search in
 *
 * @return `true` if `Operator` is contained in `Operators`, `false` otherwise.
 */
template <typename Operator, typename... Operators>
__host__ __device__ static constexpr bool has_operator()
{
  return ((std::is_same_v<Operators, Operator>) || ...);
}

}  // namespace detail
}  // namespace cuco
