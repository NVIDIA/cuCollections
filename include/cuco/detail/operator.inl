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

#include <type_traits>

namespace cuco {
namespace experimental {
namespace detail {

/**
 * @brief CRTP mixin which augments a given `Reference` with an `Operator`.
 *
 * @tparam Operator Operator type, i.e., `cuco::op::*`
 * @tparam Reference The reference type.
 *
 * @note This primary template should never be instantiated.
 */
template <typename Operator, typename Reference>
class operator_impl {
  // type-dependent dummy to make diagnostics a bit nicer
  template <typename, typename>
  static constexpr bool supports_operator()
  {
    return false;
  }
  static_assert(supports_operator<Operator, Reference>(),
                "Operator type is not supported by reference type.");
};

/**
 * @brief Checks if the given `Operator` is contained in a list of `Operators`.
 *
 * @tparam Operator Operator type, i.e., `cuco::op::*`
 * @tparam Operators List of operators to search in
 *
 * @return `true` iff `Operator` is contained in `Operators`, `false` otherwise.
 */
template <typename Operator, typename... Operators>
static constexpr bool has_operator()
{
  return ((std::is_same_v<Operators, Operator>) || ...);
}

}  // namespace detail
}  // namespace experimental
}  // namespace cuco