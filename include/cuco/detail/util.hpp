/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

namespace cuco {
namespace detail {

/**
 * @brief Compute the number of bits of a simple type.
 *
 * @tparam T The type we want to infer its size in bits
 *
 * @return Size of type T in bits
 */
template <typename T>
[[nodiscard]] static constexpr std::size_t type_bits() noexcept
{
  return sizeof(T) * CHAR_BIT;
}

// safe division
#ifndef SDIV
#define SDIV(x, y) (((x) + (y)-1) / (y))
#endif

}  // namespace detail
}  // namespace cuco