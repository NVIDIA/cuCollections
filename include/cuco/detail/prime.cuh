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
 * limitations under the License.
 */

#pragma once

namespace cuco {
namespace detail {

// safe division
#define SDIV(x, y) (((x) + (y)-1) / (y))

/**
 * @brief Indicates whether the input `num` is a prime number.
 *
 * @param num
 * @return A boolean indicating whether the input `num` is a prime number
 */
constexpr bool is_prime(std::size_t num) noexcept
{
  bool flag = true;
  // 0 and 1 are not prime numbers
  if (num == 0 || num == 1) {
    flag = false;
  } else {
    for (auto i = 2; i <= num / 2; ++i) {
      if (num % i == 0) {
        flag = false;
        break;
      }
    }
  }
  return flag;
}

/**
 * @brief Computes the smallest prime number greater than or equal to `num`.
 *
 * @param num
 * @return The smallest prime number greater than or equal to `num`
 */
constexpr std::size_t compute_prime(std::size_t num) noexcept
{
  while (not is_prime(num)) {
    num++;
  }
  return num;
}

/**
 * @brief Calculates the adjusted/valid capacity based on `CGSize` and the initial `capacity`.
 *
 * @tparam CGSize Cooperative Group size
 * @param capacity The initially requested capacity
 * @return An adjusted capacity greater than or equal to `capacity`
 */
template <std::size_t CGSize>
constexpr std::size_t get_valid_capacity(std::size_t capacity) noexcept
{
  auto const min_prime = compute_prime(SDIV(capacity, CGSize * 2));
  return min_prime * CGSize * 2;
}

}  // namespace detail

}  // namespace cuco
