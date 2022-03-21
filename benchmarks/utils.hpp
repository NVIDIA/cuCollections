/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <algorithm>
#include <iterator>

/**
 * @brief Count the number of unique elements within a range
 */
template <typename Iter>
std::size_t count_unique(Iter begin, Iter end)
{
  using value_type = typename std::iterator_traits<Iter>::value_type;

  const auto size = std::distance(begin, end);
  std::vector<value_type> v(size);
  std::copy(begin, end, v.begin());
  std::sort(v.begin(), v.end());

  return std::distance(v.begin(), std::unique(v.begin(), v.end()));
}