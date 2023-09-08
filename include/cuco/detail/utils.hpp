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

#include <cuco/detail/error.hpp>
#include <cuco/detail/utility/cuda.hpp>

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

namespace cuco {
namespace detail {

template <typename Iterator>
constexpr inline index_type distance(Iterator begin, Iterator end)
{
  using category = typename std::iterator_traits<Iterator>::iterator_category;
  static_assert(std::is_base_of_v<std::random_access_iterator_tag, category>,
                "Input iterator should be a random access iterator.");
  // `int64_t` instead of arch-dependant `long int`
  return static_cast<index_type>(std::distance(begin, end));
}

/**
 * @brief C++17 constexpr backport of `std::lower_bound`.
 *
 * @tparam ForwardIt Type of input iterator
 *
 * @param first Iterator defining the start of the range to examine
 * @param last Iterator defining the start of the range to examine
 * @param value Value to compare the elements to
 *
 * @return Iterator pointing to the first element in the range [first, last) that does not satisfy
 * element < value
 */
template <class ForwardIt>
__host__ __device__ constexpr ForwardIt lower_bound(
  ForwardIt first,
  ForwardIt last,
  typename cuda::std::iterator_traits<ForwardIt>::value_type const& value)
{
  using diff_type = typename cuda::std::iterator_traits<ForwardIt>::difference_type;

  ForwardIt it{};
  diff_type count = cuda::std::distance(first, last);
  diff_type step{};

  while (count > 0) {
    it   = first;
    step = count / 2;
    cuda::std::advance(it, step);

    if (*it < value) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }

  return first;
}

/**
 * @brief Finds the largest element in the ordered range [first, last) smaller than or equal to
 * `value`.
 *
 * @tparam ForwardIt Type of input iterator
 *
 * @param first Iterator defining the start of the range to examine
 * @param last Iterator defining the start of the range to examine
 * @param value Value to compare the elements to
 *
 * @return Iterator pointing to the infimum value, else `last`
 */
template <class ForwardIt>
__host__ __device__ constexpr ForwardIt infimum(
  ForwardIt first,
  ForwardIt last,
  typename cuda::std::iterator_traits<ForwardIt>::value_type const& value)
{
  auto it = lower_bound(first, last, value);

  // If lower_bound returns the beginning, and it's not equal to value, then the value is smaller
  // than all elements.
  if (it == first && *it != value) { return last; }

  // If lower_bound returned an iterator pointing to a value larger than the given value,
  // we need to step back to get the next smallest.
  if (it == last || *it != value) { --it; }

  return it;
}

/**
 * @brief Finds the smallest element in the ordered range [first, last) larger than or equal to
 * `value`.
 *
 * @tparam ForwardIt Type of input iterator
 *
 * @param first Iterator defining the start of the range to examine
 * @param last Iterator defining the start of the range to examine
 * @param value Value to compare the elements to
 *
 * @return Iterator pointing to the supremum value, else `last`
 */
template <class ForwardIt>
__host__ __device__ constexpr ForwardIt supremum(
  ForwardIt first,
  ForwardIt last,
  typename cuda::std::iterator_traits<ForwardIt>::value_type const& value)
{
  return lower_bound(first, last, value);
}

}  // namespace detail
}  // namespace cuco
