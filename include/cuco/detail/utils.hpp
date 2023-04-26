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

#include <iterator>
#include <type_traits>

namespace cuco {
namespace detail {

using index_type = int64_t;  ///< index type for internal use

/**
 * @brief Compute the number of bits of a simple type.
 *
 * @tparam T The type we want to infer its size in bits
 *
 * @return Size of type T in bits
 */
template <typename T>
static constexpr std::size_t type_bits() noexcept
{
  return sizeof(T) * CHAR_BIT;
}

// safe division
#ifndef SDIV
#define SDIV(x, y) (((x) + (y)-1) / (y))
#endif

template <typename Kernel>
auto get_grid_size(Kernel kernel, std::size_t block_size, std::size_t dynamic_smem_bytes = 0)
{
  int grid_size{-1};
  CUCO_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &grid_size, kernel, block_size, dynamic_smem_bytes));
  int dev_id{-1};
  CUCO_CUDA_TRY(cudaGetDevice(&dev_id));
  int num_sms{-1};
  CUCO_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
  grid_size *= num_sms;
  return grid_size;
}

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
 * @tparam T Type of `value`
 *
 * @param first Iterator defining the start of the range to examine
 * @param last Iterator defining the start of the range to examine
 * @param value Value to compare the elements to
 *
 * @return Iterator pointing to the first element in the range [first, last) that does not satisfy
 * element < value
 */
template <class ForwardIt, class T>
constexpr ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value)
{
  using diff_type = typename std::iterator_traits<ForwardIt>::difference_type;

  ForwardIt it{};
  diff_type count = std::distance(first, last);
  diff_type step{};

  while (count > 0) {
    it   = first;
    step = count / 2;
    std::advance(it, step);

    if (static_cast<T>(*it) < value) {
      first = ++it;
      count -= step + 1;
    } else
      count = step;
  }

  return first;
}

}  // namespace detail

namespace experimental::detail {

// TODO docs
template <template <typename> typename Predicate, typename...>
constexpr void find_arg(...)
{  // end of parameter pack
  static_assert(Predicate<void>(), "Missing required parameter");
}

template <template <typename> typename Predicate, typename T, typename... Args>
constexpr decltype(auto) find_arg(T&& t, Args&&... args)
{
  // if the current head of the parameter pack matches the predicate
  if constexpr (Predicate<std::remove_reference_t<T>>()) { return std::forward<T>(t); }
  // else go to the next element in the pack
  if constexpr (not Predicate<std::remove_reference_t<T>>()) {
    return find_arg<Predicate>(std::forward<Args&&>(args)...);
  }
}

// TODO docs
template <typename Compare, typename...>
constexpr void find_arg(...)
{  // end of parameter pack
  static_assert(std::is_same_v<Compare, void>, "Missing required parameter");
}

template <typename Compare, typename T, typename... Args>
constexpr decltype(auto) find_arg(T&& t, Args&&... args)
{
  // if the current head of the parameter pack matches the Compare type
  if constexpr (std::is_same_v<Compare, T>) { return std::forward<T>(t); }
  // else go to the next element in the pack
  if constexpr (not std::is_same_v<Compare, T>) {
    return find_arg<Compare>(std::forward<Args&&>(args)...);
  }
}

}  // namespace experimental::detail
}  // namespace cuco
