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

#include <cuco/detail/probing_scheme_impl.cuh>

namespace cuco {
namespace experimental {
/**
 * @brief Public double hashing scheme class.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam Hash1 Unary callable type
 * @tparam Hash2 Unary callable type
 */
template <int CGSize,
          int WindowSize,
          enable_window_probing UsesWindowProbing,
          typename Hash1,
          typename Hash2>
class double_hashing : private detail::probing_scheme_base<CGSize, WindowSize, UsesWindowProbing> {
 public:
  using probing_scheme_base_type =
    detail::probing_scheme_base<CGSize, WindowSize, UsesWindowProbing>;  ///< The base probe scheme
                                                                         ///< type
  using probing_scheme_base_type::cg_size;
  using probing_scheme_base_type::uses_window_probing;
  using probing_scheme_base_type::window_size;

  /**
   *@brief Constructs double hashing probing scheme with the two hasher callables.
   *
   * @param hash1 First hasher
   * @param hash2 Second hasher
   */
  double_hashing(Hash1 const& hash1, Hash2 const& hash2) : hash1_{hash1}, hash2_{hash2} {}

  /**
   * @brief Operator to return a probing iterator
   *
   * @tparam ProbeKey Type of probing key
   * @tparam SizeType Type of storage size
   *
   * @param probe_key The probing key
   * @param upper_bound Upper bound of the iteration
   * @return An iterator whose value_type is convertible to slot index type
   */
  template <typename ProbeKey, typename SizeType>
  __device__ constexpr auto operator()(ProbeKey const& probe_key,
                                       SizeType const upper_bound) const noexcept
  {
    auto const hash_value = hash1_(probe_key);

    auto const [start, step_size] = [&]() {
      if constexpr (uses_window_probing == enable_window_probing::YES) {
        // step size in range [1, prime - 1] * window_size
        return thrust::pair<SizeType, SizeType>{
          hash_value % (upper_bound / window_size) * window_size,
          (hash2_(probe_key) % (upper_bound / window_size - 1) + 1) * window_size};
      }
      if constexpr (uses_window_probing == enable_window_probing::NO) {
        // step size in range [1, prime - 1]
        return thrust::pair<SizeType, SizeType>{hash_value % upper_bound,
                                                hash2_(probe_key) % (upper_bound - 1) + 1};
      }
    }();
    return iterator<SizeType>{start, step_size, upper_bound};
  }

  /**
   * @brief Probing iterator class.
   *
   * @tparam SizeType Type of size
   */
  template <typename SizeType>
  class iterator {
   public:
    using size_type = SizeType;  ///< Size type

    /**
     *@brief Constructs an probing iterator
     *
     * @param start Iteration starting point
     * @param step_size Double hashing step size
     * @param upper_bound Upper bound of the iteration
     */
    __device__ constexpr iterator(SizeType start, SizeType step_size, SizeType upper_bound) noexcept
      : curr_index_{start}, step_size_{step_size}, upper_bound_{upper_bound}
    {
    }

    /**
     * @brief Dereference operator
     *
     * @return Current slot ndex
     */
    __device__ constexpr auto operator*() const noexcept { return curr_index_; }

    /**
     * @brief Prefix increment operator
     *
     * @return Current iterator
     */
    __device__ constexpr auto operator++() noexcept
    {
      curr_index_ = (curr_index_ + step_size_) % upper_bound_;
      return *this;
    }

    /**
     * @brief Postfix increment operator
     *
     * @return Old iterator before increment
     */
    __device__ constexpr auto operator++(int) noexcept
    {
      auto temp = *this;
      ++(*this);
      return temp;
    }

   private:
    size_type curr_index_;
    size_type step_size_;
    size_type upper_bound_;
  };

 private:
  Hash1 hash1_;
  Hash2 hash2_;
};

}  // namespace experimental
}  // namespace cuco
