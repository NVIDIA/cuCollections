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

#include <cooperative_groups.h>

namespace cuco {
namespace experimental {
/**
 * @brief Public double hashing scheme class.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam Hash1 Unary callable type
 * @tparam Hash2 Unary callable type
 */
template <int CGSize, typename Hash1, typename Hash2>
class double_hashing : private detail::probing_scheme_base<CGSize> {
 public:
  using probing_scheme_base_type =
    detail::probing_scheme_base<CGSize>;  ///< The base probe scheme type
  using probing_scheme_base_type::cg_size;

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
   * @tparam Extent Type of extent
   *
   * @param probe_key The probing key
   * @param upper_bound Upper bound of the iteration
   * @return An iterator whose value_type is convertible to slot index type
   */
  template <typename ProbeKey, typename Extent>
  __device__ constexpr auto operator()(ProbeKey const& probe_key, Extent upper_bound) const noexcept
  {
    return iterator<Extent>{
      hash1_(probe_key) % upper_bound,
      hash2_(probe_key) % (upper_bound - 1) + 1,  // step size in range [1, prime - 1]
      upper_bound};
  }

  /**
   * @brief Operator to return a probing iterator
   *
   * @tparam ProbeKey Type of probing key
   * @tparam Extent Type of extent
   *
   * @param g the Cooperative Group to generate probing iterator
   * @param probe_key The probing key
   * @param upper_bound Upper bound of the iteration
   * @return An iterator whose value_type is convertible to slot index type
   */
  template <typename ProbeKey, typename Extent>
  __device__ constexpr auto operator()(cooperative_groups::thread_block_tile<cg_size> const& g,
                                       ProbeKey const& probe_key,
                                       Extent upper_bound) const noexcept
  {
    return iterator<Extent>{(hash1_(probe_key) + g.thread_rank()) % upper_bound,
                            (hash2_(probe_key) % (upper_bound / cg_size - 1) + 1) * cg_size,
                            upper_bound};
  }

  /**
   * @brief Probing iterator class.
   *
   * @tparam Extent Type of Extent
   */
  template <typename Extent>
  class iterator {
   public:
    using extent_type = Extent;                            ///< Extent type
    using size_type   = typename extent_type::value_type;  ///< Size type

    /**
     *@brief Constructs an probing iterator
     *
     * @param start Iteration starting point
     * @param step_size Double hashing step size
     * @param upper_bound Upper bound of the iteration
     */
    __device__ constexpr iterator(size_type start,
                                  size_type step_size,
                                  extent_type upper_bound) noexcept
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
    extent_type upper_bound_;
  };

 private:
  Hash1 hash1_;
  Hash2 hash2_;
};

}  // namespace experimental
}  // namespace cuco
