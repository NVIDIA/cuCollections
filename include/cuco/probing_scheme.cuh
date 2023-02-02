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

#include <cuco/detail/probing_scheme_base.cuh>

#include <cooperative_groups.h>

namespace cuco {
namespace experimental {
/**
 * @brief Public linear probing scheme class.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam Hash Unary callable type
 */
template <int32_t CGSize, typename Hash>
class linear_probing : private detail::probing_scheme_base<CGSize> {
 public:
  using probing_scheme_base_type =
    detail::probing_scheme_base<CGSize>;  ///< The base probe scheme type
  using probing_scheme_base_type::cg_size;

  /**
   *@brief Constructs double hashing probing scheme with the two hasher callables.
   *
   * @param hash Hasher
   */
  explicit constexpr linear_probing(Hash const& hash);

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
  __device__ constexpr auto operator()(ProbeKey const& probe_key,
                                       Extent upper_bound) const noexcept;

  /**
   * @brief Operator to return a CG-based probing iterator
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
                                       Extent upper_bound) const noexcept;

 private:
  Hash hash_;
};

/**
 * @brief Public double hashing scheme class.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam Hash1 Unary callable type
 * @tparam Hash2 Unary callable type
 */
template <int32_t CGSize, typename Hash1, typename Hash2>
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
  explicit constexpr double_hashing(Hash1 const& hash1, Hash2 const& hash2);

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
  __device__ constexpr auto operator()(ProbeKey const& probe_key,
                                       Extent upper_bound) const noexcept;

  /**
   * @brief Operator to return a CG-based probing iterator
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
                                       Extent upper_bound) const noexcept;

 private:
  Hash1 hash1_;
  Hash2 hash2_;
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/probing_scheme_impl.inl>
