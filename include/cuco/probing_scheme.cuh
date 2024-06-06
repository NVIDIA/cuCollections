/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cuda/std/type_traits>

#include <cooperative_groups.h>

namespace cuco {
/**
 * @brief Public linear probing scheme class.
 *
 * @note Linear probing is efficient when few collisions are present, e.g., low occupancy or low
 * multiplicity.
 *
 * @note `Hash` should be callable object type.
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
   *@brief Constructs linear probing scheme with the hasher callable.
   *
   * @param hash Hasher
   */
  __host__ __device__ constexpr linear_probing(Hash const& hash = {});

  /**
   *@brief Makes a copy of the current probing method with the given hasher
   *
   * @tparam NewHash New hasher type
   *
   * @param hash Hasher
   *
   * @return Copy of the current probing method
   */
  template <typename NewHash>
  [[nodiscard]] __host__ __device__ constexpr auto with_hash_function(
    NewHash const& hash) const noexcept;

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
  __host__ __device__ constexpr auto operator()(ProbeKey const& probe_key,
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
  __host__ __device__ constexpr auto operator()(
    cooperative_groups::thread_block_tile<cg_size> const& g,
    ProbeKey const& probe_key,
    Extent upper_bound) const noexcept;

 private:
  Hash hash_;
};

/**
 * @brief Public double hashing scheme class.
 *
 * @note Default probing scheme for cuco data structures. It shows superior performance over linear
 * probing especially when dealing with high multiplicty and/or high occupancy use cases.
 *
 * @note `Hash1` and `Hash2` should be callable object type.
 *
 * @note `Hash2` needs to be able to construct from an integer value to avoid secondary clustering.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam Hash1 Unary callable type
 * @tparam Hash2 Unary callable type
 */
template <int32_t CGSize, typename Hash1, typename Hash2 = Hash1>
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
  __host__ __device__ constexpr double_hashing(Hash1 const& hash1 = {}, Hash2 const& hash2 = {1});

  /**
   *@brief Makes a copy of the current probing method with the given hasher
   *
   * @tparam NewHash1 First new hasher type
   * @tparam NewHash2 Second new hasher type
   *
   * @param hash1 First hasher
   * @param hash2 second hasher
   *
   * @return Copy of the current probing method
   */
  template <typename NewHash1, typename NewHash2 = NewHash1>
  [[nodiscard]] __host__ __device__ constexpr auto with_hash_function(NewHash1 const& hash1,
                                                                      NewHash2 const& hash2 = {
                                                                        1}) const noexcept;

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
  __host__ __device__ constexpr auto operator()(ProbeKey const& probe_key,
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
  __host__ __device__ constexpr auto operator()(
    cooperative_groups::thread_block_tile<cg_size> const& g,
    ProbeKey const& probe_key,
    Extent upper_bound) const noexcept;

 private:
  Hash1 hash1_;
  Hash2 hash2_;
};

/**
 * @brief wrapper to test if a probing scheme is perfect
 * @tparam T is a probing scheme that does not have a method "is_perfect".
 **/
template <typename T, typename = void>
struct is_perfect_hashing : cuda::std::false_type {};

/**
 * @brief wrapper to test if a probing scheme is perfect
 * @tparam T is a probing scheme that does have a method "is_perfect".
 **/
template <typename T>
struct is_perfect_hashing<T, cuda::std::void_t<decltype(std::declval<T>().is_perfect())>>
  : cuda::std::true_type {};


/**
 * @brief Public perfect probing scheme class.
 *
 * @note Perfect hash functions guarantee no collisions. User is responsible for supplying a perfect
 * hash
 *
 * @note User must ensure that the Hash function in combination with the input key set actually
 * forma a perfect hash function
 *
 * @note User must ensure that the maximum hash values is smaller than the map's capacity
 *
 * @note `Hash` should be callable object type.
 *
 * @tparam CGSize Size of CUDA Cooperative Groups
 * @tparam Hash Unary callable type
 */
template <int32_t CGSize, typename Hash>
class perfect_probing : private detail::probing_scheme_base<CGSize> {
 public:
  using probing_scheme_base_type =
    detail::probing_scheme_base<CGSize>;  ///< The base probe scheme type
  using probing_scheme_base_type::cg_size;
  static bool constexpr is_perfect() { return true; }

  /**
   *@brief Constructs perfect probing scheme with the hasher callable.
   *
   * @param hash Hasher
   */
  __host__ __device__ constexpr perfect_probing(Hash const& hash = {});

  /**
   *@brief Makes a copy of the current probing method with the given hasher
   *
   * @tparam NewHash New hasher type
   *
   * @param hash Hasher
   *
   * @return Copy of the current probing method
   */
  template <typename NewHash>
  [[nodiscard]] __host__ __device__ constexpr auto with_hash_function(
    NewHash const& hash) const noexcept;

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
  __host__ __device__ constexpr auto operator()(ProbeKey const& probe_key,
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
  __host__ __device__ constexpr auto operator()(
    cooperative_groups::thread_block_tile<cg_size> const& g,
    ProbeKey const& probe_key,
    Extent upper_bound) const noexcept;

  /**
   * @brief Perfect probing iterator class.
   *
   * @tparam Extent Type of Extent
   */
  template <typename Extent>
  class probing_iterator {
   public:
    using extent_type = Extent;                            ///< Extent type
    using size_type   = typename extent_type::value_type;  ///< Size type

    /**
     * @brief Constructs an probing iterator
     *
     * @param start Iteration starting point
     * @param upper_bound Upper bound of the iteration
     */
    __host__ __device__ constexpr probing_iterator(size_type start,
                                                   extent_type upper_bound) noexcept
      : curr_index_{start}, upper_bound_{upper_bound}
    {
    }

    /**
     * @brief Dereference operator
     *
     * @return Current slot index
     */
    __host__ __device__ constexpr auto operator*() const noexcept { return curr_index_; }

    /**
     * @brief Prefix increment operator
     *
     * @return Current iterator
     */
    __host__ __device__ constexpr auto operator++() noexcept
    {
      // TODO: This is still getting called, and should never really be used
      curr_index_ = (curr_index_ + 1) % upper_bound_;
      // curr_index_ = 0;
      return *this;
    }

    /**
     * @brief Postfix increment operator
     *
     * @return Old iterator before increment
     */
    __host__ __device__ constexpr auto operator++(int32_t) noexcept
    {
      auto temp = *this;
      ++(*this);
      return temp;
    }

   private:
    size_type curr_index_;
    extent_type upper_bound_;
  };  // class probing_iterator
 private:
  Hash hash_;
};

}  // namespace cuco

#include <cuco/detail/probing_scheme_impl.inl>
