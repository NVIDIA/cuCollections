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

#include <cuco/detail/equal_wrapper.cuh>
#include <cuco/operator.hpp>
#include <cuco/sentinel.cuh>

#include <cuda/std/atomic>

namespace cuco {
namespace experimental {

/**
 * @brief Device non-owning "ref" type that can be used in device code to perform arbitrary
 * operations defined in `include/cuco/operator.hpp`
 *
 * @note Concurrent modify and lookup will be supported if both kinds of operators are specified
 * during the ref construction.
 * @note cuCollections data stuctures always place the slot keys on the left-hand
 * side when invoking the key comparison predicate.
 * @note Ref types are trivially-copyable and are intended to be passed by value.
 * @note `ProbingScheme::cg_size` indicates how many threads are used to handle one independent
 * device operation. `cg_size == 1` uses the scalar (or non-CG) code paths.
 *
 * @throw If the size of the given key type is larger than 8 bytes
 * @throw If the given key type doesn't have unique object representations, i.e.,
 * `cuco::bitwise_comparable_v<Key> == false`
 * @throw If the probing scheme type is not inherited from `cuco::detail::probing_scheme_base`
 *
 * @tparam Key Type used for keys. Requires `cuco::is_bitwise_comparable_v<Key>` returning true
 * @tparam Scope The scope in which operations will be performed by individual threads.
 * @tparam KeyEqual Binary callable type used to compare two keys for equality
 * @tparam ProbingScheme Probing scheme (see `include/cuco/probing_scheme.cuh` for options)
 * @tparam StorageRef Storage ref type
 * @tparam Operators Device operator options defined in `include/cuco/operator.hpp`
 */
template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class static_set_ref
  : public detail::operator_impl<
      Operators,
      static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>>... {
  static_assert(sizeof(Key) <= 8, "Container does not support key types larger than 8 bytes.");

  static_assert(
    cuco::is_bitwise_comparable_v<Key>,
    "Key type must have unique object representations or have been explicitly declared as safe for "
    "bitwise comparison via specialization of cuco::is_bitwise_comparable_v<Key>.");

  static_assert(
    std::is_base_of_v<cuco::experimental::detail::probing_scheme_base<ProbingScheme::cg_size>,
                      ProbingScheme>,
    "ProbingScheme must inherit from cuco::detail::probing_scheme_base");

 public:
  using key_type            = Key;                                     ///< Key Type
  using probing_scheme_type = ProbingScheme;                           ///< Type of probing scheme
  using storage_ref_type    = StorageRef;                              ///< Type of storage ref
  using window_type         = typename storage_ref_type::window_type;  ///< Window type
  using value_type          = typename storage_ref_type::value_type;   ///< Storage element type
  using extent_type         = typename storage_ref_type::extent_type;  ///< Extent type
  using size_type           = typename storage_ref_type::size_type;    ///< Probing scheme size type
  using key_equal           = KeyEqual;  ///< Type of key equality binary callable

  static constexpr auto cg_size = probing_scheme_type::cg_size;  ///< Cooperative group size
  static constexpr auto window_size =
    storage_ref_type::window_size;             ///< Number of elements handled per window
  static constexpr auto thread_scope = Scope;  ///< Thread scope

  /**
   * @brief Constructs static_set_ref.
   *
   * @param empty_key_sentinel Sentinel indicating empty key
   * @param predicate Key equality binary callable
   * @param probing_scheme Probing scheme
   * @param storage_ref Non-owning ref of slot storage
   */
  __host__ __device__ explicit constexpr static_set_ref(
    cuco::empty_key<key_type> empty_key_sentinel,
    key_equal const& predicate,
    probing_scheme_type const& probing_scheme,
    storage_ref_type storage_ref) noexcept;

  /**
   * @brief Gets the maximum number of elements the container can hold.
   *
   * @return The maximum number of elements the container can hold
   */
  [[nodiscard]] __host__ __device__ constexpr auto capacity() const noexcept;

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] __host__ __device__ constexpr key_type empty_key_sentinel() const noexcept;

 private:
  cuco::empty_key<key_type> empty_key_sentinel_;            ///< Empty key sentinel
  detail::equal_wrapper<value_type, key_equal> predicate_;  ///< Key equality binary callable
  probing_scheme_type probing_scheme_;                      ///< Probing scheme
  storage_ref_type storage_ref_;                            ///< Slot storage ref

  // Mixins need to be friends with this class in order to access private members
  template <typename Op, typename Ref>
  friend class detail::operator_impl;
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/static_set/static_set_ref.inl>
