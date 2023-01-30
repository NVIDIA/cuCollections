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
 * @brief Device reference of static_set.
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
    storage_ref_type::window_size;      ///< Number of elements handled per window
  static constexpr auto scope = Scope;  ///< Thread scope

  // TODO default ctor?

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
   * @brief Gets the maximum number of elements the hash map can hold.
   *
   * @return The maximum number of elements the hash map can hold
   */
  [[nodiscard]] __host__ __device__ constexpr auto capacity() const noexcept;

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] __host__ __device__ inline constexpr key_type empty_key_sentinel() const noexcept;

  /**
   * @brief Create a reference with new operators from the current object.
   *
   * Note that this function uses move semantics and thus invalidates the current object.
   *
   * @warning Using two or more reference objects to the same container but with
   * a different set of operators concurrently is undefined behavior.
   *
   * @tparam NewOperators List of `cuco::op::*_tag` types
   *
   * @param ops List of operators, e.g., `cuco::insert`
   *
   * @return copy of `*this` with `NewOperators...`
   */
  template <typename... NewOperators>
  [[nodiscard]] __host__ __device__ auto with(NewOperators... ops) && noexcept;

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
