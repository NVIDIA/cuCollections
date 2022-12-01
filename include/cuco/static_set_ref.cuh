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

#include <cuco/detail/open_address_container/open_address_container_ref.cuh>
#include <cuco/function.hpp>
#include <cuco/sentinel.cuh>  // TODO .hpp

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
          typename... Functions>
class static_set_ref
  : public detail::open_address_container_ref<
      static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Functions...>,
      Key,
      Scope,
      KeyEqual,
      ProbingScheme,
      StorageRef,
      Functions...>,
    public detail::function_impl<
      Functions,
      static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Functions...>>... {
  using base_type = detail::open_address_container_ref<
    static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, Functions...>,
    Key,
    Scope,
    KeyEqual,
    ProbingScheme,
    StorageRef,
    Functions...>;  // no Functions...?
 public:
  using key_type            = typename base_type::key_type;             ///< Key Type
  using probing_scheme_type = typename base_type::probing_scheme_type;  ///< Type of probing scheme
  using storage_ref_type    = typename base_type::storage_ref_type;  ///< Type of slot storage ref
  using window_type         = typename base_type::window_type;  ///< Probing scheme element type
  using value_type          = typename base_type::value_type;   ///< Probing scheme element type
  using size_type           = typename base_type::size_type;    ///< Probing scheme size type
  using key_equal = typename base_type::key_equal;  ///< Type of key equality binary callable

  /// CG size
  static constexpr int cg_size = probing_scheme_type::cg_size;
  /// Number of elements handled per window
  static constexpr int window_size = storage_ref_type::window_size;

  static constexpr cuda::thread_scope scope = Scope;  ///< Thread scope

  // TODO default/copy/move ctor

  /**
   * @brief Constructs static_set_ref.
   *
   * @param empty_key_sentinel Sentinel indicating empty key
   * @param predicate Key equality binary callable
   * @param probing_scheme Probing scheme
   * @param storage_ref Non-owning ref of slot storage
   */
  __host__ __device__ static_set_ref(cuco::sentinel::empty_key<Key> empty_key_sentinel,
                                     KeyEqual const& predicate,
                                     ProbingScheme const& probing_scheme,
                                     StorageRef storage_ref) noexcept
    : base_type{empty_key_sentinel, predicate, probing_scheme, storage_ref}
  {
  }

  /**
   * @brief Create a reference with functions.
   *
   * @tparam NewFunctions List of `cuco::function::*` types
   */
  template <typename... NewFunctions>
  using make_with_functions = static_set_ref<Key,
                                             Scope,
                                             KeyEqual,
                                             ProbingScheme,
                                             StorageRef,
                                             NewFunctions...>;  //< Type alias for the current ref
                                                                // type with a new set of functions

  /**
   * @brief Create a reference with new functions from the current object.
   *
   * @tparam NewFunctions List of `cuco::function::*` types
   *
   * @return copy of `this` with `newFunctions`
   */
  template <typename... NewFunctions>
  [[nodiscard]] __host__ __device__ auto with_functions() const
  {
    return static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, NewFunctions...>(
      this->empty_key_sentinel_,
      this->predicate_.equal_,
      this->probing_scheme_,
      this->storage_ref_);
  }

  /**
   * @brief Conversion operator for reference family.
   *
   * @tparam NewFunctions List of `cuco::function::*` types
   */
  template <typename... NewFunctions>
  [[nodiscard]] __host__ __device__
  operator static_set_ref<Key, Scope, KeyEqual, ProbingScheme, StorageRef, NewFunctions...>() const
  {
    return with_functions<NewFunctions...>();
  }

  // Mixins need to be friends with this class in order to access private members
  template <typename F, typename Ref>
  friend class detail::function_impl;
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/static_set/static_set_ref.inl>
