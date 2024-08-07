/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuco/detail/bitwise_compare.cuh>
#include <cuco/detail/equal_wrapper.cuh>
#include <cuco/operator.hpp>

#include <cuda/atomic>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <thrust/tuple.h>

#include <cooperative_groups.h>

namespace cuco {

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_map_ref<
  Key,
  T,
  Scope,
  KeyEqual,
  ProbingScheme,
  StorageRef,
  Operators...>::static_map_ref(cuco::empty_key<Key> empty_key_sentinel,
                                cuco::empty_value<T> empty_value_sentinel,
                                KeyEqual const& predicate,
                                ProbingScheme const& probing_scheme,
                                cuda_thread_scope<Scope>,
                                StorageRef storage_ref) noexcept
  : impl_{
      cuco::pair{empty_key_sentinel, empty_value_sentinel}, predicate, probing_scheme, storage_ref}
{
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_map_ref<
  Key,
  T,
  Scope,
  KeyEqual,
  ProbingScheme,
  StorageRef,
  Operators...>::static_map_ref(cuco::empty_key<Key> empty_key_sentinel,
                                cuco::empty_value<T> empty_value_sentinel,
                                cuco::erased_key<Key> erased_key_sentinel,
                                KeyEqual const& predicate,
                                ProbingScheme const& probing_scheme,
                                cuda_thread_scope<Scope>,
                                StorageRef storage_ref) noexcept
  : impl_{cuco::pair{empty_key_sentinel, empty_value_sentinel},
          erased_key_sentinel,
          predicate,
          probing_scheme,
          storage_ref}
{
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename... OtherOperators>
__host__ __device__ constexpr static_map_ref<Key,
                                             T,
                                             Scope,
                                             KeyEqual,
                                             ProbingScheme,
                                             StorageRef,
                                             Operators...>::
  static_map_ref(
    static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, OtherOperators...>&&
      other) noexcept
  : impl_{std::move(other.impl_)}
{
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_map_ref<Key,
                                             T,
                                             Scope,
                                             KeyEqual,
                                             ProbingScheme,
                                             StorageRef,
                                             Operators...>::key_equal
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::key_eq()
  const noexcept
{
  return this->impl_.key_eq();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_map_ref<Key,
                                             T,
                                             Scope,
                                             KeyEqual,
                                             ProbingScheme,
                                             StorageRef,
                                             Operators...>::const_iterator
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::end()
  const noexcept
{
  return this->impl_.end();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_map_ref<Key,
                                             T,
                                             Scope,
                                             KeyEqual,
                                             ProbingScheme,
                                             StorageRef,
                                             Operators...>::iterator
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::end() noexcept
{
  return this->impl_.end();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr auto
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::capacity()
  const noexcept
{
  return impl_.capacity();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr auto
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::storage_ref()
  const noexcept
{
  return this->impl_.storage_ref();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr auto
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::probing_scheme()
  const noexcept
{
  return this->impl_.probing_scheme();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr static_map_ref<Key,
                                             T,
                                             Scope,
                                             KeyEqual,
                                             ProbingScheme,
                                             StorageRef,
                                             Operators...>::extent_type
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::window_extent()
  const noexcept
{
  return impl_.window_extent();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr Key
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
  empty_key_sentinel() const noexcept
{
  return impl_.empty_key_sentinel();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr T
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
  empty_value_sentinel() const noexcept
{
  return impl_.empty_value_sentinel();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
__host__ __device__ constexpr Key
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::
  erased_key_sentinel() const noexcept
{
  return impl_.erased_key_sentinel();
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename... NewOperators>
auto static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::with(
  NewOperators...) && noexcept
{
  return static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, NewOperators...>{
    std::move(*this)};
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename... NewOperators>
__host__ __device__ auto constexpr static_map_ref<Key,
                                                  T,
                                                  Scope,
                                                  KeyEqual,
                                                  ProbingScheme,
                                                  StorageRef,
                                                  Operators...>::with_operators(NewOperators...)
  const noexcept
{
  return static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, NewOperators...>{
    cuco::empty_key<Key>{this->empty_key_sentinel()},
    cuco::empty_value<T>{this->empty_value_sentinel()},
    this->key_eq(),
    this->impl_.probing_scheme(),
    {},
    this->impl_.storage_ref()};
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename CG, cuda::thread_scope NewScope>
__device__ constexpr auto
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::make_copy(
  CG const& tile,
  window_type* const memory_to_use,
  cuda_thread_scope<NewScope> scope) const noexcept
{
  this->impl_.make_copy(tile, memory_to_use);
  return static_map_ref<Key, T, NewScope, KeyEqual, ProbingScheme, StorageRef, Operators...>{
    cuco::empty_key<Key>{this->empty_key_sentinel()},
    cuco::empty_value<T>{this->empty_value_sentinel()},
    cuco::erased_key<Key>{this->erased_key_sentinel()},
    this->key_eq(),
    this->impl_.probing_scheme(),
    scope,
    storage_ref_type{this->window_extent(), memory_to_use}};
}

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
template <typename CG>
__device__ constexpr void
static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>::initialize(
  CG const& tile) noexcept
{
  this->impl_.initialize(tile);
}

namespace detail {

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<
  op::insert_tag,
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type = typename base_type::key_type;
  using value_type  = typename base_type::value_type;
  using mapped_type = T;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Inserts an element.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param value The element to insert
   *
   * @return True if the given element is successfully inserted
   */
  template <typename Value>
  __device__ bool insert(Value const& value) noexcept
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.insert(value);
  }

  /**
   * @brief Inserts an element.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   *
   * @return True if the given element is successfully inserted
   */
  template <typename Value>
  __device__ bool insert(cooperative_groups::thread_block_tile<cg_size> const& group,
                         Value const& value) noexcept
  {
    auto& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.insert(group, value);
  }
};

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<
  op::insert_or_assign_tag,
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type = typename base_type::key_type;
  using value_type  = typename base_type::value_type;
  using mapped_type = T;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Inserts a key-value pair `{k, v}` if it's not present in the map. Otherwise, assigns `v`
   * to the mapped_type corresponding to the key `k`.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param value The element to insert
   */
  template <typename Value>
  __device__ void insert_or_assign(Value const& value) noexcept
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");

    ref_type& ref_ = static_cast<ref_type&>(*this);

    auto const val       = ref_.impl_.heterogeneous_value(value);
    auto const key       = ref_.impl_.extract_key(val);
    auto& probing_scheme = ref_.impl_.probing_scheme();
    auto storage_ref     = ref_.impl_.storage_ref();
    auto probing_iter    = probing_scheme(key, storage_ref.window_extent());

    while (true) {
      auto const window_slots = storage_ref[*probing_iter];

      for (auto& slot_content : window_slots) {
        auto const eq_res =
          ref_.impl_.predicate_.operator()<is_insert::YES>(key, slot_content.first);
        auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
        auto slot_ptr = (storage_ref.data() + *probing_iter)->data() + intra_window_index;

        // If the key is already in the container, update the payload and return
        if (eq_res == detail::equal_result::EQUAL) {
          cuda::atomic_ref<mapped_type, Scope> payload_ref(slot_ptr->second);
          payload_ref.store(val.second, cuda::memory_order_relaxed);
          return;
        }
        if (eq_res == detail::equal_result::AVAILABLE) {
          if (attempt_insert_or_assign(slot_ptr, val)) { return; }
        }
      }
      ++probing_iter;
    }
  }

  /**
   * @brief Inserts an element.
   *
   * @brief Inserts a key-value pair `{k, v}` if it's not present in the map. Otherwise, assigns `v`
   * to the mapped_type corresponding to the key `k`.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   */
  template <typename Value>
  __device__ void insert_or_assign(cooperative_groups::thread_block_tile<cg_size> const& group,
                                   Value const& value) noexcept
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);

    auto const val       = ref_.impl_.heterogeneous_value(value);
    auto const key       = ref_.impl_.extract_key(val);
    auto& probing_scheme = ref_.impl_.probing_scheme();
    auto storage_ref     = ref_.impl_.storage_ref();
    auto probing_iter    = probing_scheme(group, key, storage_ref.window_extent());

    while (true) {
      auto const window_slots = storage_ref[*probing_iter];

      auto const [state, intra_window_index] = [&]() {
        auto res = detail::equal_result::UNEQUAL;
        for (auto i = 0; i < window_size; ++i) {
          res = ref_.impl_.predicate_.operator()<is_insert::YES>(key, window_slots[i].first);
          if (res != detail::equal_result::UNEQUAL) {
            return detail::window_probing_results{res, i};
          }
        }
        // returns dummy index `-1` for UNEQUAL
        return detail::window_probing_results{res, -1};
      }();

      auto slot_ptr = (storage_ref.data() + *probing_iter)->data() + intra_window_index;

      auto const group_contains_equal = group.ballot(state == detail::equal_result::EQUAL);
      if (group_contains_equal) {
        auto const src_lane = __ffs(group_contains_equal) - 1;
        if (group.thread_rank() == src_lane) {
          cuda::atomic_ref<mapped_type, Scope> payload_ref(slot_ptr->second);
          payload_ref.store(val.second, cuda::memory_order_relaxed);
        }
        group.sync();
        return;
      }

      auto const group_contains_available = group.ballot(state == detail::equal_result::AVAILABLE);
      if (group_contains_available) {
        auto const src_lane = __ffs(group_contains_available) - 1;
        auto const status =
          (group.thread_rank() == src_lane) ? attempt_insert_or_assign(slot_ptr, val) : false;

        // Exit if inserted or assigned
        if (group.shfl(status, src_lane)) { return; }
      } else {
        ++probing_iter;
      }
    }
  }

 private:
  /**
   * @brief Attempts to insert an element into a slot or update the matching payload with the given
   * element
   *
   * @brief Inserts a key-value pair `{k, v}` if it's not present in the map. Otherwise, assigns `v`
   * to the mapped_type corresponding to the key `k`.
   *
   * @tparam Value Input type which is implicitly convertible to 'value_type'
   *
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   *
   * @return Returns `true` if the given `value` is inserted or `value` has a match in the map.
   */
  template <typename Value>
  __device__ constexpr bool attempt_insert_or_assign(value_type* slot, Value const& value) noexcept
  {
    ref_type& ref_    = static_cast<ref_type&>(*this);
    auto expected_key = ref_.impl_.empty_slot_sentinel().first;

    cuda::atomic_ref<key_type, Scope> key_ref(slot->first);
    auto const success = key_ref.compare_exchange_strong(
      expected_key, static_cast<key_type>(value.first), cuda::memory_order_relaxed);

    // if key success or key was already present in the map
    if (success or (ref_.impl_.predicate().equal_to(value.first, expected_key) ==
                    detail::equal_result::EQUAL)) {
      // Update payload
      cuda::atomic_ref<mapped_type, Scope> payload_ref(slot->second);
      payload_ref.store(value.second, cuda::memory_order_relaxed);
      return true;
    }
    return false;
  }
};

// TODO use insert_or_apply internally
template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<
  op::insert_or_apply_tag,
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type = typename base_type::key_type;
  using value_type = typename base_type::value_type;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Inserts a key-value pair `{k, v}` if it's not present in the map. Otherwise, applies
   * `Op` binary function to the mapped_type corresponding to the key `k` and the value `v`.
   *
   * @tparam Value Input type which is implicitly convertible to 'value_type'
   * @tparam Op Callable type which is used as apply operation and can be
   *   called with arguments as Op(cuda::atomic_ref<T, Scope>, T). Op strictly must
   *   have this signature to atomically apply the operation.

   * @param value The element to insert
   * @param op The callable object to perform binary operation between existing value at the slot
   *  and the element to insert.
   *
   * @return Returns `true` if the given `value` is inserted successfully.
   */

  template <typename Value, typename Op>
  __device__ bool insert_or_apply(Value const& value, Op op)
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");

    static_assert(
      cuda::std::is_invocable_v<Op, cuda::atomic_ref<T, Scope>, T>,
      "insert_or_apply expects `Op` to be a callable as `Op(cuda::atomic_ref<T, Scope>, T)`");

    auto& ref_ = static_cast<ref_type&>(*this);

    // directly dispatch to implementation if no init element is given
    auto constexpr use_direct_apply = false;
    return ref_.insert_or_apply_impl<use_direct_apply>(value, op);
  }

  /**
   * @brief Inserts a key-value pair `{k, v}` if it's not present in the map. Otherwise, applies
   * `Op` binary function to the mapped_type corresponding to the key `k` and the value `v`.
   *
   * @tparam Value Input type which is implicitly convertible to 'value_type'
   * @tparam Init Type of init value convertible to payload type
   * @tparam Op Callable type which is used as apply operation and can be
   *   called with arguments as Op(cuda::atomic_ref<T, Scope>, T). Op strictly must
   *   have this signature to atomically apply the operation.

   * @param value The element to insert
   * @param init The init value of the op
   * @param op The callable object to perform binary operation between existing value at the slot
   *  and the element to insert.
   *
   * @return Returns `true` if the given `value` is inserted successfully.
   */
  template <typename Value,
            typename Init,
            typename Op,
            typename = cuda::std::enable_if_t<std::is_convertible_v<Value, value_type>>>
  __device__ bool insert_or_apply(Value const& value, Init init, Op op)
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");

    static_assert(
      cuda::std::is_invocable_v<Op, cuda::atomic_ref<T, Scope>, T>,
      "insert_or_apply expects `Op` to be a callable as `Op(cuda::atomic_ref<T, Scope>, T)`");

    auto& ref_ = static_cast<ref_type&>(*this);
    return ref_.dispatch_insert_or_apply(value, init, op);
  }

  /**
   * @brief Inserts a key-value pair `{k, v}` if it's not present in the map. Otherwise, applies
   * `Op` binary function to the mapped_type corresponding to the key `k` and the value `v`.
   *
   * @tparam Value Input type which is implicitly convertible to 'value_type'
   * @tparam Op Callable type which is used as apply operation and can be
   *   called with arguments as Op(cuda::atomic_ref<T, Scope>, T). Op strictly must
   *   have this signature to atomically apply the operation.
   *
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   * @param op The callable object to perform binary operation between existing value at the slot
   *  and the element to insert.
   *
   * @return Returns `true` if the given `value` is inserted successfully.
   */

  template <typename Value, typename Op>
  __device__ bool insert_or_apply(cooperative_groups::thread_block_tile<cg_size> const& group,
                                  Value const& value,
                                  Op op)
  {
    static_assert(
      cuda::std::is_invocable_v<Op, cuda::atomic_ref<T, Scope>, T>,
      "insert_or_apply expects `Op` to be a callable as `Op(cuda::atomic_ref<T, Scope>, T)`");
    auto& ref_ = static_cast<ref_type&>(*this);

    auto constexpr use_direct_apply = false;
    return ref_.insert_or_apply_impl<use_direct_apply>(group, value, op);
  }

  /**
   * @brief Inserts a key-value pair `{k, v}` if it's not present in the map. Otherwise, applies
   * `Op` binary function to the mapped_type corresponding to the key `k` and the value `v`.
   *
   * @tparam Value Input type which is implicitly convertible to 'value_type'
   * @tparam Init Type of init value convertible to payload type
   * @tparam Op Callable type which is used as apply operation and can be
   *   called with arguments as Op(cuda::atomic_ref<T, Scope>, T). Op strictly must
   *   have this signature to atomically apply the operation.
   *
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   * @param init The init value of the op
   * @param op The callable object to perform binary operation between existing value at the slot
   *  and the element to insert.
   *
   * @return Returns `true` if the given `value` is inserted successfully.
   */
  template <typename Value, typename Init, typename Op>
  __device__ bool insert_or_apply(cooperative_groups::thread_block_tile<cg_size> const& group,
                                  Value const& value,
                                  Init init,
                                  Op op)
  {
    auto& ref_ = static_cast<ref_type&>(*this);
    static_assert(
      cuda::std::is_invocable_v<Op, cuda::atomic_ref<T, Scope>, T>,
      "insert_or_apply expects `Op` to be a callable as `Op(cuda::atomic_ref<T, Scope>, T)`");

    return ref_.dispatch_insert_or_apply(group, value, init, op);
  }

 private:
  /**
   * @brief dispatches `insert_or_apply_impl` based on condition `init == empty_value_sentinel`.
   *
   * @tparam Value Input type which is implicitly convertible to 'value_type'
   * @tparam Init Type of init value convertible to payload type
   * @tparam Op Callable type which is used as apply operation and can be
   *   called with arguments as Op(cuda::atomic_ref<T, Scope>, T). Op strictly must
   *   have this signature to atomically apply the operation.
   *
   * @param value The element to insert
   * @param init The init value of the op
   * @param op The callable object to perform binary operation between existing value at the slot
   *  and the element to insert.
   *
   * @return Returns `true` if the given `value` is inserted successfully.
   */
  template <typename Value, typename Init, typename Op>
  __device__ bool dispatch_insert_or_apply(Value const& value, Init init, Op op)
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);
    // if init equals sentinel value, then we can just `apply` op instead of write
    if (cuco::detail::bitwise_compare(init, ref_.empty_value_sentinel())) {
      return ref_.insert_or_apply_impl<true>(value, op);
    } else {
      return ref_.insert_or_apply_impl<false>(value, op);
    }
  }

  /**
   * @brief dispatches `insert_or_apply_impl` based on condition `init == empty_value_sentinel`.
   *
   * @tparam Value Input type which is implicitly convertible to 'value_type'
   * @tparam Init Type of init value convertible to payload type
   * @tparam Op Callable type which is used as apply operation and can be
   *   called with arguments as Op(cuda::atomic_ref<T, Scope>, T). Op strictly must
   *   have this signature to atomically apply the operation.
   *
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   * @param init The init value of the op
   * @param op The callable object to perform binary operation between existing value at the slot
   *  and the element to insert.
   */
  template <typename Value, typename Init, typename Op>
  __device__ bool dispatch_insert_or_apply(
    cooperative_groups::thread_block_tile<cg_size> const& group,
    Value const& value,
    Init init,
    Op op)
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);
    // if init equals sentinel value, then we can just `apply` op instead of write
    if (cuco::detail::bitwise_compare(init, ref_.empty_value_sentinel())) {
      return ref_.insert_or_apply_impl<true>(group, value, op);
    } else {
      return ref_.insert_or_apply_impl<false>(group, value, op);
    }
  }

  /**
   * @brief Inserts a key-value pair `{k, v}` if it's not present in the map. Otherwise, applies
   * `Op` binary function to the mapped_type corresponding to the key `k` and the value `v`.
   *
   * @tparam UseDirectApply Boolean condition which enables direct apply `op` instead of
   * wait_for_payload with an atomic_store on an empty slot.
   * @tparam Value Input type which is implicitly convertible to 'value_type'
   * @tparam Init Type of init value convertible to payload type
   * @tparam Op Callable type which is used as apply operation and can be
   *   called with arguments as Op(cuda::atomic_ref<T, Scope>, T). Op strictly must
   *   have this signature to atomically apply the operation.
   *
   * @param value The element to insert
   * @param op The callable object to perform binary operation between existing value at the slot
   *  and the element to insert.
   *
   * @return Returns `true` if the given `value` is inserted successfully.
   */
  template <bool UseDirectApply, typename Value, typename Op>
  __device__ bool insert_or_apply_impl(Value const& value, Op op)
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);

    auto const val         = ref_.impl_.heterogeneous_value(value);
    auto const key         = ref_.impl_.extract_key(val);
    auto& probing_scheme   = ref_.impl_.probing_scheme();
    auto storage_ref       = ref_.impl_.storage_ref();
    auto probing_iter      = probing_scheme(key, storage_ref.window_extent());
    auto const empty_value = ref_.empty_value_sentinel();

    // wait for payload only when init != sentinel and insert strategy is not `packed_cas`
    auto constexpr wait_for_payload = (not UseDirectApply) and (sizeof(value_type) > 8);

    while (true) {
      auto const window_slots = storage_ref[*probing_iter];

      for (auto& slot_content : window_slots) {
        auto const eq_res =
          ref_.impl_.predicate_.operator()<is_insert::YES>(key, slot_content.first);
        auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
        auto slot_ptr = (storage_ref.data() + *probing_iter)->data() + intra_window_index;

        // If the key is already in the container, update the payload and return
        if (eq_res == detail::equal_result::EQUAL) {
          // wait for payload only when performing insert operation
          if constexpr (wait_for_payload) {
            ref_.impl_.wait_for_payload(slot_ptr->second, empty_value);
          }
          op(cuda::atomic_ref<T, Scope>{slot_ptr->second}, val.second);
          return false;
        }
        if (eq_res == detail::equal_result::AVAILABLE) {
          switch (ref_.attempt_insert_or_apply<UseDirectApply>(slot_ptr, slot_content, val, op)) {
            case insert_result::SUCCESS: return true;
            case insert_result::DUPLICATE: {
              // wait for payload only when performing insert operation
              if constexpr (wait_for_payload) {
                ref_.impl_.wait_for_payload(slot_ptr->second, empty_value);
              }
              op(cuda::atomic_ref<T, Scope>{slot_ptr->second}, val.second);
              return false;
            }
            default: continue;
          }
        }
      }
      ++probing_iter;
    }
  }

  /**
   * @brief Inserts a key-value pair `{k, v}` if it's not present in the map. Otherwise, applies
   * `Op` binary function to the mapped_type corresponding to the key `k` and the value `v`.
   *
   * @tparam UseDirectApply Boolean condition which enables direct apply `op` instead of
   * wait_for_payload with an atomic_store on an empty slot.
   * @tparam Value Input type which is implicitly convertible to 'value_type'
   * @tparam Init Type of init value convertible to payload type
   * @tparam Op Callable type which is used as apply operation and can be
   *   called with arguments as Op(cuda::atomic_ref<T, Scope>, T). Op strictly must
   *   have this signature to atomically apply the operation.
   *
   * @param group The Cooperative Group used to perform group insert
   * @param value The element to insert
   * @param init The init value of the op
   * @param op The callable object to perform binary operation between existing value at the slot
   *  and the element to insert.
   *
   * @return Returns `true` if the given `value` is inserted successfully.
   */
  template <bool UseDirectApply, typename Value, typename Op>
  __device__ bool insert_or_apply_impl(cooperative_groups::thread_block_tile<cg_size> const& group,
                                       Value const& value,
                                       Op op)
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);

    auto const val         = ref_.impl_.heterogeneous_value(value);
    auto const key         = ref_.impl_.extract_key(val);
    auto& probing_scheme   = ref_.impl_.probing_scheme();
    auto storage_ref       = ref_.impl_.storage_ref();
    auto probing_iter      = probing_scheme(group, key, storage_ref.window_extent());
    auto const empty_value = ref_.empty_value_sentinel();

    // wait for payload only when init != sentinel and insert strategy is not `packed_cas`
    auto constexpr wait_for_payload = (not UseDirectApply) and (sizeof(value_type) > 8);

    while (true) {
      auto const window_slots = storage_ref[*probing_iter];

      auto const [state, intra_window_index] = [&]() {
        auto res = detail::equal_result::UNEQUAL;
        for (auto i = 0; i < window_size; ++i) {
          res = ref_.impl_.predicate_.operator()<is_insert::YES>(key, window_slots[i].first);
          if (res != detail::equal_result::UNEQUAL) {
            return detail::window_probing_results{res, i};
          }
        }
        // returns dummy index `-1` for UNEQUAL
        return detail::window_probing_results{res, -1};
      }();

      auto* slot_ptr = (storage_ref.data() + *probing_iter)->data() + intra_window_index;

      auto const group_contains_equal = group.ballot(state == detail::equal_result::EQUAL);
      if (group_contains_equal) {
        auto const src_lane = __ffs(group_contains_equal) - 1;
        if (group.thread_rank() == src_lane) {
          if constexpr (wait_for_payload) {
            ref_.impl_.wait_for_payload(slot_ptr->second, empty_value);
          }
          op(cuda::atomic_ref<T, Scope>{slot_ptr->second}, val.second);
        }
        return false;
      }

      auto const group_contains_available = group.ballot(state == detail::equal_result::AVAILABLE);
      if (group_contains_available) {
        auto const src_lane = __ffs(group_contains_available) - 1;
        auto const status   = [&, target_idx = intra_window_index]() {
          if (group.thread_rank() != src_lane) { return insert_result::CONTINUE; }
          return ref_.attempt_insert_or_apply<UseDirectApply>(
            slot_ptr, window_slots[target_idx], val, op);
        }();

        switch (group.shfl(status, src_lane)) {
          case insert_result::SUCCESS: return true;
          case insert_result::DUPLICATE: {
            if (group.thread_rank() == src_lane) {
              if constexpr (wait_for_payload) {
                ref_.impl_.wait_for_payload(slot_ptr->second, empty_value);
              }
              op(cuda::atomic_ref<T, Scope>{slot_ptr->second}, val.second);
            }
            return false;
          }
          default: continue;
        }
      } else {
        ++probing_iter;
      }
    }
  }

  /**
   * @brief Attempts to insert a key-value pair `{k, v}` if it's not present in the map. Otherwise,
   * applies `Op` binary function to the mapped_type corresponding to the key `k` and the value `v`.
   *
   * @tparam UseDirectApply Boolean condition which enables direct apply `op` instead of
   * wait_for_payload with an atomic_store on an empty slot when insert strategy is not
   * `packed_cas`.
   * @tparam Value Input type which is implicitly convertible to 'value_type'
   * @tparam Op Callable type which is used as apply operation and can be
   *   called with arguments as Op(cuda::atomic_ref<T, Scope>, T). Op strictly must
   *   have this signature to atomically apply the operation.
   *
   * @param address Pointer to the slot in memory
   * @param expected Element to compare against
   * @param desired Element to insert
   * @param op The callable object to perform binary operation between existing value at the slot
   *  and the element to insert.
   */
  template <bool UseDirectApply, typename Value, typename Op>
  [[nodiscard]] __device__ insert_result attempt_insert_or_apply(value_type* address,
                                                                 value_type const& expected,
                                                                 Value const& desired,
                                                                 Op op) noexcept
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);

    if constexpr (sizeof(value_type) <= 8) {
      return ref_.impl_.packed_cas(address, expected, desired);  // no need to wait for payload
    } else {
      using mapped_type = T;

      cuda::atomic_ref<key_type, Scope> key_ref(address->first);
      auto expected_key  = expected.first;
      auto const success = key_ref.compare_exchange_strong(
        expected_key, static_cast<key_type>(desired.first), cuda::memory_order_relaxed);

      // if key success
      if (success) {
        cuda::atomic_ref<mapped_type, Scope> payload_ref(address->second);
        // if init values == sentinel then directly apply the `op`
        if constexpr (UseDirectApply) {
          op(payload_ref, desired.second);
        } else {
          payload_ref.store(desired.second, cuda::memory_order_relaxed);
        }
        return insert_result::SUCCESS;
      }

      // Our key was already present in the slot, so our key is a duplicate
      // Shouldn't use `predicate` operator directly since it includes a redundant bitwise compare
      if (ref_.impl_.predicate_.equal_to(desired.first, expected_key) ==
          detail::equal_result::EQUAL) {
        return insert_result::DUPLICATE;
      }

      return insert_result::CONTINUE;
    }
  }
};

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<
  op::insert_and_find_tag,
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type = typename base_type::key_type;
  using value_type     = typename base_type::value_type;
  using mapped_type    = T;
  using iterator       = typename base_type::iterator;
  using const_iterator = typename base_type::const_iterator;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Inserts the given element into the map.
   *
   * @note This API returns a pair consisting of an iterator to the inserted element (or to the
   * element that prevented the insertion) and a `bool` denoting whether the insertion took place or
   * not.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param value The element to insert
   *
   * @return a pair consisting of an iterator to the element and a bool indicating whether the
   * insertion is successful or not.
   */
  template <typename Value>
  __device__ thrust::pair<iterator, bool> insert_and_find(Value const& value) noexcept
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.insert_and_find(value);
  }

  /**
   * @brief Inserts the given element into the map.
   *
   * @note This API returns a pair consisting of an iterator to the inserted element (or to the
   * element that prevented the insertion) and a `bool` denoting whether the insertion took place or
   * not.
   *
   * @tparam Value Input type which is convertible to 'value_type'
   *
   * @param group The Cooperative Group used to perform group insert_and_find
   * @param value The element to insert
   *
   * @return a pair consisting of an iterator to the element and a bool indicating whether the
   * insertion is successful or not.
   */
  template <typename Value>
  __device__ thrust::pair<iterator, bool> insert_and_find(
    cooperative_groups::thread_block_tile<cg_size> const& group, Value const& value) noexcept
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.insert_and_find(group, value);
  }
};

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<
  op::erase_tag,
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type = typename base_type::key_type;
  using value_type = typename base_type::value_type;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Erases an element.
   *
   * @tparam ProbeKey Input key type which is convertible to 'key_type'
   *
   * @param key The element to erase
   *
   * @return True if the given element is successfully erased
   */
  template <typename ProbeKey>
  __device__ bool erase(ProbeKey const& key) noexcept
  {
    ref_type& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.erase(key);
  }

  /**
   * @brief Erases an element.
   *
   * @tparam ProbeKey Input key type which is convertible to 'key_type'
   *
   * @param group The Cooperative Group used to perform group insert
   * @param key The element to erase
   *
   * @return True if the given element is successfully erased
   */
  template <typename ProbeKey>
  __device__ bool erase(cooperative_groups::thread_block_tile<cg_size> const& group,
                        ProbeKey const& key) noexcept
  {
    auto& ref_ = static_cast<ref_type&>(*this);
    return ref_.impl_.erase(group, key);
  }
};

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<
  op::contains_tag,
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type = typename base_type::key_type;
  using value_type = typename base_type::value_type;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Indicates whether the probe key `key` was inserted into the container.
   *
   * @note If the probe key `key` was inserted into the container, returns
   * true. Otherwise, returns false.
   *
   * @tparam ProbeKey Input key type which is convertible to 'key_type'
   *
   * @param key The key to search for
   *
   * @return A boolean indicating whether the probe key is present
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ bool contains(ProbeKey const& key) const noexcept
  {
    // CRTP: cast `this` to the actual ref type
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return ref_.impl_.contains(key);
  }

  /**
   * @brief Indicates whether the probe key `key` was inserted into the container.
   *
   * @note If the probe key `key` was inserted into the container, returns
   * true. Otherwise, returns false.
   *
   * @tparam ProbeKey Input key type which is convertible to 'key_type'
   *
   * @param group The Cooperative Group used to perform group contains
   * @param key The key to search for
   *
   * @return A boolean indicating whether the probe key is present
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ bool contains(
    cooperative_groups::thread_block_tile<cg_size> const& group, ProbeKey const& key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return ref_.impl_.contains(group, key);
  }
};

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<
  op::find_tag,
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type = typename base_type::key_type;
  using value_type     = typename base_type::value_type;
  using iterator       = typename base_type::iterator;
  using const_iterator = typename base_type::const_iterator;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Finds an element in the map with key equivalent to the probe key.
   *
   * @note Returns a un-incrementable input iterator to the element whose key is equivalent to
   * `key`. If no such element exists, returns `end()`.
   *
   * @tparam ProbeKey Input key type which is convertible to 'key_type'
   *
   * @param key The key to search for
   *
   * @return An iterator to the position at which the equivalent key is stored
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ const_iterator find(ProbeKey const& key) const noexcept
  {
    // CRTP: cast `this` to the actual ref type
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return ref_.impl_.find(key);
  }

  /**
   * @brief Finds an element in the map with key equivalent to the probe key.
   *
   * @note Returns a un-incrementable input iterator to the element whose key is equivalent to
   * `key`. If no such element exists, returns `end()`.
   *
   * @tparam ProbeKey Input key type which is convertible to 'key_type'
   *
   * @param group The Cooperative Group used to perform this operation
   * @param key The key to search for
   *
   * @return An iterator to the position at which the equivalent key is stored
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ const_iterator find(
    cooperative_groups::thread_block_tile<cg_size> const& group, ProbeKey const& key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    return ref_.impl_.find(group, key);
  }
};

template <typename Key,
          typename T,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Operators>
class operator_impl<
  op::for_each_tag,
  static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>> {
  using base_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type = static_map_ref<Key, T, Scope, KeyEqual, ProbingScheme, StorageRef, Operators...>;
  using key_type = typename base_type::key_type;
  using value_type     = typename base_type::value_type;
  using iterator       = typename base_type::iterator;
  using const_iterator = typename base_type::const_iterator;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Executes a callback on every element in the container with key equivalent to the probe
   * key.
   *
   * @note Passes a copy of the element whose `key` matches with a key from the input key sequence
   * to the callback.
   *
   * @tparam ProbeKey Input type which is convertible to 'key_type'
   * @tparam CallbackOp Unary callback functor or device lambda
   *
   * @param key The key to search for
   * @param callback_op Function to call on every element found
   */
  template <class ProbeKey, class CallbackOp>
  __device__ void for_each(ProbeKey const& key, CallbackOp&& callback_op) const noexcept
  {
    // CRTP: cast `this` to the actual ref type
    auto const& ref_ = static_cast<ref_type const&>(*this);
    ref_.impl_.for_each(key, std::forward<CallbackOp>(callback_op));
  }

  /**
   * @brief Executes a callback on every element in the container with key equivalent to the probe
   * key.
   *
   * @note Passes an un-incrementable input iterator to the element whose key is equivalent to
   * `key` to the callback.
   *
   * @note This function uses cooperative group semantics, meaning that any thread may call the
   * callback if it finds a matching element. If multiple elements are found within the same group,
   * each thread with a match will call the callback with its associated element.
   *
   * @note Synchronizing `group` within `callback_op` is undefined behavior.
   *
   * @tparam ProbeKey Input type which is convertible to 'key_type'
   * @tparam CallbackOp Unary callback functor or device lambda
   *
   * @param group The Cooperative Group used to perform this operation
   * @param key The key to search for
   * @param callback_op Function to call on every element found
   */
  template <class ProbeKey, class CallbackOp>
  __device__ void for_each(cooperative_groups::thread_block_tile<cg_size> const& group,
                           ProbeKey const& key,
                           CallbackOp&& callback_op) const noexcept
  {
    // CRTP: cast `this` to the actual ref type
    auto const& ref_ = static_cast<ref_type const&>(*this);
    ref_.impl_.for_each(group, key, std::forward<CallbackOp>(callback_op));
  }
};

}  // namespace detail
}  // namespace cuco
