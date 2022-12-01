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

#include <cuco/detail/equal_wrapper.cuh>
#include <cuco/detail/pair.cuh>
#include <cuco/function.hpp>
#include <cuco/sentinel.cuh>  // TODO .hpp

#include <cooperative_groups.h>

#include <cuda/std/atomic>

namespace cuco {
namespace experimental {
namespace detail {

/**
 * @brief Device reference base class for open addressing-based containers.
 *
 * @note Cannot be instantiated directly.
 */
template <typename Reference,
          typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Functions>
class open_address_container_ref {
 public:
  using ref_type            = Reference;                       ///< Type of derived reference type
  using key_type            = Key;                             ///< Key Type
  using probing_scheme_type = ProbingScheme;                   ///< Type of probing scheme
  using storage_ref_type    = StorageRef;                      ///< Type of slot storage ref
  using window_type = typename storage_ref_type::window_type;  ///< Probing scheme element type
  using value_type  = typename storage_ref_type::value_type;   ///< Probing scheme element type
  using size_type   = typename storage_ref_type::size_type;    ///< Probing scheme size type
  using key_equal =
    detail::equal_wrapper<value_type, KeyEqual>;  ///< Type of key equality binary callable

  /// CG size
  static constexpr int cg_size = probing_scheme_type::cg_size;
  /// Number of elements handled per window
  static constexpr int window_size = storage_ref_type::window_size;

  static constexpr cuda::thread_scope scope = Scope;  ///< Thread scope

  // TODO __device__ inline void clear() noexcept { storage_ref_.clear(); }

  /**
   * @brief Gets the maximum number of elements the hash map can hold.
   *
   * @return The maximum number of elements the hash map can hold
   */
  [[nodiscard]] __host__ __device__ inline size_type capacity() const noexcept
  {
    return storage_ref_.capacity();
  }

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] __host__ __device__ inline key_type empty_key_sentinel() const noexcept
  {
    return empty_key_sentinel_;
  }

 protected:
  /**
   * @brief Constructs open_address_container_ref.
   *
   * @param empty_key_sentinel Sentinel indicating empty key
   * @param predicate Key equality binary callable
   * @param probing_scheme Probing scheme
   * @param storage_ref Non-owning ref of slot storage
   */
  __host__ __device__ open_address_container_ref(cuco::sentinel::empty_key<Key> empty_key_sentinel,
                                                 KeyEqual const& predicate,
                                                 ProbingScheme const& probing_scheme,
                                                 StorageRef storage_ref) noexcept
    : empty_key_sentinel_{empty_key_sentinel},
      predicate_{empty_key_sentinel_.value, predicate},
      probing_scheme_{probing_scheme},
      storage_ref_{storage_ref}
  {
  }

  /**
   * @brief Extracts the key field from the container's `value_type`.
   *
   * @param value The value the key should be extracted from
   *
   * @return The key
   */
  __device__ inline constexpr auto extract_key(value_type const& value) const noexcept
  {
    if constexpr (cuco::detail::is_std_pair_like<value_type>::value) { return std::get<0>(value); }
    if constexpr (cuco::detail::is_thrust_pair_like<value_type>::value) {
      return thrust::get<0>(value);  // TODO raw_reference_cast?
    }
    if constexpr (not(cuco::detail::is_std_pair_like<value_type>::value or
                      cuco::detail::is_thrust_pair_like<value_type>::value)) {
      return value;
    }
  }

  cuco::sentinel::empty_key<key_type> empty_key_sentinel_;  ///< Empty key sentinel
  key_equal predicate_;                                     ///< Key equality binary callable
  probing_scheme_type probing_scheme_;                      ///< Probing scheme
  storage_ref_type storage_ref_;                            ///< Slot storage ref
};

template <typename Reference,
          typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Functions>
class function_impl<function::insert,
                    open_address_container_ref<Reference,
                                               Key,
                                               Scope,
                                               KeyEqual,
                                               ProbingScheme,
                                               StorageRef,
                                               Functions...>> {
  using base_type =
    open_address_container_ref<Reference, Key, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type   = Reference;
  using key_type   = typename base_type::key_type;
  using value_type = typename base_type::value_type;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Inserts an element.
   *
   * @param value The element to insert
   * @return True if the given element is successfully inserted
   */
  __device__ inline bool insert(value_type const& value) noexcept
  {
    ref_type& ref_    = static_cast<ref_type&>(*this);
    auto const key    = ref_.extract_key(value);
    auto probing_iter = ref_.probing_scheme_(key, ref_.storage_ref_.num_windows());

    while (true) {
      auto const window_slots = ref_.storage_ref_.window(*probing_iter);

      for (auto& slot_content : window_slots) {
        auto const eq_res = ref_.predicate_(ref_.extract_key(slot_content), key);

        // If the key is already in the map, return false
        // TODO this needs to be disabled for static_multimap
        if (eq_res == detail::equal_result::EQUAL) { return false; }
        if (eq_res == detail::equal_result::EMPTY) {
          auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
          switch (attempt_insert(
            (ref_.storage_ref_.windows() + *probing_iter)->data() + intra_window_index, value)) {
            case insert_result::CONTINUE: continue;
            case insert_result::SUCCESS: return true;
            case insert_result::DUPLICATE: return false;
          }
        }
      }
      ++probing_iter;
    }
  }

  /**
   * @brief Inserts an element.
   *
   * @param g The Cooperative Group used to perform group insert
   * @param value The element to insert
   * @return True if the given element is successfully inserted
   */
  __device__ inline bool insert(cooperative_groups::thread_block_tile<cg_size> group,
                                value_type const& value) noexcept
  {
    auto& ref_        = static_cast<ref_type&>(*this);
    auto const key    = ref_.extract_key(value);
    auto probing_iter = ref_.probing_scheme_(group, key, ref_.storage_ref_.num_windows());

    while (true) {
      auto const window_slots = ref_.storage_ref_.window(*probing_iter);

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (ref_.predicate_(ref_.extract_key(window_slots[i]), key)) {
            case detail::equal_result::EMPTY:
              return cuco::pair<detail::equal_result, int32_t>{detail::equal_result::EMPTY, i};
            case detail::equal_result::EQUAL:
              return cuco::pair<detail::equal_result, int32_t>{detail::equal_result::EQUAL, i};
            default: continue;
          }
        }
        return cuco::pair<detail::equal_result, int32_t>{detail::equal_result::UNEQUAL, -1};
      }();

      // If the key is already in the map, return false
      // TODO this needs to be disabled for static_multimap
      if (group.any(state == detail::equal_result::EQUAL)) { return false; }

      auto const group_contains_empty = group.ballot(state == detail::equal_result::EMPTY);

      if (group_contains_empty) {
        auto const src_lane = __ffs(group_contains_empty) - 1;
        auto const status =
          (group.thread_rank() == src_lane)
            ? attempt_insert(
                (ref_.storage_ref_.windows() + *probing_iter)->data() + intra_window_index, value)
            : insert_result::CONTINUE;

        switch (group.shfl(status, src_lane)) {
          case insert_result::SUCCESS: return true;
          case insert_result::DUPLICATE:
            return false;  // TODO this needs to be disabled for static_multimap
          default: continue;
        }
      } else {
        ++probing_iter;
      }
    }
  }

 private:
  enum class insert_result : int32_t { CONTINUE = 0, SUCCESS = 1, DUPLICATE = 2 };

  /**
   * @brief Attempts to insert an element into a slot.
   *
   * @note Dispatches the correct implementation depending on the container
   * type and presence of other function mixins.
   *
   * @param slot Pointer to the slot in memory
   * @param value Element to insert
   *
   * @return Result of this operation, i.e., success/continue/duplicate
   */
  __device__ inline insert_result attempt_insert(value_type* slot, value_type const& value)
  {
    // code path for static_set -> cas
    if constexpr (std::is_same_v<key_type, value_type>) { return cas(slot, value); }

    // TODO code path for static_map and static_multimap
  }

  /**
   * @brief Try insert using simple atomic compare-and-swap.
   *
   * @param slot Pointer to the slot in memory
   * @param value Element to insert
   *
   * @return Result of this operation, i.e., success/continue/duplicate
   */
  __device__ inline insert_result cas(value_type* slot, value_type const& value)
  {
    auto& ref_ = static_cast<ref_type&>(*this);

    auto ref      = cuda::atomic_ref<value_type, Scope>{*slot};
    auto expected = ref_.empty_key_sentinel_.value;
    bool result   = ref.compare_exchange_strong(expected, value, cuda::std::memory_order_relaxed);
    if (result) {
      return insert_result::SUCCESS;
    } else {
      auto old = expected;
      return ref_.predicate_(old, value) == detail::equal_result::EQUAL ? insert_result::DUPLICATE
                                                                        : insert_result::CONTINUE;
    }
  }

  // TODO packed_cas, back_to_back_cas, cas_dependent_write
};

template <typename Reference,
          typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef,
          typename... Functions>
class function_impl<function::contains,
                    open_address_container_ref<Reference,
                                               Key,
                                               Scope,
                                               KeyEqual,
                                               ProbingScheme,
                                               StorageRef,
                                               Functions...>> {
  using base_type =
    open_address_container_ref<Reference, Key, Scope, KeyEqual, ProbingScheme, StorageRef>;
  using ref_type   = Reference;
  using key_type   = typename base_type::key_type;
  using value_type = typename base_type::value_type;

  static constexpr auto cg_size     = base_type::cg_size;
  static constexpr auto window_size = base_type::window_size;

 public:
  /**
   * @brief Indicates whether the probe key `key` was inserted into the container.
   *
   * If the probe key `key` was inserted into the container, returns
   * true. Otherwise, returns false.
   *
   * @tparam ProbeKey Probe key type
   *
   * @param key The key to search for
   * @return A boolean indicating whether the probe key is present
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ inline bool contains(ProbeKey const& key) const noexcept
  {
    // CRTP: cast `this` to the actual reference type
    auto const& ref_ = static_cast<ref_type const&>(*this);

    auto probing_iter = ref_.probing_scheme_(key, ref_.storage_ref_.num_windows());

    while (true) {
      // TODO do we need to use atomic_ref::load here?
      auto const window_slots = ref_.storage_ref_.window(*probing_iter);

      for (auto& slot_content : window_slots) {
        switch (ref_.predicate_(ref_.extract_key(slot_content), key)) {
          case detail::equal_result::UNEQUAL: continue;
          case detail::equal_result::EMPTY: return false;
          case detail::equal_result::EQUAL: return true;
        }
      }
      ++probing_iter;
    }
  }

  /**
   * @brief Indicates whether the probe key `key` was inserted into the container.
   *
   * If the probe key `key` was inserted into the container, returns
   * true. Otherwise, returns false.
   *
   * @tparam ProbeKey Probe key type
   *
   * @param g The Cooperative Group used to perform group contains
   * @param key The key to search for
   * @return A boolean indicating whether the probe key is present
   */
  template <typename ProbeKey>
  [[nodiscard]] __device__ inline bool contains(
    cooperative_groups::thread_block_tile<cg_size> const& g, ProbeKey const& key) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);

    auto probing_iter = ref_.probing_scheme_(g, key, ref_.storage_ref_.num_windows());

    while (true) {
      auto const window_slots = ref_.storage_ref_.window(*probing_iter);

      auto const state = [&]() {
        for (auto& slot : window_slots) {
          switch (ref_.predicate_(ref_.extract_key(slot), key)) {
            case detail::equal_result::EMPTY: return detail::equal_result::EMPTY;
            case detail::equal_result::EQUAL: return detail::equal_result::EQUAL;
            default: continue;
          }
        }
        return detail::equal_result::UNEQUAL;
      }();

      if (g.any(state == detail::equal_result::EQUAL)) { return true; }
      if (g.any(state == detail::equal_result::EMPTY)) { return false; }

      ++probing_iter;
    }
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco