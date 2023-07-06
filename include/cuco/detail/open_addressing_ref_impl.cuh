/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cuco/extent.cuh>
#include <cuco/pair.cuh>

#include <thrust/distance.h>
#include <thrust/pair.h>

#include <cuda/atomic>

#include <cooperative_groups.h>

#include <cstdint>
#include <type_traits>

namespace cuco {
namespace experimental {
namespace detail {

/**
 * @brief Common device non-owning "ref" implementation class.
 *
 * @note This class should NOT be used directly.
 *
 * @throw If the given key type doesn't have unique object representations, i.e.,
 * `cuco::bitwise_comparable_v<Key> == false`
 * @throw If the probing scheme type is not inherited from `cuco::detail::probing_scheme_base`
 *
 * @tparam Key Type used for keys. Requires `cuco::is_bitwise_comparable_v<Key>` returning true
 * @tparam Scope The scope in which operations will be performed by individual threads.
 * @tparam ProbingScheme Probing scheme (see `include/cuco/probing_scheme.cuh` for options)
 * @tparam StorageRef Storage ref type
 */
template <typename Key, cuda::thread_scope Scope, typename ProbingScheme, typename StorageRef>
class open_addressing_ref_impl {
  static_assert(
    cuco::is_bitwise_comparable_v<Key>,
    "Key type must have unique object representations or have been explicitly declared as safe for "
    "bitwise comparison via specialization of cuco::is_bitwise_comparable_v<Key>.");

  static_assert(
    std::is_base_of_v<cuco::experimental::detail::probing_scheme_base<ProbingScheme::cg_size>,
                      ProbingScheme>,
    "ProbingScheme must inherit from cuco::detail::probing_scheme_base");

  static_assert(is_window_extent_v<typename StorageRef::extent_type>,
                "Extent must be of type cuco::window_extent");

 public:
  using key_type            = Key;                                     ///< Key type
  using probing_scheme_type = ProbingScheme;                           ///< Type of probing scheme
  using storage_ref_type    = StorageRef;                              ///< Type of storage ref
  using window_type         = typename storage_ref_type::window_type;  ///< Window type
  using value_type          = typename storage_ref_type::value_type;   ///< Storage element type
  using extent_type         = typename storage_ref_type::extent_type;  ///< Extent type
  using size_type           = typename storage_ref_type::size_type;    ///< Probing scheme size type
  using iterator            = typename storage_ref_type::iterator;     ///< Slot iterator type
  using const_iterator = typename storage_ref_type::const_iterator;    ///< Const slot iterator type

  static constexpr auto cg_size = probing_scheme_type::cg_size;  ///< Cooperative group size
  static constexpr auto window_size =
    storage_ref_type::window_size;  ///< Number of elements handled per window

  /**
   * @brief Constructs open_addressing_ref_impl.
   *
   * @param empty_slot_sentinel Sentinel indicating an empty slot
   * @param probing_scheme Probing scheme
   * @param storage_ref Non-owning ref of slot storage
   */
  __host__ __device__ explicit constexpr open_addressing_ref_impl(
    value_type empty_slot_sentinel,
    probing_scheme_type const& probing_scheme,
    storage_ref_type storage_ref) noexcept
    : empty_slot_sentinel_{empty_slot_sentinel},
      probing_scheme_{probing_scheme},
      storage_ref_{storage_ref}
  {
  }

  /**
   * @brief Gets the maximum number of elements the container can hold.
   *
   * @return The maximum number of elements the container can hold
   */
  [[nodiscard]] __host__ __device__ constexpr auto capacity() const noexcept
  {
    return storage_ref_.capacity();
  }

  /**
   * @brief Returns a const_iterator to one past the last slot.
   *
   * @return A const_iterator to one past the last slot
   */
  [[nodiscard]] __host__ __device__ constexpr const_iterator end() const noexcept
  {
    return storage_ref_.end();
  }

  /**
   * @brief Returns an iterator to one past the last slot.
   *
   * @return An iterator to one past the last slot
   */
  [[nodiscard]] __host__ __device__ constexpr iterator end() noexcept { return storage_ref_.end(); }

  /**
   * @brief Inserts an element.
   *
   * @tparam Predicate Predicate type
   *
   * @param key Key of the element to insert
   * @param value The element to insert
   * @param predicate Predicate used to compare slot content against `key`
   *
   * @return True if the given element is successfully inserted
   */
  template <typename Predicate>
  __device__ bool insert(key_type const& key,
                         value_type const& value,
                         Predicate const& predicate) noexcept
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");
    auto probing_iter = probing_scheme_(key, storage_ref_.window_extent());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      for (auto& slot_content : window_slots) {
        auto const eq_res = predicate(slot_content, key);

        // If the key is already in the container, return false
        if (eq_res == detail::equal_result::EQUAL) { return false; }
        if (eq_res == detail::equal_result::EMPTY) {
          auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
          switch (attempt_insert(
            (storage_ref_.data() + *probing_iter)->data() + intra_window_index, value, predicate)) {
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
   * @tparam Predicate Predicate type
   *
   * @param group The Cooperative Group used to perform group insert
   * @param key Key of the element to insert
   * @param value The element to insert
   * @param predicate Predicate used to compare slot content against `key`
   *
   * @return True if the given element is successfully inserted
   */
  template <typename Predicate>
  __device__ bool insert(cooperative_groups::thread_block_tile<cg_size> const& group,
                         key_type const& key,
                         value_type const& value,
                         Predicate const& predicate) noexcept
  {
    auto probing_iter = probing_scheme_(group, key, storage_ref_.window_extent());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (predicate(window_slots[i], key)) {
            case detail::equal_result::EMPTY: return window_results{detail::equal_result::EMPTY, i};
            case detail::equal_result::EQUAL: return window_results{detail::equal_result::EQUAL, i};
            default: continue;
          }
        }
        // returns dummy index `-1` for UNEQUAL
        return window_results{detail::equal_result::UNEQUAL, -1};
      }();

      // If the key is already in the container, return false
      if (group.any(state == detail::equal_result::EQUAL)) { return false; }

      auto const group_contains_empty = group.ballot(state == detail::equal_result::EMPTY);

      if (group_contains_empty) {
        auto const src_lane = __ffs(group_contains_empty) - 1;
        auto const status =
          (group.thread_rank() == src_lane)
            ? attempt_insert((storage_ref_.data() + *probing_iter)->data() + intra_window_index,
                             value,
                             predicate)
            : insert_result::CONTINUE;

        switch (group.shfl(status, src_lane)) {
          case insert_result::SUCCESS: return true;
          case insert_result::DUPLICATE: return false;
          default: continue;
        }
      } else {
        ++probing_iter;
      }
    }
  }

  /**
   * @brief Inserts the given element into the container.
   *
   * @note This API returns a pair consisting of an iterator to the inserted element (or to the
   * element that prevented the insertion) and a `bool` denoting whether the insertion took place or
   * not.
   *
   * @tparam Predicate Predicate type
   *
   * @param key Key of the element to insert
   * @param value The element to insert
   * @param predicate Predicate used to compare slot content against `key`
   *
   * @return a pair consisting of an iterator to the element and a bool indicating whether the
   * insertion is successful or not.
   */
  template <typename Predicate>
  __device__ thrust::pair<iterator, bool> insert_and_find(key_type const& key,
                                                          value_type const& value,
                                                          Predicate const& predicate) noexcept
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");
    auto probing_iter = probing_scheme_(key, storage_ref_.window_extent());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      for (auto i = 0; i < window_size; ++i) {
        auto const eq_res = predicate(window_slots[i], key);
        auto* window_ptr  = (storage_ref_.data() + *probing_iter)->data();

        // If the key is already in the container, return false
        if (eq_res == detail::equal_result::EQUAL) { return {iterator{&window_ptr[i]}, false}; }
        if (eq_res == detail::equal_result::EMPTY) {
          switch (attempt_insert(window_ptr + i, value, predicate)) {
            case insert_result::SUCCESS: {
              return {iterator{&window_ptr[i]}, true};
            }
            case insert_result::DUPLICATE: {
              return {iterator{&window_ptr[i]}, false};
            }
            default: continue;
          }
        }
      }
      ++probing_iter;
    };
  }

  /**
   * @brief Inserts the given element into the container.
   *
   * @note This API returns a pair consisting of an iterator to the inserted element (or to the
   * element that prevented the insertion) and a `bool` denoting whether the insertion took place or
   * not.
   *
   * @tparam Predicate Predicate type
   *
   * @param group The Cooperative Group used to perform group insert_and_find
   * @param key Key of the element to insert
   * @param value The element to insert
   * @param predicate Predicate used to compare slot content against `key`
   *
   * @return a pair consisting of an iterator to the element and a bool indicating whether the
   * insertion is successful or not.
   */
  template <typename Predicate>
  __device__ thrust::pair<iterator, bool> insert_and_find(
    cooperative_groups::thread_block_tile<cg_size> const& group,
    key_type const& key,
    value_type const& value,
    Predicate const& predicate) noexcept
  {
    auto probing_iter = probing_scheme_(group, key, storage_ref_.window_extent());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (predicate(window_slots[i], key)) {
            case detail::equal_result::EMPTY: return window_results{detail::equal_result::EMPTY, i};
            case detail::equal_result::EQUAL: return window_results{detail::equal_result::EQUAL, i};
            default: continue;
          }
        }
        // returns dummy index `-1` for UNEQUAL
        return window_results{detail::equal_result::UNEQUAL, -1};
      }();

      auto* slot_ptr = (storage_ref_.data() + *probing_iter)->data() + intra_window_index;

      // If the key is already in the container, return false
      auto const group_finds_equal = group.ballot(state == detail::equal_result::EQUAL);
      if (group_finds_equal) {
        auto const src_lane = __ffs(group_finds_equal) - 1;
        auto const res      = group.shfl(reinterpret_cast<intptr_t>(slot_ptr), src_lane);
        return {iterator{reinterpret_cast<value_type*>(res)}, false};
      }

      auto const group_contains_empty = group.ballot(state == detail::equal_result::EMPTY);
      if (group_contains_empty) {
        auto const src_lane = __ffs(group_contains_empty) - 1;
        auto const res      = group.shfl(reinterpret_cast<intptr_t>(slot_ptr), src_lane);
        auto const status   = (group.thread_rank() == src_lane)
                                ? attempt_insert(slot_ptr, value, predicate)
                                : insert_result::CONTINUE;

        switch (group.shfl(status, src_lane)) {
          case insert_result::SUCCESS: {
            return {iterator{reinterpret_cast<value_type*>(res)}, true};
          }
          case insert_result::DUPLICATE: {
            return {iterator{reinterpret_cast<value_type*>(res)}, false};
          }
          default: continue;
        }
      } else {
        ++probing_iter;
      }
    }
  }

  /**
   * @brief Indicates whether the probe key `key` was inserted into the container.
   *
   * @note If the probe key `key` was inserted into the container, returns true. Otherwise, returns
   * false.
   *
   * @tparam ProbeKey Probe key type
   * @tparam Predicate Predicate type
   *
   * @param key The key to search for
   * @param predicate Predicate used to compare slot content against `key`
   *
   * @return A boolean indicating whether the probe key is present
   */
  template <typename ProbeKey, typename Predicate>
  [[nodiscard]] __device__ bool contains(ProbeKey const& key,
                                         Predicate const& predicate) const noexcept
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");
    auto probing_iter = probing_scheme_(key, storage_ref_.window_extent());

    while (true) {
      // TODO atomic_ref::load if insert operator is present
      auto const window_slots = storage_ref_[*probing_iter];

      for (auto& slot_content : window_slots) {
        switch (predicate(slot_content, key)) {
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
   * @note If the probe key `key` was inserted into the container, returns true. Otherwise, returns
   * false.
   *
   * @tparam ProbeKey Probe key type
   * @tparam Predicate Predicate type
   *
   * @param group The Cooperative Group used to perform group contains
   * @param key The key to search for
   * @param predicate Predicate used to compare slot content against `key`
   *
   * @return A boolean indicating whether the probe key is present
   */
  template <typename ProbeKey, typename Predicate>
  [[nodiscard]] __device__ bool contains(
    cooperative_groups::thread_block_tile<cg_size> const& group,
    ProbeKey const& key,
    Predicate const& predicate) const noexcept
  {
    auto probing_iter = probing_scheme_(group, key, storage_ref_.window_extent());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      auto const state = [&]() {
        for (auto& slot : window_slots) {
          switch (predicate(slot, key)) {
            case detail::equal_result::EMPTY: return detail::equal_result::EMPTY;
            case detail::equal_result::EQUAL: return detail::equal_result::EQUAL;
            default: continue;
          }
        }
        return detail::equal_result::UNEQUAL;
      }();

      if (group.any(state == detail::equal_result::EQUAL)) { return true; }
      if (group.any(state == detail::equal_result::EMPTY)) { return false; }

      ++probing_iter;
    }
  }

  /**
   * @brief Finds an element in the container with key equivalent to the probe key.
   *
   * @note Returns a un-incrementable input iterator to the element whose key is equivalent to
   * `key`. If no such element exists, returns `end()`.
   *
   * @tparam ProbeKey Probe key type
   * @tparam Predicate Predicate type
   *
   * @param key The key to search for
   * @param predicate Predicate used to compare slot content against `key`
   *
   * @return An iterator to the position at which the equivalent key is stored
   */
  template <typename ProbeKey, typename Predicate>
  [[nodiscard]] __device__ const_iterator find(ProbeKey const& key,
                                               Predicate const& predicate) const noexcept
  {
    static_assert(cg_size == 1, "Non-CG operation is incompatible with the current probing scheme");
    auto probing_iter = probing_scheme_(key, storage_ref_.window_extent());

    while (true) {
      // TODO atomic_ref::load if insert operator is present
      auto const window_slots = storage_ref_[*probing_iter];

      for (auto i = 0; i < window_size; ++i) {
        switch (predicate(window_slots[i], key)) {
          case detail::equal_result::EMPTY: {
            return this->end();
          }
          case detail::equal_result::EQUAL: {
            return const_iterator{&(*(storage_ref_.data() + *probing_iter))[i]};
          }
          default: continue;
        }
      }
      ++probing_iter;
    }
  }

  /**
   * @brief Finds an element in the container with key equivalent to the probe key.
   *
   * @note Returns a un-incrementable input iterator to the element whose key is equivalent to
   * `key`. If no such element exists, returns `end()`.
   *
   * @tparam ProbeKey Probe key type
   * @tparam Predicate Predicate type
   *
   * @param group The Cooperative Group used to perform this operation
   * @param key The key to search for
   * @param predicate Predicate used to compare slot content against `key`
   *
   * @return An iterator to the position at which the equivalent key is stored
   */
  template <typename ProbeKey, typename Predicate>
  [[nodiscard]] __device__ const_iterator
  find(cooperative_groups::thread_block_tile<cg_size> const& group,
       ProbeKey const& key,
       Predicate const& predicate) const noexcept
  {
    auto probing_iter = probing_scheme_(group, key, storage_ref_.window_extent());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (predicate(window_slots[i], key)) {
            case detail::equal_result::EMPTY: return window_results{detail::equal_result::EMPTY, i};
            case detail::equal_result::EQUAL: return window_results{detail::equal_result::EQUAL, i};
            default: continue;
          }
        }
        // returns dummy index `-1` for UNEQUAL
        return window_results{detail::equal_result::UNEQUAL, -1};
      }();

      // Find a match for the probe key, thus return an iterator to the entry
      auto const group_finds_match = group.ballot(state == detail::equal_result::EQUAL);
      if (group_finds_match) {
        auto const src_lane = __ffs(group_finds_match) - 1;
        auto const res      = group.shfl(
          reinterpret_cast<intptr_t>(&(*(storage_ref_.data() + *probing_iter))[intra_window_index]),
          src_lane);
        return const_iterator{reinterpret_cast<value_type*>(res)};
      }

      // Find an empty slot, meaning that the probe key isn't present in the container
      if (group.any(state == detail::equal_result::EMPTY)) { return this->end(); }

      ++probing_iter;
    }
  }

 private:
  /// Three-way insert result enum
  enum class insert_result : int32_t { CONTINUE = 0, SUCCESS = 1, DUPLICATE = 2 };

  /**
   * @brief Helper struct to store intermediate window probing results.
   */
  struct window_results {
    detail::equal_result state_;  ///< Equal result
    int32_t intra_window_index_;  ///< Intra-window index

    /**
     * @brief Constructs window_results.
     *
     * @param state The three way equality result
     *@param Intra-window index
     */
    __device__ explicit constexpr window_results(detail::equal_result state, int32_t index) noexcept
      : state_{state}, intra_window_index_{index}
    {
    }
  };

  /**
   * @brief Attempts to insert an element into a slot.
   *
   * @note Dispatches the correct implementation depending on the container
   * type and presence of other operator mixins.
   *
   * @tparam Predicate Predicate type
   *
   * @param slot Pointer to the slot in memory
   * @param value Element to insert
   * @param predicate Predicate used to compare slot content against `key`
   *
   * @return Result of this operation, i.e., success/continue/duplicate
   */
  template <typename Predicate>
  [[nodiscard]] __device__ insert_result attempt_insert(value_type* slot,
                                                        value_type const& value,
                                                        Predicate const& predicate)
  {
    // temporary workaround due to performance regression
    // https://github.com/NVIDIA/libcudacxx/issues/366
    auto old = [&]() {
      value_type expected = this->empty_slot_sentinel_;
      value_type val      = value;
      if constexpr (sizeof(value_type) == sizeof(unsigned int)) {
        auto* expected_ptr = reinterpret_cast<unsigned int*>(&expected);
        auto* value_ptr    = reinterpret_cast<unsigned int*>(&val);
        if constexpr (Scope == cuda::thread_scope_system) {
          return atomicCAS_system(reinterpret_cast<unsigned int*>(slot), *expected_ptr, *value_ptr);
        } else if constexpr (Scope == cuda::thread_scope_device) {
          return atomicCAS(reinterpret_cast<unsigned int*>(slot), *expected_ptr, *value_ptr);
        } else if constexpr (Scope == cuda::thread_scope_block) {
          return atomicCAS_block(reinterpret_cast<unsigned int*>(slot), *expected_ptr, *value_ptr);
        } else {
          static_assert(cuco::dependent_false<decltype(Scope)>, "Unsupported thread scope");
        }
      }
      if constexpr (sizeof(value_type) == sizeof(unsigned long long int)) {
        auto* expected_ptr = reinterpret_cast<unsigned long long int*>(&expected);
        auto* value_ptr    = reinterpret_cast<unsigned long long int*>(&val);
        if constexpr (Scope == cuda::thread_scope_system) {
          return atomicCAS_system(
            reinterpret_cast<unsigned long long int*>(slot), *expected_ptr, *value_ptr);
        } else if constexpr (Scope == cuda::thread_scope_device) {
          return atomicCAS(
            reinterpret_cast<unsigned long long int*>(slot), *expected_ptr, *value_ptr);
        } else if constexpr (Scope == cuda::thread_scope_block) {
          return atomicCAS_block(
            reinterpret_cast<unsigned long long int*>(slot), *expected_ptr, *value_ptr);
        } else {
          static_assert(cuco::dependent_false<decltype(Scope)>, "Unsupported thread scope");
        }
      }
    }();
    auto* old_ptr = reinterpret_cast<value_type*>(&old);
    if (*slot == *old_ptr) {
      // Shouldn't use `predicate` operator directly since it includes a redundant bitwise compare
      return predicate.equal_to(*old_ptr, value) == detail::equal_result::EQUAL
               ? insert_result::DUPLICATE
               : insert_result::CONTINUE;
    } else {
      return insert_result::SUCCESS;
    }
  }

  value_type empty_slot_sentinel_;      ///< Sentinel value indicating an empty slot
  probing_scheme_type probing_scheme_;  ///< Probing scheme
  storage_ref_type storage_ref_;        ///< Slot storage ref
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
