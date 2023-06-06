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
#include <cuco/detail/pair.cuh>

#include <cuda/atomic>

#include <cooperative_groups.h>

namespace cuco {
namespace experimental {
namespace detail {

/**
 */
template <typename Key, cuda::thread_scope Scope, typename ProbingScheme, typename StorageRef>
class open_addressing_ref_impl {
  static_assert(sizeof(Key) <= 8, "Container does not support key types larger than 8 bytes.");

  static_assert(
    cuco::is_bitwise_comparable_v<Key>,
    "Key type must have unique object representations or have been explicitly declared as safe for "
    "bitwise comparison via specialization of cuco::is_bitwise_comparable_v<Key>.");

 public:
  using key_type            = Key;                                     ///< Key Type
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
   * @param empty_key_sentinel Sentinel indicating empty key
   * @param predicate Key equality binary callable
   * @param probing_scheme Probing scheme
   * @param storage_ref Non-owning ref of slot storage
   */
  __host__ __device__ explicit constexpr open_addressing_ref_impl(
    key_type empty_key_sentinel,
    probing_scheme_type const& probing_scheme,
    storage_ref_type storage_ref) noexcept
    : empty_key_sentinel_{empty_key_sentinel},
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
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] __host__ __device__ constexpr key_type empty_key_sentinel() const noexcept
  {
    return empty_key_sentinel_;
  }

  [[nodiscard]] __host__ __device__ constexpr probing_scheme_type probing_scheme() const noexcept
  {
    return probing_scheme_;
  }

  [[nodiscard]] __host__ __device__ constexpr storage_ref_type storage_ref() const noexcept
  {
    return storage_ref_;
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
   * @brief Indicates whether the probe key `key` was inserted into the container.
   *
   * If the probe key `key` was inserted into the container, returns
   * true. Otherwise, returns false.
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
    auto probing_iter = probing_scheme_(key, storage_ref_.num_windows());

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
   * If the probe key `key` was inserted into the container, returns
   * true. Otherwise, returns false.
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
    auto probing_iter = probing_scheme_(group, key, storage_ref_.num_windows());

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
    auto probing_iter = probing_scheme_(key, storage_ref_.num_windows());

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
    auto probing_iter = probing_scheme_(group, key, storage_ref_.num_windows());

    while (true) {
      auto const window_slots = storage_ref_[*probing_iter];

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (predicate(window_slots[i], key)) {
            case detail::equal_result::EMPTY: return cuco::pair{detail::equal_result::EMPTY, i};
            case detail::equal_result::EQUAL: return cuco::pair{detail::equal_result::EQUAL, i};
            default: continue;
          }
        }
        // returns dummy index `-1` for UNEQUAL
        return cuco::pair<detail::equal_result, int32_t>{detail::equal_result::UNEQUAL, -1};
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
  key_type empty_key_sentinel_;         ///< Empty key sentinel
  probing_scheme_type probing_scheme_;  ///< Probing scheme
  storage_ref_type storage_ref_;        ///< Slot storage ref
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
