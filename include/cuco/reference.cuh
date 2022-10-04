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

#include <cuco/detail/bitwise_compare.cuh>

#include <cuda/atomic>
#include <cuda/std/array>

namespace cuco {
namespace experimental {
namespace detail {
/**
 * @brief Enum of equality comparison results.
 */
enum class result : int32_t { UNEQUAL = 0, EMPTY = 1, EQUAL = 2 };

/**
 * @brief Equality wrapper.
 *
 * User-provided equality binary callable cannot be used to compared against sentinel value.
 *
 * @tparam T Right-hand side Element type
 * @tparam Equal Type of user-provided equality binary callable
 */
template <typename T, typename Equal>
struct equal_wrapper {
  T sentinel_;   ///< Sentinel value
  Equal equal_;  ///< Custom equality callable

  /**
   * @brief Equality wrapper ctor.
   *
   * @param sentinel Sentinel value
   * @param equal Equality binary callable
   */
  equal_wrapper(T const sentinel, Equal const& equal) : sentinel_{sentinel}, equal_{equal} {}

  /**
   * @brief Equality operator.
   *
   * @tparam U Left-hand side Element type
   *
   * @param lhs Left-hand side element to check equality
   * @param rhs Right-hand side element to check equality
   * @return Equality comparison result
   */
  template <typename U>
  __device__ inline result operator()(T const& lhs, U const& rhs) const noexcept
  {
    return cuco::detail::bitwise_compare(lhs, sentinel_)
             ? result::EMPTY
             : ((equal_(lhs, rhs)) ? result::EQUAL : result::UNEQUAL);
  }
};
}  // namespace detail

/**
 * @brief Device reference of static_set.
 */
template <typename Key,
          cuda::thread_scope Scope,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef>
class static_set_ref {
 public:
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

  /**
   * @brief Constructs static_set_ref.
   *
   * @param empty_key_sentienl Sentinel indicating empty key
   * @param predicate Key equality binary callable
   * @param probing_scheme Probing scheme
   * @param storage_ref Non-owning ref of slot storage
   */
  static_set_ref(Key empty_key_sentienl,
                 KeyEqual const& predicate,
                 ProbingScheme const& probing_scheme,
                 StorageRef storage_ref) noexcept
    : empty_key_sentienl_{empty_key_sentienl},
      predicate_{empty_key_sentienl_, predicate},
      probing_scheme_{probing_scheme},
      storage_ref_{storage_ref}
  {
  }

  /**
   * @brief Gets window array.
   *
   * @return Pointer to the first slot
   */
  __device__ inline window_type* windows() noexcept { return storage_ref_.windows(); }

  /**
   * @brief Gets window array.
   *
   * @return Pointer to the first slot
   */
  __device__ inline window_type const* windows() const noexcept { return storage_ref_.windows(); }

  /**
   * @brief Inserts a key.
   *
   * @param key The key to insert
   * @return True if the given key is successfully inserted
   */
  __device__ inline bool insert(value_type const& key) noexcept
  {
    auto probing_iter = probing_scheme_(key, storage_ref_.num_windows());

    while (true) {
      auto const window_slots = storage_ref_.window(*probing_iter);

      for (auto& slot_content : window_slots) {
        auto const eq_res = predicate_(slot_content, key);

        // If the key is already in the map, return false
        if (eq_res == detail::result::EQUAL) { return false; }
        if (eq_res == detail::result::EMPTY) {
          auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
          switch (attempt_insert((windows() + *probing_iter)->data() + intra_window_index, key)) {
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
   * @brief Inserts a key.
   *
   * @param g The Cooperative Group used to perform group insert
   * @param key The key to insert
   * @return True if the given key is successfully inserted
   */
  __device__ inline bool insert(cooperative_groups::thread_block_tile<cg_size> const& g,
                                value_type const& key) noexcept
  {
    auto probing_iter = probing_scheme_(g, key, storage_ref_.num_windows());

    while (true) {
      auto const window_slots = storage_ref_.window(*probing_iter);

      auto const [state, intra_window_index] = [&]() {
        for (auto i = 0; i < window_size; ++i) {
          switch (predicate_(window_slots[i], key)) {
            case detail::result::EMPTY:
              return cuco::pair<detail::result, int32_t>{detail::result::EMPTY, i};
            case detail::result::EQUAL:
              return cuco::pair<detail::result, int32_t>{detail::result::EQUAL, i};
            default: continue;
          }
        }
        return cuco::pair<detail::result, int32_t>{detail::result::UNEQUAL, -1};
      }();

      // If the key is already in the map, return false
      if (g.any(state == detail::result::EQUAL)) { return false; }

      auto const group_contains_empty = g.ballot(state == detail::result::EMPTY);

      if (group_contains_empty) {
        auto const src_lane = __ffs(group_contains_empty) - 1;
        auto const status =
          (g.thread_rank() == src_lane)
            ? attempt_insert((windows() + *probing_iter)->data() + intra_window_index, key)
            : insert_result::CONTINUE;

        switch (g.shfl(status, src_lane)) {
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
   * @brief Indicates whether the probe key `key` was inserted into the map.
   *
   * If the probe key `key` was inserted into the map, returns
   * true. Otherwise, returns false.
   *
   * @tparam ProbeKey Probe key type
   *
   * @param key The key to search for
   * @return A boolean indicating whether the probe key was inserted
   */
  template <typename ProbeKey>
  __device__ inline bool contains(ProbeKey const& key) const noexcept
  {
    auto probing_iter = probing_scheme_(key, storage_ref_.num_windows());

    while (true) {
      auto const window_slots = storage_ref_.window(*probing_iter);

      for (auto& slot_content : window_slots) {
        switch (predicate_(slot_content, key)) {
          case detail::result::UNEQUAL: continue;
          case detail::result::EMPTY: return false;
          case detail::result::EQUAL: return true;
        }
      }
      ++probing_iter;
    }
  }

  /**
   * @brief Indicates whether the probe key `key` was inserted into the map.
   *
   * If the probe key `key` was inserted into the map, returns
   * true. Otherwise, returns false.
   *
   * @tparam ProbeKey Probe key type
   *
   * @param g The Cooperative Group used to perform group contains
   * @param key The key to search for
   * @return A boolean indicating whether the probe key was inserted
   */
  template <typename ProbeKey>
  __device__ inline bool contains(cooperative_groups::thread_block_tile<cg_size> const& g,
                                  ProbeKey const& key) const noexcept
  {
    auto probing_iter = probing_scheme_(g, key, storage_ref_.num_windows());

    while (true) {
      auto const window_slots = storage_ref_.window(*probing_iter);

      auto const state = [&]() {
        for (auto& slot : window_slots) {
          switch (predicate_(slot, key)) {
            case detail::result::EMPTY: return detail::result::EMPTY;
            case detail::result::EQUAL: return detail::result::EQUAL;
            default: continue;
          }
        }
        return detail::result::UNEQUAL;
      }();

      if (g.any(state == detail::result::EQUAL)) { return true; }
      if (g.any(state == detail::result::EMPTY)) { return false; }

      ++probing_iter;
    }
  }

 private:
  enum class insert_result : int32_t { CONTINUE = 0, SUCCESS = 1, DUPLICATE = 2 };

  __device__ inline insert_result attempt_insert(value_type* slot, value_type const& key)
  {
    auto ref      = cuda::atomic_ref<value_type, Scope>{*slot};
    auto expected = empty_key_sentienl_;
    bool result   = ref.compare_exchange_strong(expected, key, cuda::std::memory_order_relaxed);
    if (result) {
      return insert_result::SUCCESS;
    } else {
      auto old = expected;
      return predicate_(old, key) == detail::result::EQUAL ? insert_result::DUPLICATE
                                                           : insert_result::CONTINUE;
    }
  }

 private:
  key_type empty_key_sentienl_;         ///< Empty key sentinel
  key_equal predicate_;                 ///< Key equality binary callable
  probing_scheme_type probing_scheme_;  ///< Probing scheme
  storage_ref_type storage_ref_;        ///< Slot storage ref
};
}  // namespace experimental
}  // namespace cuco
