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
          typename StorageView>
class static_set_ref {
 public:
  using key_type            = Key;                            ///< Key Type
  using probing_scheme_type = ProbingScheme;                  ///< Type of probing scheme
  using storage_view_type   = StorageView;                    ///< Type of slot storage view
  using value_type = typename storage_view_type::value_type;  ///< Probing scheme element type
  using size_type  = typename storage_view_type::size_type;   ///< Probing scheme size type
  using key_equal =
    detail::equal_wrapper<value_type, KeyEqual>;  ///< Type of key equality binary callable

  /// CG size
  static constexpr int cg_size = probing_scheme_type::cg_size;
  /// Number of elements handled per window
  static constexpr int window_size = probing_scheme_type::window_size;
  /// Whether window probing is used
  static constexpr enable_window_probing uses_window_probing =
    probing_scheme_type::uses_window_probing;

  /**
   * @brief Constructs static_set_ref.
   *
   * @param empty_key_sentienl Sentinel indicating empty key
   * @param predicate Key equality binary callable
   * @param probing_scheme Probing scheme
   * @param slot_view View of slot storage
   */
  static_set_ref(Key empty_key_sentienl,
                 KeyEqual const& predicate,
                 ProbingScheme const& probing_scheme,
                 StorageView slot_view) noexcept
    : empty_key_sentienl_{empty_key_sentienl},
      predicate_{empty_key_sentienl_, predicate},
      probing_scheme_{probing_scheme},
      slot_view_{slot_view}
  {
  }

  /**
   * @brief Gets slots array.
   *
   * @return Pointer to the first slot
   */
  __device__ inline value_type* slots() noexcept { return slot_view_.slots(); }

  /**
   * @brief Gets slots array.
   *
   * @return Pointer to the first slot
   */
  __device__ inline value_type const* slots() const noexcept { return slot_view_.slots(); }

  /**
   * @brief Inserts a key.
   *
   * @param key The key to insert
   * @return True if the given key is successfully inserted
   */
  __device__ inline bool insert(value_type const& key) noexcept
  {
    auto probing_iter = probing_scheme_(key, slot_view_.capacity());

    while (true) {
      auto window_slots = window(*probing_iter);

      for (auto& slot_content : window_slots) {
        auto const eq_res = predicate_(slot_content, key);

        // If the key is already in the map, return false
        if (eq_res == detail::result::EQUAL) { return false; }
        if (eq_res == detail::result::EMPTY) {
          auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
          auto const idx                = *probing_iter + intra_window_index;
          switch (attempt_insert(slots() + idx, key)) {
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
    auto probing_iter = probing_scheme_(g, key, slot_view_.capacity());

    while (true) {
      auto window_slots = window(*probing_iter);

      detail::result eq_result{detail::result::UNEQUAL};
      auto empty_idx = 0;
      // Loop over the all window slots find the index of the first empty slot
      for (auto i = 0; i < window_size; ++i) {
        auto res = predicate_(window_slots[i], key);
        if (res == detail::result::EMPTY) { empty_idx = min(empty_idx, i); }
        eq_result = max(eq_result, res);
      }

      // If the key is already in the map, return false
      if (g.any(eq_result == detail::result::EQUAL)) { return false; }

      auto const group_contains_empty = g.ballot(eq_result == detail::result::EMPTY);

      if (group_contains_empty) {
        auto const src_lane = __ffs(group_contains_empty) - 1;
        auto status         = insert_result::CONTINUE;
        if (g.thread_rank() == src_lane) {
          auto insert_location = *probing_iter + empty_idx;
          status               = attempt_insert(slots() + insert_location, key);
        }
        status = g.shfl(status, src_lane);

        // successful insert
        if (status == insert_result::SUCCESS) { return true; }
        // duplicate present during insert
        if (status == insert_result::DUPLICATE) { return false; }
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
    auto probing_iter = probing_scheme_(key, slot_view_.capacity());

    while (true) {
      auto window_slots = window(*probing_iter);

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

  /**
   * @brief Returns an array of elements (window) for a given index.
   *
   * @param window_index Index of the first element of the window
   * @return An array of elements
   */
  __device__ cuda::std::array<value_type, window_size> window(size_type window_index) const noexcept
  {
    cuda::std::array<value_type, window_size> slot_array;
    memcpy(&slot_array[0], slot_view_.slots() + window_index, window_size * sizeof(value_type));
    return slot_array;
  }

 private:
  key_type empty_key_sentienl_;         ///< Empty key sentinel
  key_equal predicate_;                 ///< Key equality binary callable
  probing_scheme_type probing_scheme_;  ///< Probing scheme
  storage_view_type slot_view_;         ///< View of slot storage
};
}  // namespace experimental
}  // namespace cuco
