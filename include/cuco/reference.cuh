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

namespace cuco {

/**
 * @brief Device reference of static_set.
 */
template <typename ProbingScheme, typename Key, typename KeyEqual>
class static_set_ref {
 public:
  using probing_scheme_type = ProbingScheme;                       ///< Type of probing scheme
  using key_type            = Key;                                 ///< Key Type
  using value_type          = typename ProbingScheme::value_type;  ///< Probing scheme element type
  using key_equal           = KeyEqual;  ///< Type of key equality binary callable
  using slot_view_type = typename probing_scheme_type::slot_view_type;  ///< Slot storage view type

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
   * @param slots Pointer to slot storage
   * @param capacity Number of slots in the storage
   * @param empty_key_sentienl Sentinel indicating empty key
   * @param predicate Key equality binary callable
   */
  static_set_ref(value_type* slots,
                 std::size_t const capacity,
                 Key empty_key_sentienl,
                 KeyEqual const& predicate)
    : probing_scheme_{slot_view_type{slots, capacity}},
      empty_key_sentienl_{empty_key_sentienl},
      predicate_{predicate}
  {
  }

  /**
   * @brief Inserts a key.
   *
   * @param key The key to insert
   * @return True if the given key is successfully inserted
   */
  __device__ inline bool insert(key_type const& key) noexcept
  {
    /*
    auto iter = probing_scheme_(key);
    while (true) {
      auto slot_keys = get_window(iter);

      auto is_available = [](slots_keys){
        for (k in slot_keys) {
          res = res or equality_wrapper(k, key);
        };
        return res;
      }();

      if (is_available) {
        switch (cas(slot_key, key, sentinel)) {
               case CONTINUE: continue;
               case SUCCESS: return true;
               case DUPLICATE: return false;
        }
      }
      iter++;
    }
    */
    return true;
  }

 private:
  probing_scheme_type probing_scheme_;  ///< Probing scheme
  key_type empty_key_sentienl_;         ///< Empty key sentinel
  key_equal predicate_;                 ///< Key equality binary callable
};
}  // namespace cuco
