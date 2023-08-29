/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuco/detail/trie/bit_vector/bit_vector.cuh>
#include <cuco/trie_ref.cuh>

namespace cuco {
namespace experimental {

/**
 * @brief Trie class
 *
 * @tparam label_type type of individual characters of vector keys (eg. char or int)
 */
template <typename label_type>
class trie {
 public:
  constexpr trie();
  ~trie() noexcept(false);

  /**
   * @brief Insert new key into trie
   *
   * @param key Key to insert
   */
  void insert(const std::vector<label_type>& key) noexcept;

  /**
   * @brief Build level-by-level trie indexes after inserting all keys
   *
   * In addition, a snapshot of current trie state is copied to device
   */
  void build() noexcept(false);

  /**
   * @brief Bulk lookup vector of keys
   *
   * @tparam KeyIt Device-accessible iterator to individual characters of keys
   * @tparam OffsetIt Device-accessible iterator to positions of key boundaries
   * @tparam OutputIt Device-accessible iterator to lookup result
   *
   * @param keys_begin Begin iterator to individual key characters
   * @param offsets_begin Begin iterator to offsets of key boundaries
   * @param offsets_end End iterator to offsets
   * @param outputs_begin Begin iterator to results
   * @param stream Stream to execute lookup kernel
   */
  template <typename KeyIt, typename OffsetIt, typename OutputIt>
  void lookup(KeyIt keys_begin,
              OffsetIt offsets_begin,
              OffsetIt offsets_end,
              OutputIt outputs_begin,
              cuda_stream_ref stream = {}) const noexcept;

  using size_type = std::size_t;  ///< size type

  /**
   * @brief Get current size i.e. number of keys inserted
   *
   * @return Number of keys
   */
  size_type constexpr size() const noexcept { return num_keys_; }

  /**
   * @brief Get device ref with operators.
   *
   * @tparam Operators Set of `cuco::op` to be provided by the ref
   *
   * @param ops List of operators, e.g., `cuco::bv_read`
   *
   * @return Device ref of the current `trie` object
   */
  template <typename... Operators>
  [[nodiscard]] auto ref(Operators... ops) const noexcept;

 private:
  size_type num_keys_;                ///< Number of keys inserted into trie
  size_type num_nodes_;               ///< Number of internal nodes
  std::vector<label_type> last_key_;  ///< Last key inserted into trie

  static constexpr label_type root_label_ = sizeof(label_type) == 1 ? ' ' : -1;  ///< Sentinel value

  struct level;
  size_type num_levels_;       ///< Number of trie levels
  std::vector<level> levels_;  ///< Host-side array of levels
  level* d_levels_ptr_;        ///< Device-side array of levels

  using bv_read_ref = bit_vector_ref<bit_vector::device_storage_ref, bv_read_tag>;  ///< Read ref
  thrust::device_vector<bv_read_ref> d_louds_refs_;  ///< refs to per-level louds bitvectors
  thrust::device_vector<bv_read_ref> d_outs_refs_;   ///< refs to per-level outs bitvectors

  bv_read_ref* d_louds_refs_ptr_;  ///< Raw pointer to d_louds_refs_
  bv_read_ref* d_outs_refs_ptr_;   ///< Raw pointer to d_outs_refs_

  trie<label_type>* device_ptr_;  ///< Device-side copy of trie

  template <typename... Operators>
  using ref_type =
    cuco::experimental::trie_ref<label_type, Operators...>;  ///< Non-owning container ref type

  // Mixins need to be friends with this class in order to access private members
  template <typename Op, typename Ref>
  friend class detail::operator_impl;

  /**
   * @brief Struct to represent each trie level
   */
  struct level {
    level();
    level(level&&) = default;  ///< Move constructor

    bit_vector louds_;  ///< Indicates links to next and previous level
    bit_vector outs_;   ///< Indicates terminal nodes of valid keys

    std::vector<label_type> labels_;              ///< Stores individual characters of keys
    thrust::device_vector<label_type> d_labels_;  ///< Device-side copy of `labels`
    label_type* d_labels_ptr_;                    ///< Raw pointer to d_labels

    size_type offset_;  ///< Cumulative node count in parent levels
  };
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/trie/trie.inl>
