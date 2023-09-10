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

#include <cuco/detail/trie/dynamic_bitset/dynamic_bitset.cuh>
#include <cuco/trie_ref.cuh>

namespace cuco {
namespace experimental {

/**
 * @brief Trie class
 *
 * @tparam label_type type of individual characters of vector keys (eg. char or int)
 * @tparam Allocator Type of allocator used for device storage
 */
template <typename LabelType, class Allocator = thrust::device_malloc_allocator<std::byte>>
class trie {
 public:
  /**
   * @brief Constructs an empty trie
   *
   * @param allocator Allocator used for allocating device storage
   */
  constexpr trie(Allocator const& allocator = Allocator{});
  ~trie() noexcept(false);

  /**
   * @brief Insert a single key into trie
   *
   * @param key Key to insert
   */
  void insert(const std::vector<LabelType>& key) noexcept;

  /**
   * @brief Build level-by-level trie indexes after inserting all keys
   *
   * In addition, a snapshot of current trie state is copied to device
   */
  void build() noexcept(false);

  /**
   * @brief For every pair (`offsets_begin[i]`, `offsets_begin[i + 1]`) in the range
   * `[offsets_begin, offsets_end)`, checks if the key defined by characters in the range
   * [`keys_begin[offsets_begin[i]]`, `keys_begin[offsets_begin[i + 1]]`) is present in trie.
   * Stores the index of key if it exists in trie (-1 otherwise) in `outputs_begin[i]`
   *
   * @tparam KeyIt Device-accessible iterator whose `value_type` can be converted to trie's
   * `LabelType`
   * @tparam OffsetIt Device-accessible iterator whose `value_type` can be converted to trie's
   * `size_type`
   * @tparam OutputIt Device-accessible iterator whose `value_type` can be constructed from boolean
   * type
   *
   * @param keys_begin Begin iterator to individual key characters
   * @param offsets_begin Begin iterator to offsets of key boundaries
   * @param offsets_end End iterator to offsets
   * @param outputs_begin Begin iterator to lookup results
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
  Allocator allocator_;              ///< Allocator
  size_type num_keys_;               ///< Number of keys inserted into trie
  size_type num_nodes_;              ///< Number of internal nodes
  std::vector<LabelType> last_key_;  ///< Last key inserted into trie

  static constexpr LabelType root_label_ = sizeof(LabelType) == 1 ? ' ' : -1;  ///< Sentinel value

  struct level;
  size_type num_levels_;       ///< Number of trie levels
  std::vector<level> levels_;  ///< Host-side array of levels
  level* d_levels_ptr_;        ///< Device-side array of levels

  using bitset_ref = detail::dynamic_bitset<>::ref_type;  ///< Read ref
  thrust::device_vector<bitset_ref> louds_refs_;          ///< refs to per-level louds bitsets
  thrust::device_vector<bitset_ref> outs_refs_;           ///< refs to per-level outs bitsets

  bitset_ref* louds_refs_ptr_;  ///< Raw pointer to d_louds_refs_
  bitset_ref* outs_refs_ptr_;   ///< Raw pointer to d_outs_refs_

  trie<LabelType>* device_ptr_;  ///< Device-side copy of trie

  template <typename... Operators>
  using ref_type =
    cuco::experimental::trie_ref<LabelType, Allocator, Operators...>;  ///< Non-owning container ref
                                                                       ///< type

  // Mixins need to be friends with this class in order to access private members
  template <typename Op, typename Ref>
  friend class detail::operator_impl;

  /**
   * @brief Struct to represent each trie level
   */
  struct level {
    level();
    level(level&&) = default;  ///< Move constructor

    detail::dynamic_bitset<> louds_;  ///< Indicates links to next and previous level
    detail::dynamic_bitset<> outs_;   ///< Indicates terminal nodes of valid keys

    thrust::device_vector<LabelType> labels_;  ///< Stores individual characters of keys
    LabelType* labels_ptr_;                    ///< Raw pointer to labels

    size_type offset_;  ///< Cumulative node count in parent levels
  };
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/trie/trie.inl>
