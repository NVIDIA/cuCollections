namespace cuco {
namespace experimental {

template <typename LabelType, class Allocator, typename... Operators>
__host__ __device__ constexpr trie_ref<LabelType, Allocator, Operators...>::trie_ref(
  const trie<LabelType, Allocator>* trie) noexcept
  : trie_{trie}
{
}

namespace detail {

template <typename LabelType, class Allocator, typename... Operators>
class operator_impl<op::trie_lookup_tag, trie_ref<LabelType, Allocator, Operators...>> {
  using ref_type  = trie_ref<LabelType, Allocator, Operators...>;
  using size_type = size_t;

 public:
  /**
   * @brief Lookup a single key in trie
   *
   * @param key Iterator to first character of search key
   * @param length Number of characters in key
   *
   * @return Index of key if it exists in trie, -1 otherwise
   */
  template <typename KeyIt>
  [[nodiscard]] __device__ size_type lookup_key(KeyIt key, size_type length) const noexcept
  {
    auto const& trie = static_cast<ref_type const&>(*this).trie_;

    // Level-by-level search. node_id is updated at each level
    size_type node_id = 0;
    for (size_type cur_depth = 1; cur_depth <= length; cur_depth++) {
      if (!search_label_in_children(key[cur_depth - 1], node_id, cur_depth)) { return -1lu; }
    }

    // Check for terminal node bit that indicates a valid key
    size_type leaf_level_id = length;
    if (!trie->outs_refs_ptr_[leaf_level_id].test(node_id)) { return -1lu; }

    // Key exists in trie, generate the index
    auto offset = trie->d_levels_ptr_[leaf_level_id].offset_;
    auto rank   = trie->outs_refs_ptr_[leaf_level_id].rank(node_id);

    return offset + rank;
  }

 private:
  /**
   * @brief Find position of last child of a node
   *
   * @param louds louds bitvector of current level
   * @param node_id node index in current level
   *
   * @return Position of last child
   */
  template <typename BitVectorRef>
  [[nodiscard]] __device__ size_type get_last_child_position(BitVectorRef louds,
                                                             size_type& node_id) const noexcept
  {
    size_type node_pos = 0;
    if (node_id != 0) {
      node_pos = louds.select(node_id - 1) + 1;
      node_id  = node_pos - node_id;
    }

    auto pos_end = louds.find_next(node_pos);
    return node_id + (pos_end - node_pos);
  }

  /**
   * @brief Search for a target label in children nodes of a parent node
   *
   * @param target Label to search for
   * @param node_id Index of parent node
   * @param level_id Index of current level
   *
   * @return Boolean indicating success of search process
   */
  [[nodiscard]] __device__ bool search_label_in_children(LabelType target,
                                                         size_type& node_id,
                                                         size_type level_id) const noexcept
  {
    auto const& trie = static_cast<ref_type const&>(*this).trie_;
    auto louds       = trie->louds_refs_ptr_[level_id];

    auto end   = get_last_child_position(louds, node_id);  // Position of last child
    auto begin = node_id;  // Position of first child, initialized after find_last_child call

    auto& level = trie->d_levels_ptr_[level_id];
    auto labels = level.labels_ptr_;

    // Binary search labels array of current level
    while (begin < end) {
      node_id    = (begin + end) / 2;
      auto label = labels[node_id];
      if (target < label) {
        end = node_id;
      } else if (target > label) {
        begin = node_id + 1;
      } else {
        break;
      }
    }
    return begin < end;
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
