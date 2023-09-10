#pragma once

#include <cuco/operator.hpp>

namespace cuco {
namespace experimental {

template <typename LabelType>
class trie;

/**
 * @brief Device non-owning "ref" type that can be used in device code to perform arbitrary
 * operations defined in `include/cuco/operator.hpp`
 *
 * @tparam LabelType Trie label type
 * @tparam Operators Device operator options defined in `include/cuco/operator.hpp`
 */
template <typename LabelType, typename... Operators>
class trie_ref : public detail::operator_impl<Operators, trie_ref<LabelType, Operators...>>... {
 public:
  /**
   * @brief Constructs trie_ref.
   *
   * @param trie Non-owning ref of trie
   */
  __host__ __device__ explicit constexpr trie_ref(const trie<LabelType>* trie) noexcept;

 private:
  const trie<LabelType>* trie_;

  // Mixins need to be friends with this class in order to access private members
  template <typename Op, typename Ref>
  friend class detail::operator_impl;
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/trie/trie_ref.inl>
