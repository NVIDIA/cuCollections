#pragma once

#include <cuco/operator.hpp>

namespace cuco {
namespace experimental {

struct Rank;

/**
 * @brief Device non-owning "ref" type that can be used in device code to perform arbitrary
 * operations defined in `include/cuco/operator.hpp`
 *
 * @tparam StorageRef Storage ref type
 * @tparam Operators Device operator options defined in `include/cuco/operator.hpp`
 */
template <typename StorageRef, typename... Operators>
class bit_vector_ref
  : public detail::operator_impl<Operators, bit_vector_ref<StorageRef, Operators...>>... {
 public:
  using storage_ref_type = StorageRef;  ///< Type of storage ref

  /**
   * @brief Constructs bit_vector_ref.
   *
   * @param words_ref Non-owning ref of words slot storage
   * @param ranks_ref Non-owning ref of ranks slot storage
   * @param selects_ref Non-owning ref of selects slot storage
   * @param ranks0_ref Non-owning ref of ranks0 slot storage
   * @param selects0_ref Non-owning ref of selects0 slot storage
   */
  __host__ __device__ explicit constexpr bit_vector_ref(storage_ref_type words_ref,
                                                        storage_ref_type ranks_ref,
                                                        storage_ref_type selects_ref,
                                                        storage_ref_type ranks0_ref,
                                                        storage_ref_type selects0_ref) noexcept;

 private:
  storage_ref_type words_ref_, ranks_ref_, selects_ref_, ranks0_ref_, selects0_ref_;

  // Mixins need to be friends with this class in order to access private members
  template <typename Op, typename Ref>
  friend class detail::operator_impl;
};

}  // namespace experimental
}  // namespace cuco
