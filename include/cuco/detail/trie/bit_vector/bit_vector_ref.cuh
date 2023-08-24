#pragma once

#include <cuco/operator.hpp>

namespace cuco {
namespace experimental {

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
   * @param storage Struct with non-owning refs to bitvector slot storages
   */
  __host__ __device__ explicit constexpr bit_vector_ref(storage_ref_type storage) noexcept;

 private:
  storage_ref_type storage_;

  // Mixins need to be friends with this class in order to access private members
  template <typename Op, typename Ref>
  friend class detail::operator_impl;
};

}  // namespace experimental
}  // namespace cuco
