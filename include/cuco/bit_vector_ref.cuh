#pragma once

#include <cuco/operator.hpp>

namespace cuco {
namespace experimental {

struct Rank;

template <typename StorageRef,
          typename... Operators>
class bit_vector_ref
  : public detail::operator_impl<
      Operators,
      bit_vector_ref<StorageRef, Operators...>>... {
 public:
  /**
   * @brief Constructs bit_vector_ref.
   *
   * @param storage_ref Non-owning ref of slot storage
   */
  __host__ __device__ explicit constexpr bit_vector_ref(
    uint64_t* words, Rank* ranks, uint32_t* selects, uint32_t num_selects) noexcept;

  /**
   * @brief Gets the maximum number of elements the container can hold.
   *
   * @return The maximum number of elements the container can hold
   */
  [[nodiscard]] __host__ __device__ constexpr auto capacity() const noexcept;

 private:
  uint64_t* words_;
  Rank* ranks_;
  uint32_t* selects_;
  uint32_t num_selects_;

  // Mixins need to be friends with this class in order to access private members
  template <typename Op, typename Ref>
  friend class detail::operator_impl;
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/bit_vector/bit_vector_ref.inl>
