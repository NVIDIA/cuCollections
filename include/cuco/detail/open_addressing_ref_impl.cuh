/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuco/sentinel.cuh>

#include <cuda/atomic>

namespace cuco {
namespace experimental {
namespace detail {

/**
 */
template <typename Key, cuda::thread_scope Scope, typename ProbingScheme, typename StorageRef>
class open_addressing_ref_impl {
  static_assert(sizeof(Key) <= 8, "Container does not support key types larger than 8 bytes.");

  static_assert(
    cuco::is_bitwise_comparable_v<Key>,
    "Key type must have unique object representations or have been explicitly declared as safe for "
    "bitwise comparison via specialization of cuco::is_bitwise_comparable_v<Key>.");

 public:
  using key_type            = Key;            ///< Key Type
  using probing_scheme_type = ProbingScheme;  ///< Type of probing scheme
  using storage_ref_type    = StorageRef;     ///< Type of storage ref

  /**
   * @brief Constructs open_addressing_ref_impl.
   *
   * @param empty_key_sentinel Sentinel indicating empty key
   * @param predicate Key equality binary callable
   * @param probing_scheme Probing scheme
   * @param storage_ref Non-owning ref of slot storage
   */
  __host__ __device__ explicit constexpr open_addressing_ref_impl(
    key_type empty_key_sentinel,
    probing_scheme_type const& probing_scheme,
    storage_ref_type storage_ref) noexcept
    : empty_key_sentinel_{empty_key_sentinel},
      probing_scheme_{probing_scheme},
      storage_ref_{storage_ref}
  {
  }

  /**
   * @brief Gets the maximum number of elements the container can hold.
   *
   * @return The maximum number of elements the container can hold
   */
  [[nodiscard]] __host__ __device__ constexpr auto capacity() const noexcept
  {
    return storage_ref_.capacity();
  }

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  [[nodiscard]] __host__ __device__ constexpr key_type empty_key_sentinel() const noexcept
  {
    return empty_key_sentinel_;
  }

  [[nodiscard]] __host__ __device__ constexpr probing_scheme_type probing_scheme() const noexcept
  {
    return probing_scheme_;
  }

  [[nodiscard]] __host__ __device__ constexpr storage_ref_type storage_ref() const noexcept
  {
    return storage_ref_;
  }

 private:
  key_type empty_key_sentinel_;         ///< Empty key sentinel
  probing_scheme_type probing_scheme_;  ///< Probing scheme
  storage_ref_type storage_ref_;        ///< Slot storage ref
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
