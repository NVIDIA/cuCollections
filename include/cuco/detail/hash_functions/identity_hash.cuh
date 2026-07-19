/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/functional>
#include <cuda/std/type_traits>

namespace cuco::detail {

/**
 * @brief An Identity hash function to hash the given argument on host and device
 *
 * @note `identity_hash` is perfect if `hash_table_capacity >= |input set|`
 *
 * @note `identity_hash` is only intended to be used perfectly.
 *
 * @note Perfect hashes are deterministic, and thus do not need seeds.
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct identity_hash : private cuda::std::identity {
  using argument_type = Key;  ///< The type of the values taken as argument
  /// The type of the hash values produced
  using result_type = cuda::std::conditional_t<sizeof(Key) <= 4, uint32_t, uint64_t>;

  static_assert(cuda::std::is_convertible_v<Key, result_type>,
                "Key type must be convertible to result_type");

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param x The input argument to hash
   * @return A resulting hash value for `x`
   */
  __host__ __device__ result_type operator()(Key const& x) const
  {
    return static_cast<result_type>(cuda::std::identity::operator()(x));
  }
};  // identity_hash

}  //  namespace cuco::detail
