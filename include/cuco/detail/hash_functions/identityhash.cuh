/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuco/detail/hash_functions/utils.cuh>

namespace cuco::detail {

/**
 * @brief An Identity hash function to hash the given argument on host and device.
 *
 * -----------------------------------------------------------------------------
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct IdentityHash {
  using argument_type = Key;  ///< The type of the values taken as argument
  using result_type   = Key;  ///< The type of the hash values produced

  /**
   * @brief Constructs a IdentityHash hash function.
   *
   * @note IdentityHash is perfect iff hash_table_capacity >= |input set|
   *
   * @note IdentityHash is only intended to be used perfectly.
   *
   * @note Perfect hashes are deterministic, and thus do not need seeds.
   */
  __host__ __device__ constexpr IdentityHash() : {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return The resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    return thrust::identity<Key>(key);
  }
};  // Identity hash

}  //  namespace cuco::detail
