/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <thrust/functional.h>

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
 * @tparam Key The type of the values to hash. Must have size <= uint64_t
 */
template <typename Key>
struct identity_hash : private thrust::identity<Key> {
  using argument_type = Key;       // The type of the values taken as argument
  using result_type   = uint64_t;  // The type of the hash values produced

  static_assert(sizeof(Key) <= sizeof(uint64_t),
                "Key type is too large to be reinterpreted as uint64_t in identity_hash");

  __host__ __device__ result_type operator()(const Key& x) const
  {
    return static_cast<result_type>(thrust::identity<Key>::operator()(x));
  }
};  // identity_hash

}  //  namespace cuco::detail
