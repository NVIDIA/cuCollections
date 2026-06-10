/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <cuco/constraints.cuh>
#include <cuco/detail/probing_scheme/probing_scheme_base.cuh>
#include <cuco/utility/traits.hpp>

#include <cuda/std/type_traits>

namespace cuco::detail {

/**
 * @brief Enforces compile-time constraints for open-addressing containers.
 *
 * @tparam Key Type used for keys. Requires `sizeof(Key) <= cuco::open_addressing_max_key_size` and
 * `cuco::is_bitwise_comparable_v<Key>`
 * @tparam Value Type used for storage values. Requires
 * `sizeof(Value) <= cuco::open_addressing_max_slot_size`
 * @tparam ProbingScheme Probing scheme (see `include/cuco/probing_scheme.cuh` for options)
 */
template <typename Key, typename Value, typename ProbingScheme>
struct open_addressing_compatible {
  /// Determines if the container is a key/value or key-only store
  static constexpr auto has_payload = not cuda::std::is_same_v<Key, Value>;

  static_assert(sizeof(Key) <= cuco::open_addressing_max_key_size,
                "Key size exceeds the maximum supported size (8 bytes, or 16 with sm_90+).");

  static_assert(sizeof(Value) <= cuco::open_addressing_max_slot_size,
                "Slot size exceeds the maximum supported size (16 bytes, or 32 with sm_90+).");

  static_assert(
    [] {
      if constexpr (has_payload) {
        constexpr auto payload_size = sizeof(typename Value::second_type);
#if defined(CUCO_HAS_128BIT_ATOMICS)
        return payload_size <= 16;
#else
        return payload_size <= 8;
#endif
      } else {
        return true;
      }
    }(),
    "Payload size exceeds the maximum supported size (8 bytes, or 16 with sm_90+).");

  static_assert(
    cuco::is_bitwise_comparable_v<Key>,
    "Key type must have unique object representations or have been explicitly declared as safe for "
    "bitwise comparison via specialization of cuco::is_bitwise_comparable_v<Key>.");

  static_assert(
    [] {
      if constexpr (has_payload) {
        return cuco::is_bitwise_comparable_v<typename Value::second_type>;
      } else {
        return true;
      }
    }(),
    "Payload type must have unique object representations or have been explicitly "
    "declared as safe for bitwise comparison via specialization of "
    "cuco::is_bitwise_comparable_v<T>.");

  static_assert(cuda::std::is_base_of_v<cuco::detail::probing_scheme_base<ProbingScheme::cg_size>,
                                        ProbingScheme>,
                "ProbingScheme must inherit from cuco::detail::probing_scheme_base");
};

}  // namespace cuco::detail
