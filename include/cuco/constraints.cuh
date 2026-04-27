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

#include <cuco/detail/__config>

#include <cstddef>

namespace cuco {

/// Maximum supported key size (in bytes) for open-addressing containers.
inline constexpr std::size_t open_addressing_max_key_size =
#if defined(CUCO_HAS_128BIT_ATOMICS)
  16;
#else
  8;
#endif

/// Maximum supported payload/mapped type size (in bytes) for open-addressing containers.
/// Tied to `open_addressing_max_key_size`: a slot stores at most a key plus an equally-sized
/// payload.
inline constexpr std::size_t open_addressing_max_payload_size = open_addressing_max_key_size;

/// Maximum supported slot size (in bytes) for open-addressing containers.
/// Tied to `open_addressing_max_key_size`: a slot stores at most a key plus an equally-sized
/// payload.
inline constexpr std::size_t open_addressing_max_slot_size =
  open_addressing_max_key_size + open_addressing_max_payload_size;

}  // namespace cuco
