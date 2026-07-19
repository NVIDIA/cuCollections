/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
