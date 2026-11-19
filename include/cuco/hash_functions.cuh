/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/__config>

#include <cuda/functional>

namespace cuco {

/**
 * @brief Default 32-bit xxHash function, provided by CCCL.
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
using default_hash_function = cuda::hash<Key, cuda::hash_algorithm::xxhash_32>;

}  // namespace cuco
