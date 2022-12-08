/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuco/detail/probe_sequence_impl.cuh>

namespace cuco {

using hash_value_type = uint32_t;

/**
 * @brief A `MurmurHash3_32` hash function to hash the given argument on host and device.
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct MurmurHash3_32 : public detail::MurmurHash3_32<Key> {
};

}  // namespace cuco
