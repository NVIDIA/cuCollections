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

#include <cuco/static_set.cuh>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <vector>

namespace cuco::benchmark::perfect_static_set::defaults {

template <typename Key>
using HASHER = identityhash_32<Key>;  // set hashing scheme

auto const CGSize   = 4;
auto const CAPACITY = 4'294'967'296;

template <typename Key>
using PROBER = perfect_probing<CGSize, HASHER<Key>>;  // set probing scheme

}  // namespace cuco::benchmark::perfect_static_set::defaults
