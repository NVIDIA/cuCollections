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

#include <key_generator.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>

auto constexpr DEFAULT_N             = 100'000'000;
auto constexpr DEFAULT_OCCUPANCY     = 0.5;
auto constexpr DEFAULT_MULTIPLICITY  = 1;
auto constexpr DEFAULT_MATCHING_RATE = 0.5;
auto constexpr DEFAULT_DISTRIBUTION  = dist_type::UNIFORM;

auto const DEFAULT_OCCUPANCY_RANGE     = nvbench::range(0.2, 0.8, 0.2);
auto const DEFAULT_MATCHING_RATE_RANGE = nvbench::range(0.5, 1., 0.5);

using KEY_LIST   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using VALUE_LIST = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using DISTRIBUTION_LIST =
  nvbench::enum_type_list<dist_type::GAUSSIAN, dist_type::GEOMETRIC, dist_type::UNIFORM>;
using MULTIPLICITY_LIST = nvbench::enum_type_list<1, 2, 4, 8, 16>;
