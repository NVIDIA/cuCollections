/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <nvbench/nvbench.cuh>

#include <cstdint>

namespace cuco::benchmark {

namespace dist_type {
struct unique {
};

struct uniform {
  int64_t multiplicity;  // TODO assert >0
};

struct gaussian {
  double skew;  // TODO assert >0
};
}  // namespace dist_type

template <typename Dist>
auto dist_from_state(nvbench::state const& state)
{
  if constexpr (std::is_same_v<Dist, dist_type::unique>) {
    return Dist{};
  } else if constexpr (std::is_same_v<Dist, dist_type::uniform>) {
    auto const multiplicity = state.get_int64_or_default("Multiplicity", defaults::MULTIPLICITY);
    return Dist{multiplicity};
  } else if constexpr (std::is_same_v<Dist, dist_type::gaussian>) {
    auto const skew = state.get_float64_or_default("Skew", defaults::SKEW);
    return Dist{skew};
  } else {
    // TODO static assert fail
  }
}

}  // namespace cuco::benchmark

NVBENCH_DECLARE_TYPE_STRINGS(cuco::benchmark::dist_type::unique, "UNIQUE", "dist_type::unique");
NVBENCH_DECLARE_TYPE_STRINGS(cuco::benchmark::dist_type::uniform, "UNIFORM", "dist_type::uniform");
NVBENCH_DECLARE_TYPE_STRINGS(cuco::benchmark::dist_type::gaussian,
                             "GAUSSIAN",
                             "dist_type::gaussian");