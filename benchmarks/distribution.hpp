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

#include <cuco/detail/error.hpp>
#include <cuco/detail/utils.cuh>

#include <cstdint>

namespace cuco::benchmark::dist_type {

struct unique {
};

struct uniform : public cuco::detail::strong_type<int64_t> {
  uniform(int64_t multiplicity) : cuco::detail::strong_type<int64_t>{multiplicity}
  {
    CUCO_EXPECTS(multiplicity > 0, "Multiplicity must be greater than 0");
  }
};

struct gaussian : public cuco::detail::strong_type<double> {
  gaussian(double skew) : cuco::detail::strong_type<double>{skew}
  {
    CUCO_EXPECTS(skew > 0, "Skew must be greater than 0");
  }
};

}  // namespace cuco::benchmark::dist_type