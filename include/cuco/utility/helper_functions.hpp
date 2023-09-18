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
#include <cuco/extent.cuh>

namespace cuco {

/// The default occupancy percentage: 0.5 implies a 50% occupancy
constexpr double DEFAULT_HASH_TABLE_OCCUPANCY = 0.5;

/**
 * @brief Computes requisite size of hash table
 *
 * Computes the number of entries required in a hash table to satisfy
 * inserting a specified number of keys to achieve the specified hash table
 * occupancy.
 *
 * @note The output of this helper function for a static extent is not a build-time constant.
 *
 * @throw If the desired occupancy is no bigger than zero
 * @throw If the desired occupancy is larger than one
 *
 * @param size The number of elements that will be inserted
 * @param desired_occupancy The desired occupancy percentage, e.g., 0.5 implies a
 * 50% occupancy
 *
 * @return The size of the hash table that will satisfy the desired occupancy for the specified
 * input size
 */
constexpr std::size_t compute_hash_table_size(
  std::size_t size, double desired_occupancy = DEFAULT_HASH_TABLE_OCCUPANCY)
{
  CUCO_EXPECTS(desired_occupancy > 0., "Desired occupancy must be larger than zero");
  CUCO_EXPECTS(desired_occupancy <= 1., "Desired occupancy cannot be larger than one");

  // Calculate size of hash map based on the desired occupancy
  return static_cast<std::size_t>(static_cast<double>(size) / desired_occupancy);
}

}  // namespace cuco
