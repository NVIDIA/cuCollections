/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <catch2/catch_test_macros.hpp>

#include <cuda/functional>

TEST_CASE("Size computation", "")
{
  constexpr std::size_t num_keys{400};

  cuco::static_map<int> map{
    cuco::extent<std::size_t>{400}, cuco::empty_key{-1}, cuco::empty_value{-1}};

  auto pairs_begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator<int>(0),
    cuda::proclaim_return_type<cuco::pair<int, int>>(
      [] __device__(auto i) { return cuco::pair<int, int>(i, i); }));

  auto const num_successes = map.insert(pairs_begin, pairs_begin + num_keys);

  auto const size = map.size();

  REQUIRE(size == num_keys);
  REQUIRE(num_successes == num_keys);
}
