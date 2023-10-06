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

#include <cuco/static_set.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Rehash", "")
{
  constexpr std::size_t num_keys{400};

  cuco::experimental::static_set<int> set{num_keys, cuco::empty_key{-1}};

  thrust::device_vector<int> d_keys(num_keys);

  thrust::sequence(d_keys.begin(), d_keys.end());

  set.insert(d_keys.begin(), d_keys.end());

  set.rehash();
  REQUIRE(set.size() == num_keys);

  set.rehash(num_keys * 2);
  REQUIRE(set.size() == num_keys);

  // TODO erase num_erased keys
  // set.rehash()
  // REQUIRE(set.size() == num_keys - num_erased);
}
