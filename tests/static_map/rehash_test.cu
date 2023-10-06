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

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Rehash", "")
{
  using key_type    = int;
  using mapped_type = long;
  constexpr std::size_t num_keys{400};

  cuco::experimental::static_map<key_type, mapped_type> map{
    num_keys, cuco::empty_key<key_type>{-1}, cuco::empty_value<mapped_type>{-1}};

  thrust::device_vector<key_type> d_keys(num_keys);
  thrust::device_vector<mapped_type> d_values(num_keys);

  thrust::sequence(d_keys.begin(), d_keys.end());
  thrust::sequence(d_values.begin(), d_values.end());

  auto pairs_begin =
    thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_values.begin()));

  map.insert(pairs_begin, pairs_begin + num_keys);

  map.rehash();
  REQUIRE(map.size() == num_keys);

  map.rehash(num_keys * 2);
  REQUIRE(map.size() == num_keys);

  // TODO erase num_erased keys
  // map.rehash()
  // REQUIRE(map.size() == num_keys - num_erased);
}
