/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include <test_utils.hpp>

#include <cuco/bloom_filter.cuh>

#include <cuda/functional>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>

using size_type = int32_t;

template <typename Filter>
void test_variable_cg_size(Filter& filter, size_type num_keys)
{
  using Key = typename Filter::key_type;

  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end());

  thrust::device_vector<bool> contained(num_keys, false);

  filter.add(keys.begin(), keys.end());
  filter.contains(keys.begin(), keys.end(), contained.begin());
  REQUIRE(cuco::test::all_of(contained.begin(), contained.end(), cuda::std::identity{}));
}

// Exercises a matrix of (AddHorizontalLayout, ContainsHorizontalLayout) values to verify the
// parametric policy compiles and works across varied CG-size combinations.
TEMPLATE_TEST_CASE_SIG(
  "bloom_filter variable CG size tests",
  "",
  ((class Key, class Policy), Key, Policy),
  (int32_t, cuco::parametric_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 8, 8, 1, 8, 8, 1>),
  (int32_t, cuco::parametric_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 8, 8, 8, 1, 1, 8>),
  (int32_t, cuco::parametric_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 8, 8, 4, 2, 2, 4>),
  (int32_t, cuco::parametric_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 8, 8, 2, 4, 4, 2>))
{
  using filter_type =
    cuco::bloom_filter<Key, cuco::extent<size_t>, cuda::thread_scope_device, Policy>;
  constexpr size_type num_keys{400};

  auto filter = filter_type{1000};
  test_variable_cg_size(filter, num_keys);
}
