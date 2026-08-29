/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/bloom_filter.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <exception>

using size_type = int32_t;

template <typename Filter>
void test_unique_sequence(Filter& filter, size_type num_keys)
{
  using Key = typename Filter::key_type;

  // Generate keys
  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end());

  thrust::device_vector<bool> contained(num_keys, false);

  auto is_even =
    cuda::proclaim_return_type<bool>([] __device__(auto const& i) { return i % 2 == 0; });

  SECTION("Non-inserted keys should not be contained.")
  {
    filter.contains(keys.begin(), keys.end(), contained.begin());
    REQUIRE(cuco::test::none_of(contained.begin(), contained.end(), cuda::std::identity{}));
  }

  SECTION("All inserted keys should be contained.")
  {
    filter.add(keys.begin(), keys.end());
    filter.contains(keys.begin(), keys.end(), contained.begin());
    REQUIRE(cuco::test::all_of(contained.begin(), contained.end(), cuda::std::identity{}));
  }

  SECTION("After clearing the filter no keys should be contained.")
  {
    filter.clear();
    filter.contains(keys.begin(), keys.end(), contained.begin());
    REQUIRE(cuco::test::none_of(contained.begin(), contained.end(), cuda::std::identity{}));
  }

  SECTION("All conditionally inserted keys should be contained")
  {
    filter.add_if(keys.begin(), keys.end(), cuda::counting_iterator<std::size_t>(0), is_even);
    filter.contains_if(keys.begin(),
                       keys.end(),
                       cuda::counting_iterator<std::size_t>(0),
                       is_even,
                       contained.begin());
    REQUIRE(cuco::test::equal(
      contained.begin(),
      contained.end(),
      cuda::counting_iterator<std::size_t>(0),
      cuda::proclaim_return_type<bool>([] __device__(auto const& idx_contained, auto const& idx) {
        return ((idx % 2) == 0) == idx_contained;
      })));
  }

  // TODO test FPR but how?
}

TEMPLATE_TEST_CASE_SIG(
  "bloom_filter policy tests",
  "",
  ((class Key, class Policy), Key, Policy),
  (int32_t, cuco::bloom_filter_policy<int32_t>),
  (int32_t, cuco::bloom_filter_policy<int32_t, cuco::xxhash_64<int32_t>, 4, 1, 1, 1, 1, 1, 1>),
  (uint64_t, cuco::bloom_filter_policy<uint64_t, cuco::xxhash_64<uint64_t>, 4, 8, 12, 8, 1, 4, 2>),
  (float, cuco::bloom_filter_policy<float, cuco::xxhash_64<float>, 8, 4, 4, 2, 2, 1, 2>),
  (int32_t, cuco::bloom_filter_policy<int32_t, cuco::xxhash_64<int32_t>, 4, 8, 8, 2, 2, 1, 8>))
{
  using filter_type =
    cuco::bloom_filter<Key, cuco::extent<size_t>, cuda::thread_scope_device, Policy>;
  constexpr size_type num_keys{400};

  auto filter = filter_type{1000};

  test_unique_sequence(filter, num_keys);
}
