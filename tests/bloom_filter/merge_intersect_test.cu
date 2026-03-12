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
#include <cuco/utility/error.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/device_vector.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>
#include <exception>

using size_type = int32_t;

template <typename Filter>
void test_merge_intersect(Filter& filter_a,
                          Filter& filter_b,
                          Filter const& filter_c,
                          size_type capacity)
{
  using Key = typename Filter::key_type;

  size_type num_keys  = capacity;
  size_type half_keys = capacity / 2;

  auto to_key = cuda::proclaim_return_type<Key>([] __device__(size_type i) { return Key(i); });

  auto keys_a_begin = cuda::make_transform_iterator(cuda::counting_iterator<size_type>{0}, to_key);
  auto keys_a_end   = keys_a_begin + num_keys;

  auto keys_b_begin =
    cuda::make_transform_iterator(cuda::counting_iterator<size_type>{half_keys}, to_key);
  auto keys_b_end = keys_b_begin + num_keys;

  auto keys_intersection_begin =
    cuda::make_transform_iterator(cuda::counting_iterator<size_type>{half_keys}, to_key);
  auto keys_intersection_end = keys_intersection_begin + half_keys;

  auto keys_union_begin =
    cuda::make_transform_iterator(cuda::counting_iterator<size_type>{0}, to_key);
  auto keys_union_end = keys_union_begin + num_keys + half_keys;

  auto keys_unique_a_begin =
    cuda::make_transform_iterator(cuda::counting_iterator<size_type>{0}, to_key);
  auto keys_unique_a_end = keys_unique_a_begin + half_keys;

  auto keys_unique_b_begin =
    cuda::make_transform_iterator(cuda::counting_iterator<size_type>{num_keys}, to_key);
  auto keys_unique_b_end = keys_unique_b_begin + half_keys;

  // Helper to fill filters
  auto refill_filters = [&]() {
    filter_a.clear();
    filter_a.add(keys_a_begin, keys_a_end);

    filter_b.clear();
    filter_b.add(keys_b_begin, keys_b_end);
  };

  // Reusable output vector (sized for largest query: union)
  thrust::device_vector<bool> contained(num_keys + half_keys);

  SECTION("Merge B into A")
  {
    refill_filters();
    filter_a.merge(filter_b);

    // Check A contains all of Union
    filter_a.contains(keys_union_begin, keys_union_end, contained.begin());
    REQUIRE(cuco::test::all_of(
      contained.begin(), contained.begin() + num_keys + half_keys, cuda::std::identity{}));

    // Check B is unchanged
    filter_b.contains(keys_b_begin, keys_b_end, contained.begin());
    REQUIRE(
      cuco::test::all_of(contained.begin(), contained.begin() + num_keys, cuda::std::identity{}));
  }

  SECTION("Intersect B into A")
  {
    refill_filters();
    filter_a.intersect(filter_b);

    // Check A contains Intersection
    filter_a.contains(keys_intersection_begin, keys_intersection_end, contained.begin());
    REQUIRE(
      cuco::test::all_of(contained.begin(), contained.begin() + half_keys, cuda::std::identity{}));

    // Check A does NOT contain Unique A (approximate)
    // We expect none_of, but due to false positives, we might get some.
    // However, for this test configuration, we expect 0 false positives if the filter is
    // reasonably sized.
    filter_a.contains(keys_unique_a_begin, keys_unique_a_end, contained.begin());
    REQUIRE(
      cuco::test::none_of(contained.begin(), contained.begin() + half_keys, cuda::std::identity{}));

    // Check A does NOT contain Unique B
    filter_a.contains(keys_unique_b_begin, keys_unique_b_end, contained.begin());
    REQUIRE(
      cuco::test::none_of(contained.begin(), contained.begin() + half_keys, cuda::std::identity{}));
  }

  SECTION("Merge empty filter into A")
  {
    filter_a.clear();
    filter_a.add(keys_a_begin, keys_a_end);
    filter_b.clear();  // B is empty

    filter_a.merge(filter_b);

    // A should still contain all of Set A
    filter_a.contains(keys_a_begin, keys_a_end, contained.begin());
    REQUIRE(
      cuco::test::all_of(contained.begin(), contained.begin() + num_keys, cuda::std::identity{}));
  }

  SECTION("Intersect empty filter into A")
  {
    filter_a.clear();
    filter_a.add(keys_a_begin, keys_a_end);
    filter_b.clear();  // B is empty

    filter_a.intersect(filter_b);

    // A should now be empty (intersection with empty set)
    filter_a.contains(keys_a_begin, keys_a_end, contained.begin());
    REQUIRE(
      cuco::test::none_of(contained.begin(), contained.begin() + num_keys, cuda::std::identity{}));
  }

  SECTION("Mismatched block counts")
  {
    // also test with custom stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    REQUIRE_THROWS_AS(filter_a.merge(filter_c, stream), cuco::logic_error);
    REQUIRE_THROWS_AS(filter_a.intersect(filter_c, stream), cuco::logic_error);
    cudaStreamDestroy(stream);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "bloom_filter merge and intersect tests",
  "",
  ((class Key, class Policy), Key, Policy),
  (int32_t, cuco::default_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 1>),
  (int32_t, cuco::default_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 8>),
  (int64_t, cuco::default_filter_policy<cuco::xxhash_64<int64_t>, uint64_t, 1>),
  (int64_t, cuco::default_filter_policy<cuco::xxhash_64<int64_t>, uint64_t, 8>))
{
  using filter_type =
    cuco::bloom_filter<Key, cuco::extent<size_t>, cuda::thread_scope_device, Policy>;
  constexpr size_type capacity{1000};

  uint32_t pattern_bits = Policy::words_per_block + GENERATE(0, 1);

  // some parameter combinations might be invalid so we skip them
  try {
    [[maybe_unused]] auto policy = Policy{pattern_bits};
  } catch (std::exception const& e) {
    SKIP(e.what());
  }

  auto filter_a = filter_type{capacity, {}, {pattern_bits}};
  auto filter_b = filter_type{capacity, {}, {pattern_bits}};
  auto filter_c = filter_type{static_cast<size_t>(capacity) * 2, {}, {pattern_bits}};

  test_merge_intersect(filter_a, filter_b, filter_c, capacity);
}

TEMPLATE_TEST_CASE_SIG("bloom_filter merge and intersect arrow tests",
                       "",
                       ((class Key, class Policy), Key, Policy),
                       (int32_t, cuco::arrow_filter_policy<int32_t>),
                       (int64_t, cuco::arrow_filter_policy<int64_t>),
                       (float, cuco::arrow_filter_policy<float>))
{
  using filter_type =
    cuco::bloom_filter<Key, cuco::extent<size_t>, cuda::thread_scope_device, Policy>;
  constexpr size_type capacity{1000};  // Must match capacity used in helper logic

  auto filter_a = filter_type{capacity};
  auto filter_b = filter_type{capacity};
  auto filter_c = filter_type{static_cast<size_t>(capacity) * 2};

  test_merge_intersect(filter_a, filter_b, filter_c, capacity);
}
