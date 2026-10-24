/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCO_REQUIRE_BITWISE_COMPARABLE_PAYLOADS
#define CUCO_REQUIRE_BITWISE_COMPARABLE_PAYLOADS 0
#endif

#include <test_utils.hpp>

#include <cuco/detail/__config>
#include <cuco/static_multimap.cuh>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

using Key   = int;
using Value = float;

using extent_type = cuco::extent<std::size_t>;

using probe = cuco::linear_probing<1, cuco::murmurhash3_32<Key>>;

using map_type = cuco::static_multimap<Key,
                                       Value,
                                       extent_type,
                                       cuda::thread_scope_device,
                                       cuda::std::equal_to<Key>,
                                       probe,
                                       cuco::cuda_allocator<cuda::std::byte>,
                                       cuco::storage<2>>;

TEST_CASE("static_multimap preserves +0.0 and -0.0 payloads", "")
{
  auto map = map_type{128, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{0.0f}};

  std::uint32_t positive_zero_bits{};
  std::uint32_t negative_zero_bits{};

  float positive_zero = +0.0f;
  float negative_zero = -0.0f;

  std::memcpy(&positive_zero_bits, &positive_zero, sizeof(positive_zero));
  std::memcpy(&negative_zero_bits, &negative_zero, sizeof(negative_zero));

  REQUIRE(positive_zero_bits == 0x00000000u);
  REQUIRE(negative_zero_bits == 0x80000000u);
  REQUIRE(positive_zero_bits != negative_zero_bits);

  thrust::device_vector<cuco::pair<Key, Value>> values{cuco::pair<Key, Value>{0, positive_zero},
                                                       cuco::pair<Key, Value>{0, negative_zero}};

  map.insert(values.begin(), values.end());

  thrust::device_vector<Key> query_keys{0};
  thrust::device_vector<cuco::pair<Key, Value>> results(2);

  auto const [_, output_end] =
    map.retrieve(query_keys.begin(), query_keys.end(), cuda::discard_iterator{}, results.begin());

  auto const num_retrieved = cuda::std::distance(results.begin(), output_end);

  REQUIRE(num_retrieved == 2);

  thrust::host_vector<cuco::pair<Key, Value>> host_results = results;

  std::uint32_t first_bits{};
  std::uint32_t second_bits{};

  float first_value  = host_results[0].second;
  float second_value = host_results[1].second;

  std::memcpy(&first_bits, &first_value, sizeof(first_value));
  std::memcpy(&second_bits, &second_value, sizeof(second_value));

  REQUIRE(first_bits != second_bits);

  bool payload_preserved = (first_bits == 0x00000000u and second_bits == 0x80000000u) or
                           (first_bits == 0x80000000u and second_bits == 0x00000000u);
  REQUIRE(payload_preserved);
}
