/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file static_map_non_bitwise_comparable_payload.cu
 *
 * @brief Demonstrates using static_map with a payload type that is not
 *        bitwise comparable.
 *
 * By default, cuCollections requires payload types to be bitwise comparable.
 * Defining CUCO_REQUIRE_BITWISE_COMPARABLE_PAYLOADS to 0 allows payload types
 * that do not have unique object representations. Enabling the flag will
 * cause static_map to not permit float as a payload type.
 * 
 * @note This example is for demonstration purposes only. It is not intended to show the most
 * performant way to do the example algorithm.
 */

#ifndef CUCO_REQUIRE_BITWISE_COMPARABLE_PAYLOADS
#define CUCO_REQUIRE_BITWISE_COMPARABLE_PAYLOADS 0
#endif

#include <cuco/detail/__config>
#include <cuco/static_map.cuh>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdint>
#include <cstring>
#include <iostream>

using map_type = cuco::static_map<
  int,
  float,
  cuco::extent<std::size_t>,
  cuda::thread_scope_device,
  cuda::std::equal_to<int>,
  cuco::linear_probing<1, cuco::murmurhash3_32<int>>,
  cuco::cuda_allocator<cuda::std::byte>>;

int main()
{
  map_type map{
    128,
    cuco::empty_key<int>{-1},
    cuco::empty_value<float>{0.0f}};

    // +0.0 and -0.0 compare equal, but have different object representations.
  float positive_zero = +0.0f;
  float negative_zero = -0.0f;

  std::uint32_t positive_zero_bits{};
  std::uint32_t negative_zero_bits{};

  std::memcpy(&positive_zero_bits, &positive_zero, sizeof(positive_zero));
  std::memcpy(&negative_zero_bits, &negative_zero, sizeof(negative_zero));

  thrust::device_vector<cuco::pair<int, float>> values{
    {1, positive_zero},
    {2, negative_zero}};

  map.insert(values.begin(), values.end());

  thrust::device_vector<int> keys{1, 2};
  thrust::device_vector<cuco::pair<int, float>> results(2);

  auto [unused, output_end] =
    map.retrieve(
      keys.begin(),
      keys.end(),
      cuda::discard_iterator{},
      results.begin());

  auto const num_retrieved =
    cuda::std::distance(results.begin(), output_end);

  thrust::host_vector<cuco::pair<int, float>> host_results = results;

  for (auto const& [key, value] : host_results) {
    std::uint32_t bits{};
    std::memcpy(&bits, &value, sizeof(value));

    std::cout << "key=" << key
              << ", value=" << value
              << ", bits=0x" << std::hex << bits << std::dec << '\n';
  }
}
