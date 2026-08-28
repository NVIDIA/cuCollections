/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/roaring_bitmap.cuh>

#include <cuda/std/cstdint>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

int main()
{
  using index_type = cuda::std::uint32_t;

  thrust::device_vector<index_type> indices{0x00010002, 7, 1, 0x00010000, 7, 3, 0x00010002};

  auto bitmap =
    cuco::experimental::roaring_bitmap<index_type>::from_indices(indices.begin(), indices.end());

  thrust::device_vector<index_type> queries{1, 2, 3, 7, 0x00010000, 0x00010001, 0x00010002};
  thrust::device_vector<bool> results(queries.size());
  bitmap.contains(queries.begin(), queries.end(), results.begin());

  thrust::host_vector<bool> expected{true, false, true, true, true, false, true};
  thrust::host_vector<bool> actual = results;
  bool const success               = actual == expected;

  std::cout << "unique indices: " << bitmap.size() << '\n';
  std::cout << "serialized bytes: " << bitmap.size_bytes() << '\n';
  std::cout << "success: " << std::boolalpha << success << '\n';

  return success ? 0 : 1;
}
