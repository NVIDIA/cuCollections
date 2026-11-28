/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuco/bulk_static_multimap.cuh>

#include <thrust/device_vector.h>

#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{
  using pair_type = cuco::pair<int, int>;
  std::vector<pair_type> const input{{1, 10}, {1, 20}, {2, 30}};
  thrust::device_vector<pair_type> records(input.begin(), input.end());
  std::vector<int> const host_queries{1, 2, 3};
  thrust::device_vector<int> queries(host_queries.begin(), host_queries.end());
  auto const* first  = thrust::raw_pointer_cast(records.data());
  auto const* probes = thrust::raw_pointer_cast(queries.data());

  cuco::bulk_static_multimap<int, int> map{input.size(), -1};
  map.build(first, first + input.size());
  auto const matches = map.count(probes, probes + queries.size());
  if (matches != 3) throw std::runtime_error{"Unexpected match count"};
  thrust::device_vector<int> output_probes(matches);
  thrust::device_vector<pair_type> output_pairs(matches);
  map.retrieve(probes,
               probes + queries.size(),
               thrust::raw_pointer_cast(output_probes.data()),
               thrust::raw_pointer_cast(output_pairs.data()));
  std::cout << "Stored " << map.size() << " occurrences; retrieved " << matches << " matches\n";
}
