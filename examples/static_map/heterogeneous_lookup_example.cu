/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cuda/functional>
#include <cuda/std/tuple>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <iostream>

/**
 * @file heterogeneous_lookup_example.cu
 *
 * @brief Demonstrates how to perform heterogeneous lookups with `cuco::static_map`.
 *
 * In many workflows the format of the keys used when inserting into a hash table differs from the
 * format that is available at query time. `cuco` supports this scenario by allowing custom hash and
 * equality functors that can compare and hash "compatible" key types. This example stores keys as
 * two-field tuples `(sensor_id, channel)` and performs lookups using tuples that include an extra
 * timestamp `(sensor_id, channel, timestamp)`. The hash map only considers the first two elements
 * so both tuple types can interoperate transparently.
 */

using stored_key = cuda::std::tuple<int, int>;
using probe_key  = cuda::std::tuple<int, int, int>;
using value_type = float;

struct heterogeneous_hasher {
  template <typename Key>
  __device__ std::size_t operator()(Key const& key) const
  {
    auto const& ref  = thrust::raw_reference_cast(key);
    auto const major = cuda::std::get<0>(ref);
    auto const minor = cuda::std::get<1>(ref);
    return static_cast<std::size_t>(major * 131 + minor);
  }
};

struct heterogeneous_key_equal {
  template <typename LHS, typename RHS>
  __device__ bool operator()(LHS const& lhs, RHS const& rhs) const
  {
    auto const& left  = thrust::raw_reference_cast(lhs);
    auto const& right = thrust::raw_reference_cast(rhs);
    return (cuda::std::get<0>(left) == cuda::std::get<0>(right)) and
           (cuda::std::get<1>(left) == cuda::std::get<1>(right));
  }
};

int main()
{
  constexpr std::size_t num_entries = 4;
  auto constexpr empty_key          = stored_key{-1, -1};
  auto constexpr empty_value        = value_type{-1.0f};

  // Allocate a map with ~50% load factor.
  auto map =
    cuco::static_map{cuco::extent<std::size_t>{num_entries * 2},
                     cuco::empty_key{empty_key},
                     cuco::empty_value{empty_value},
                     heterogeneous_key_equal{},
                     cuco::linear_probing<1, heterogeneous_hasher>{heterogeneous_hasher{}}};

  // Host data describing the sensor readings we want to store.
  thrust::host_vector<stored_key> h_keys{
    stored_key{101, 3},
    stored_key{104, 8},
    stored_key{215, 1},
    stored_key{305, 0},
  };
  thrust::host_vector<value_type> h_values{36.5f, 41.2f, 27.1f, 33.8f};

  thrust::device_vector<stored_key> d_keys   = h_keys;
  thrust::device_vector<value_type> d_values = h_values;

  auto pairs_begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator<int>(0),
    cuda::proclaim_return_type<cuco::pair<stored_key, value_type>>(
      [keys = d_keys.begin(), values = d_values.begin()] __device__(int i) {
        return cuco::pair<stored_key, value_type>{keys[i], values[i]};
      }));

  map.insert(pairs_begin, pairs_begin + num_entries);

  // Probe keys include an additional timestamp field, but we only care about the first two
  // components when hashing / comparing.
  thrust::host_vector<probe_key> h_queries{
    probe_key{101, 3, 1210},  // present in the map
    probe_key{215, 1, 1345},  // present in the map
    probe_key{999, 4, 2000},  // missing entry
  };

  thrust::device_vector<probe_key> d_queries = h_queries;

  thrust::device_vector<bool> d_contains(h_queries.size());
  map.contains(d_queries.begin(), d_queries.end(), d_contains.begin());

  thrust::device_vector<value_type> d_found(h_queries.size());
  map.find(d_queries.begin(), d_queries.end(), d_found.begin());

  thrust::host_vector<bool> h_contains    = d_contains;
  thrust::host_vector<value_type> h_found = d_found;

  for (std::size_t i = 0; i < h_queries.size(); ++i) {
    auto const& query  = h_queries[i];
    auto const present = h_contains[i];
    std::cout << "Lookup (sensor " << cuda::std::get<0>(query) << ", channel "
              << cuda::std::get<1>(query) << ") -> " << (present ? "found" : "missing");

    if (present) { std::cout << ", stored value = " << h_found[i]; }

    std::cout << "\n";
  }

  return 0;
}
