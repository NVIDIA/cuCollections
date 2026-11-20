/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuco/hash_csr_multimap.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <catch2/catch_template_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

namespace {
// Gives every key one of five hash values and the same fingerprint, so that queries have to
// walk long probe sequences and compare keys.
template <class Key>
struct colliding_hash {
  __host__ __device__ std::uint32_t operator()(Key key) const
  {
    return static_cast<std::uint32_t>(key) % 5;
  }
};

template <class T>
auto ptr(thrust::device_vector<T>& values)
{
  return thrust::raw_pointer_cast(values.data());
}

template <class T>
std::vector<T> download(thrust::device_vector<T> const& values)
{
  thrust::host_vector<T> const host = values;
  return {host.begin(), host.end()};
}

template <class Key, class Hash>
void check_queries(std::vector<cuco::pair<Key, Key>> const& input,
                   cuco::hash_csr_multimap<Key, Key, cuda::std::equal_to<Key>, Hash>& map)
{
  using pair_type = cuco::pair<Key, Key>;
  constexpr Key missing{-1};

  std::multimap<Key, Key> expected;
  for (auto const& [key, value] : input) {
    expected.emplace(key, value);
  }

  thrust::device_vector<pair_type> records(input.begin(), input.end());
  map.build(ptr(records), ptr(records) + records.size());
  REQUIRE(map.size() == input.size());

  // Present keys, a repeated probe, and keys that are absent.
  std::vector<Key> queries;
  for (Key key = 0; key < 400; ++key) {
    queries.push_back(key);
  }
  queries.insert(queries.end(), {7, 7, 14'600, -5, 1'000'000});
  thrust::device_vector<Key> probes(queries.begin(), queries.end());

  std::size_t expected_count = 0;
  std::vector<std::tuple<Key, Key, Key>> expected_matches;
  for (auto const key : queries) {
    auto const [first, last] = expected.equal_range(key);
    for (auto it = first; it != last; ++it) {
      expected_matches.emplace_back(key, it->first, it->second);
    }
    expected_count += std::distance(first, last);
  }
  std::sort(expected_matches.begin(), expected_matches.end());

  REQUIRE(map.count(ptr(probes), ptr(probes) + probes.size()) == expected_count);

  thrust::device_vector<bool> contained(queries.size());
  thrust::device_vector<Key> found(queries.size());
  map.contains(ptr(probes), ptr(probes) + probes.size(), ptr(contained));
  map.find(ptr(probes), ptr(probes) + probes.size(), ptr(found));
  auto const host_contained = download(contained);
  auto const host_found     = download(found);
  for (std::size_t i = 0; i < queries.size(); ++i) {
    auto const [first, last] = expected.equal_range(queries[i]);
    REQUIRE(host_contained[i] == (first != last));
    if (first == last) {
      REQUIRE(host_found[i] == missing);
    } else {
      REQUIRE(
        std::any_of(first, last, [&](auto const& entry) { return entry.second == host_found[i]; }));
    }
  }

  thrust::device_vector<Key> output_probes(expected_count);
  thrust::device_vector<pair_type> output_matches(expected_count);
  auto const ends =
    map.retrieve(ptr(probes), ptr(probes) + probes.size(), ptr(output_probes), ptr(output_matches));
  REQUIRE(ends.first == ptr(output_probes) + expected_count);
  REQUIRE(ends.second == ptr(output_matches) + expected_count);
  auto const host_probes  = download(output_probes);
  auto const host_matches = download(output_matches);
  std::vector<std::tuple<Key, Key, Key>> matches;
  for (std::size_t i = 0; i < expected_count; ++i) {
    matches.emplace_back(host_probes[i], host_matches[i].first, host_matches[i].second);
  }
  std::sort(matches.begin(), matches.end());
  REQUIRE(matches == expected_matches);
}
}  // namespace

TEMPLATE_TEST_CASE_SIG("hash_csr_multimap matches a host multimap",
                       "",
                       ((typename Key, typename Hash), Key, Hash),
                       (std::int32_t, cuco::default_hash_function<std::int32_t>),
                       (std::int64_t, cuco::default_hash_function<std::int64_t>),
                       (std::int32_t, colliding_hash<std::int32_t>),
                       (std::int64_t, colliding_hash<std::int64_t>))
{
  using pair_type = cuco::pair<Key, Key>;
  constexpr std::size_t num_records{5'000};

  // Keys repeated about 13 times, one key repeated 500 times, and distinct keys.
  std::vector<pair_type> input;
  for (std::size_t i = 0; i < num_records; ++i) {
    auto const key = i < 4'000 ? i % 300 : i < 4'500 ? 7 : 10'000 + i;
    input.push_back({static_cast<Key>(key), static_cast<Key>(i)});
  }

  cuco::hash_csr_multimap<Key, Key, cuda::std::equal_to<Key>, Hash> map{num_records, Key{-1}};
  REQUIRE(map.capacity() == num_records);

  SECTION("A build answers count, contains, find and retrieve.") { check_queries(input, map); }

  SECTION("A rebuild replaces the previous contents.")
  {
    check_queries(input, map);
    input.resize(1'000);
    check_queries(input, map);
  }

  SECTION("An empty map matches nothing.")
  {
    input.clear();
    check_queries(input, map);
  }

  SECTION("Probes with thousands of matches fill several output batches per warp.")
  {
    std::vector<pair_type> large;
    for (std::size_t i = 0; i < 400'000; ++i) {
      large.push_back({static_cast<Key>(i % 200), static_cast<Key>(i)});
    }
    cuco::hash_csr_multimap<Key, Key, cuda::std::equal_to<Key>, Hash> large_map{large.size(),
                                                                                Key{-1}};
    check_queries(large, large_map);
  }
}
