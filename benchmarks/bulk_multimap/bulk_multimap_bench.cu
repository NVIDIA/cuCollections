/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark_defaults.hpp>
#include <benchmark_utils.hpp>

#include <cuco/bulk_static_multimap.cuh>
#include <cuco/hash_csr_multimap.cuh>
#include <cuco/static_multimap.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace cuco::benchmark;  // defaults, dist_from_state
using namespace cuco::utility;    // key_generator, distribution

/**
 * @brief Builds, counts and retrieves the `static_multimap` benchmark workloads with each
 * multimap that is constructed from a complete input range.
 *
 * `Occupancy` sizes the slot table of `static_multimap` and `hash_csr_multimap`. `BucketSize` is
 * the average number of records per bucket of `bulk_static_multimap`; 8 is its default.
 */
struct hash_graph {};
struct hash_csr {};
struct open_addressing {};

NVBENCH_DECLARE_TYPE_STRINGS(hash_graph, "HashGraph", "cuco::bulk_static_multimap");
NVBENCH_DECLARE_TYPE_STRINGS(hash_csr, "HashCSR", "cuco::hash_csr_multimap");
NVBENCH_DECLARE_TYPE_STRINGS(open_addressing, "OpenAddressing", "cuco::static_multimap");

namespace {
// Every map is given the same keys.
constexpr std::uint32_t KEY_SEED = 880;

template <typename Key>
auto make_map(hash_graph, std::size_t num_keys, nvbench::state const& state)
{
  auto const buckets =
    static_cast<std::size_t>(std::ceil(num_keys / state.get_float64("BucketSize")));
  return cuco::bulk_static_multimap<Key, Key>{num_keys, Key{-1}, buckets};
}

template <typename Key>
auto make_map(hash_csr, std::size_t num_keys, nvbench::state const& state)
{
  return cuco::hash_csr_multimap<Key, Key>{num_keys, Key{-1}, state.get_float64("Occupancy")};
}

template <typename Key>
auto make_map(open_addressing, std::size_t num_keys, nvbench::state const& state)
{
  return cuco::static_multimap<Key, Key>{
    static_cast<std::size_t>(num_keys / state.get_float64("Occupancy")),
    cuco::empty_key<Key>{-1},
    cuco::empty_value<Key>{-1}};
}

template <typename Key>
void build(cuco::static_multimap<Key, Key>& map,
           cuco::pair<Key, Key> const* first,
           cuco::pair<Key, Key> const* last,
           cuda::stream_ref stream)
{
  // Emptying the slots is part of replacing the contents, as it is for the other two.
  map.clear(stream);
  map.insert(first, last, stream);
}

template <typename Map, typename Pair>
void build(Map& map, Pair const* first, Pair const* last, cuda::stream_ref stream)
{
  map.build(first, last, stream);
}

template <typename Key, typename Dist>
thrust::device_vector<cuco::pair<Key, Key>> make_pairs(nvbench::state const& state,
                                                       thrust::device_vector<Key>& keys)
{
  [[maybe_unused]] key_generator gen{KEY_SEED};
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  thrust::device_vector<cuco::pair<Key, Key>> pairs(keys.size());
  thrust::transform(keys.begin(), keys.end(), pairs.begin(), [] __device__(Key const& key) {
    return cuco::pair<Key, Key>{key, {}};
  });
  return pairs;
}
}  // namespace

template <typename Map, typename Key, typename Dist>
void bulk_multimap_build(nvbench::state& state, nvbench::type_list<Map, Key, Dist>)
{
  auto const num_keys = static_cast<std::size_t>(state.get_int64("NumInputs"));

  thrust::device_vector<Key> keys(num_keys);
  auto const pairs  = make_pairs<Key, Dist>(state, keys);
  auto const* first = pairs.data().get();

  state.add_element_count(num_keys);

  auto map = make_map<Key>(Map{}, num_keys, state);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    build(map, first, first + num_keys, {launch.get_stream()});
  });
}

template <typename Map, typename Key, typename Dist>
void bulk_multimap_count(nvbench::state& state, nvbench::type_list<Map, Key, Dist>)
{
  auto const num_keys      = static_cast<std::size_t>(state.get_int64("NumInputs"));
  auto const matching_rate = state.get_float64("MatchingRate");

  thrust::device_vector<Key> keys(num_keys);
  auto const pairs  = make_pairs<Key, Dist>(state, keys);
  auto const* first = pairs.data().get();

  [[maybe_unused]] key_generator gen{KEY_SEED + 1};
  gen.dropout(keys.begin(), keys.end(), matching_rate);
  auto const* probes = keys.data().get();

  state.add_element_count(num_keys);

  auto map = make_map<Key>(Map{}, num_keys, state);
  build(map, first, first + num_keys, cuda::stream_ref{cudaStream_t{nullptr}});

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    [[maybe_unused]] auto const count = map.count(probes, probes + num_keys, {launch.get_stream()});
  });
}

template <typename Map, typename Key, typename Dist>
void bulk_multimap_retrieve(nvbench::state& state, nvbench::type_list<Map, Key, Dist>)
{
  auto const num_keys      = static_cast<std::size_t>(state.get_int64("NumInputs"));
  auto const matching_rate = state.get_float64("MatchingRate");

  thrust::device_vector<Key> keys(num_keys);
  auto const pairs  = make_pairs<Key, Dist>(state, keys);
  auto const* first = pairs.data().get();

  [[maybe_unused]] key_generator gen{KEY_SEED + 1};
  gen.dropout(keys.begin(), keys.end(), matching_rate);
  auto const* probes = keys.data().get();

  state.add_element_count(num_keys);

  auto map = make_map<Key>(Map{}, num_keys, state);
  build(map, first, first + num_keys, cuda::stream_ref{cudaStream_t{nullptr}});

  auto const output_size = map.count(probes, probes + num_keys);
  thrust::device_vector<Key> output_probe(output_size);
  thrust::device_vector<cuco::pair<Key, Key>> output_match(output_size);
  state.add_element_count(output_size, "Matches");

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    map.retrieve(probes,
                 probes + num_keys,
                 output_probe.data().get(),
                 output_match.data().get(),
                 {launch.get_stream()});
  });
}

using MAP_RANGE = nvbench::type_list<hash_graph, hash_csr, open_addressing>;
using KEY_RANGE = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;

using SLOT_MAP_RANGE   = nvbench::type_list<hash_csr, open_addressing>;
using BUCKET_MAP_RANGE = nvbench::type_list<hash_graph>;

auto const ONE_N             = std::vector<nvbench::int64_t>{defaults::N};
auto const ONE_OCCUPANCY     = std::vector<nvbench::float64_t>{defaults::OCCUPANCY};
auto const ONE_BUCKET_SIZE   = std::vector<nvbench::float64_t>{8};
auto const ONE_MULTIPLICITY  = std::vector<nvbench::int64_t>{defaults::MULTIPLICITY};
auto const ONE_RATE          = std::vector<nvbench::float64_t>{defaults::MATCHING_RATE};
auto const BUCKET_SIZE_RANGE = std::vector<nvbench::float64_t>{0.5, 1, 2, 4, 8, 16};

#define BULK_MULTIMAP_SWEEP(                                                       \
  OPERATION, SWEEP, MAPS, NUM_INPUTS, OCCUPANCY, BUCKET_SIZE, MULTIPLICITY, RATE)  \
  NVBENCH_BENCH_TYPES(                                                             \
    bulk_multimap_##OPERATION,                                                     \
    NVBENCH_TYPE_AXES(MAPS, KEY_RANGE, nvbench::type_list<distribution::uniform>)) \
    .set_name("bulk_multimap_" #OPERATION "_uniform_" #SWEEP)                      \
    .set_type_axes_names({"Map", "Key", "Distribution"})                           \
    .add_int64_axis("NumInputs", NUM_INPUTS)                                       \
    .add_float64_axis("Occupancy", OCCUPANCY)                                      \
    .add_float64_axis("BucketSize", BUCKET_SIZE)                                   \
    .add_int64_axis("Multiplicity", MULTIPLICITY)                                  \
    .add_float64_axis("MatchingRate", RATE)

// The sweeps of the `static_multimap` benchmarks
#define BULK_MULTIMAP_CAPACITY(OPERATION)      \
  BULK_MULTIMAP_SWEEP(OPERATION,               \
                      capacity,                \
                      MAP_RANGE,               \
                      defaults::N_RANGE_CACHE, \
                      ONE_OCCUPANCY,           \
                      ONE_BUCKET_SIZE,         \
                      ONE_MULTIPLICITY,        \
                      ONE_RATE)
#define BULK_MULTIMAP_MULTIPLICITY(OPERATION)       \
  BULK_MULTIMAP_SWEEP(OPERATION,                    \
                      multiplicity,                 \
                      MAP_RANGE,                    \
                      ONE_N,                        \
                      ONE_OCCUPANCY,                \
                      ONE_BUCKET_SIZE,              \
                      defaults::MULTIPLICITY_RANGE, \
                      ONE_RATE)
#define BULK_MULTIMAP_MATCHING_RATE(OPERATION) \
  BULK_MULTIMAP_SWEEP(OPERATION,               \
                      matching_rate,           \
                      MAP_RANGE,               \
                      ONE_N,                   \
                      ONE_OCCUPANCY,           \
                      ONE_BUCKET_SIZE,         \
                      ONE_MULTIPLICITY,        \
                      defaults::MATCHING_RATE_RANGE)
#define BULK_MULTIMAP_OCCUPANCY(OPERATION)       \
  BULK_MULTIMAP_SWEEP(OPERATION,                 \
                      occupancy,                 \
                      SLOT_MAP_RANGE,            \
                      ONE_N,                     \
                      defaults::OCCUPANCY_RANGE, \
                      ONE_BUCKET_SIZE,           \
                      ONE_MULTIPLICITY,          \
                      ONE_RATE)
// The memory and scan length trade-off of `bulk_static_multimap`
#define BULK_MULTIMAP_BUCKET_SIZE(OPERATION) \
  BULK_MULTIMAP_SWEEP(OPERATION,             \
                      bucket_size,           \
                      BUCKET_MAP_RANGE,      \
                      ONE_N,                 \
                      ONE_OCCUPANCY,         \
                      BUCKET_SIZE_RANGE,     \
                      ONE_MULTIPLICITY,      \
                      ONE_RATE)

BULK_MULTIMAP_CAPACITY(build);
BULK_MULTIMAP_CAPACITY(count);
BULK_MULTIMAP_CAPACITY(retrieve);
BULK_MULTIMAP_MULTIPLICITY(build);
BULK_MULTIMAP_MULTIPLICITY(count);
BULK_MULTIMAP_MULTIPLICITY(retrieve);
BULK_MULTIMAP_MATCHING_RATE(count);
BULK_MULTIMAP_MATCHING_RATE(retrieve);
BULK_MULTIMAP_OCCUPANCY(build);
BULK_MULTIMAP_OCCUPANCY(count);
BULK_MULTIMAP_OCCUPANCY(retrieve);
BULK_MULTIMAP_BUCKET_SIZE(build);
BULK_MULTIMAP_BUCKET_SIZE(count);
BULK_MULTIMAP_BUCKET_SIZE(retrieve);
