/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuco/bulk_static_multimap.cuh>
#include <cuco/bulk_static_multimap_builder.cuh>

#include <catch2/catch_test_macros.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using pair_type = cuco::pair<int, int>;
using storage_type = cuco::bulk_multimap_storage<int, int>;
constexpr int missing_value = std::numeric_limits<int>::min();

template <class T>
auto ptr(thrust::device_vector<T>& values) { return thrust::raw_pointer_cast(values.data()); }

template <class T>
std::vector<T> download(thrust::device_vector<T> const& values)
{
  thrust::host_vector<T> const host = values;
  return {host.begin(), host.end()};
}

struct stream_owner {
  stream_owner() { CUCO_CUDA_TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  ~stream_owner() { cudaStreamDestroy(stream); }
  cudaStream_t stream;
  operator cuda::stream_ref() const { return cuda::stream_ref{stream}; }
};

struct constant_hash {
  __host__ __device__ std::uint64_t operator()(int) const { return 0xfffffff000000001ULL; }
};

struct modulo_hash {
  __host__ __device__ unsigned operator()(int key) const { return (key % 7 + 7) % 7; }
};
struct modulo_equal {
  __host__ __device__ bool operator()(int left, int right) const
  {
    return modulo_hash{}(left) == modulo_hash{}(right);
  }
};
struct large_payload { std::uint64_t words[5]; };

template <class Hash>
void check_queries(std::vector<pair_type> const& input, std::size_t buckets, Hash hash)
{
  stream_owner stream;
  thrust::device_vector<pair_type> records(input.begin(), input.end());
  cuco::bulk_static_multimap<int, int, cuda::std::equal_to<int>, Hash> map{
    input.size(), missing_value, buckets, {}, hash, {}, stream};
  map.build_async(ptr(records), ptr(records) + records.size(), stream);
  REQUIRE(map.size() == input.size());
  REQUIRE(map.capacity() == input.size());
  REQUIRE(map.bucket_count() == buckets);

  std::vector<int> const queries{0, 1, 1, 8, -1, missing_value, 123456789};
  thrust::device_vector<int> probes(queries.begin(), queries.end());
  thrust::device_vector<int> contains(queries.size()), found(queries.size());
  map.contains_async(ptr(probes), ptr(probes) + probes.size(), ptr(contains), stream);
  map.find(ptr(probes), ptr(probes) + probes.size(), ptr(found), stream);
  auto const presence = download(contains);
  auto const values = download(found);
  std::vector<std::pair<int, int>> expected;
  for (std::size_t q = 0; q < queries.size(); ++q) {
    bool exists = false, value_matches = false;
    for (auto const& pair : input) {
      if (pair.first != queries[q]) continue;
      exists = true;
      value_matches |= values[q] == pair.second;
      expected.emplace_back(queries[q], pair.second);
    }
    REQUIRE(presence[q] == exists);
    REQUIRE((exists ? value_matches : values[q] == missing_value));
  }
  REQUIRE(map.count(ptr(probes), ptr(probes) + probes.size(), stream) == expected.size());
  thrust::device_vector<int> out_probes(expected.size());
  thrust::device_vector<pair_type> out_matches(expected.size());
  auto const result = map.retrieve(ptr(probes), ptr(probes) + probes.size(),
                                   ptr(out_probes), ptr(out_matches), stream);
  if (!expected.empty()) {
    REQUIRE(result.first - ptr(out_probes) == static_cast<std::ptrdiff_t>(expected.size()));
    REQUIRE(result.second - ptr(out_matches) == static_cast<std::ptrdiff_t>(expected.size()));
  }
  auto const matched_probes = download(out_probes);
  auto const matched_records = download(out_matches);
  std::vector<std::pair<int, int>> actual;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(matched_probes[i] == matched_records[i].first);
    actual.emplace_back(matched_records[i].first, matched_records[i].second);
  }
  std::sort(expected.begin(), expected.end());
  std::sort(actual.begin(), actual.end());
  REQUIRE(actual == expected);

  thrust::device_vector<int> all_keys(input.size()), all_values(input.size());
  map.retrieve_all(ptr(all_keys), ptr(all_values), stream);
  auto const keys = download(all_keys), payloads = download(all_values);
  expected.clear(); actual.clear();
  for (std::size_t i = 0; i < input.size(); ++i) {
    expected.emplace_back(input[i].first, input[i].second);
    actual.emplace_back(keys[i], payloads[i]);
  }
  std::sort(expected.begin(), expected.end());
  std::sort(actual.begin(), actual.end());
  REQUIRE(actual == expected);

  thrust::device_vector<std::size_t> visits(queries.size());
  auto const view = map.ref(cuco::count, cuco::find, cuco::for_each, cuco::contains);
  auto const* device_probes = ptr(probes);
  auto* device_visits = ptr(visits);
  cuco::detail::bulk_multimap_launch(queries.size(), stream, [=] __device__ (std::size_t i) {
    std::size_t count = 0;
    view.for_each(device_probes[i], [&] __device__ (auto it) { count += it->first == device_probes[i]; });
    auto const it = view.find(device_probes[i]);
    device_visits[i] = count == view.count(device_probes[i]) &&
                      view.contains(device_probes[i]) == (it != view.end());
  });
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream.stream));
  for (auto value : download(visits)) REQUIRE(value == 1);

  thrust::device_vector<int> total_visits(1, 0);
  auto* total = ptr(total_visits);
  map.for_each([=] __device__ (pair_type) { atomicAdd(total, 1); }, stream);
  REQUIRE(static_cast<std::size_t>(download(total_visits)[0]) == input.size());
  map.for_each(ptr(probes), ptr(probes) + probes.size(),
               [=] __device__ (pair_type) { atomicAdd(total, 1); }, stream);
  REQUIRE(static_cast<std::size_t>(download(total_visits)[0]) ==
          input.size() + map.count(ptr(probes), ptr(probes) + probes.size(), stream));

  map.build(ptr(records), ptr(records), stream);
  REQUIRE(map.size() == 0);
  REQUIRE(map.count(ptr(probes), ptr(probes) + probes.size(), stream) == 0);
  map.build(ptr(records), ptr(records) + records.size(), stream);
  REQUIRE(map.size() == input.size());
  if (!input.empty()) {
    auto moved = std::move(map);
    REQUIRE(moved.count(ptr(probes), ptr(probes) + probes.size(), stream) == matched_records.size());
    REQUIRE_THROWS_AS(moved.build(ptr(records) + 1, ptr(records), stream), cuco::logic_error);
  }
}

template <class Size>
__global__ void build_independent_tables(pair_type const* input,
                                        std::size_t const* counts,
                                        cuco::bulk_multimap_storage<int, int, Size>* storage,
                                        cuco::bulk_build_status* status)
{
  using builder = cuco::bulk_static_multimap_builder<int, int, 128, Size>;
  __shared__ typename builder::TempStorage temp;
  auto const table = blockIdx.x;
  auto const result = builder{temp}.build(cooperative_groups::this_thread_block(),
                                         input + table * 4097, counts[table], storage[table]);
  if (!threadIdx.x) status[table] = result;
}

template <class Size>
void check_collective_builds()
{
  std::vector<std::size_t> const counts{0, 1, 31, 127, 128, 129, 4097};
  constexpr std::size_t buckets = 137;
  std::vector<pair_type> input(counts.size() * 4097);
  for (std::size_t table = 0; table < counts.size(); ++table) {
    for (std::size_t i = 0; i < counts[table]; ++i) {
      input[table * 4097 + i] = pair_type{static_cast<int>(i % 17), static_cast<int>(i)};
    }
  }
  thrust::device_vector<pair_type> device_input(input.begin(), input.end()), output(input.size());
  thrust::device_vector<Size> offsets(counts.size() * (buckets + 1));
  thrust::device_vector<Size> counters(counts.size() * buckets);
  thrust::device_vector<std::size_t> device_counts(counts.begin(), counts.end());
  thrust::device_vector<cuco::bulk_build_status> statuses(counts.size());
  using storage = cuco::bulk_multimap_storage<int, int, Size>;
  std::vector<storage> descriptors;
  for (std::size_t table = 0; table < counts.size(); ++table) {
    descriptors.push_back({ptr(output) + table * 4097, 4097,
                           ptr(offsets) + table * (buckets + 1), buckets + 1,
                           ptr(counters) + table * buckets, buckets, buckets});
  }
  thrust::device_vector<storage> device_storage(descriptors.begin(), descriptors.end());
  build_independent_tables<<<counts.size(), 128>>>(ptr(device_input), ptr(device_counts),
                                                   ptr(device_storage), ptr(statuses));
  CUCO_CUDA_TRY(cudaGetLastError());
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  for (auto value : download(statuses)) REQUIRE(value == cuco::bulk_build_status::success);
  thrust::device_vector<Size>{}.swap(counters);
  thrust::device_vector<pair_type>{}.swap(device_input);
  thrust::device_vector<Size> query_counts(counts.size());
  auto* query_results = ptr(query_counts);
  auto const* retained = ptr(device_storage);
  auto const* record_counts = ptr(device_counts);
  cuco::detail::bulk_multimap_launch(counts.size(), cuda::stream_ref{cudaStream_t{nullptr}},
    [=] __device__ (std::size_t i) {
      auto const view = retained[i].ref(record_counts[i], cuda::std::equal_to<int>{},
                                         cuco::default_hash_function<int>{}, cuco::count);
      query_results[i] = view.count(1);
    });
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  auto const counted = download(query_counts);
  for (std::size_t i = 0; i < counts.size(); ++i) {
    REQUIRE(counted[i] == counts[i] / 17 + (counts[i] % 17 > 1));
  }
  auto const result = download(output);
  auto const starts = download(offsets);
  for (std::size_t table = 0; table < counts.size(); ++table) {
    REQUIRE(starts[table * (buckets + 1)] == 0);
    REQUIRE(starts[table * (buckets + 1) + buckets] == counts[table]);
    std::vector<std::pair<int, int>> expected, actual;
    for (std::size_t b = 0; b < buckets; ++b) {
      auto const begin = starts[table * (buckets + 1) + b];
      auto const end = starts[table * (buckets + 1) + b + 1];
      REQUIRE(begin <= end);
      REQUIRE(end <= counts[table]);
      for (auto i = begin; i < end; ++i) {
        auto const pair = result[table * 4097 + i];
        REQUIRE(cuco::detail::bulk_bucket(cuco::default_hash_function<int>{}, pair.first,
                                         static_cast<Size>(buckets)) == b);
        actual.emplace_back(pair.first, pair.second);
      }
    }
    for (std::size_t i = 0; i < counts[table]; ++i) {
      auto const pair = input[table * 4097 + i];
      expected.emplace_back(pair.first, pair.second);
    }
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    REQUIRE(actual == expected);
  }
  // Insufficient storage is rejected collectively before any table arrays are written.
  for (auto& descriptor : descriptors) descriptor.offset_capacity = buckets;
  device_storage = descriptors;
  build_independent_tables<<<counts.size(), 128>>>(ptr(device_input), ptr(device_counts),
                                                   ptr(device_storage), ptr(statuses));
  CUCO_CUDA_TRY(cudaGetLastError());
  CUCO_CUDA_TRY(cudaDeviceSynchronize());
  for (auto value : download(statuses)) REQUIRE(value == cuco::bulk_build_status::insufficient_capacity);
}
}  // namespace

TEST_CASE("Bulk multimap descriptor contracts", "[bulk_static_multimap][host]")
{
  pair_type record;
  std::size_t offsets[3], counters[2];
  storage_type descriptor{&record, 1, offsets, 3, counters, 2, 2};
  REQUIRE(descriptor.validate(1) == cuco::bulk_build_status::success);
  REQUIRE(descriptor.validate(2) == cuco::bulk_build_status::insufficient_capacity);
  descriptor.bucket_count = 0;
  REQUIRE(descriptor.validate(0) == cuco::bulk_build_status::invalid_size);
  descriptor.bucket_count = 2;
  descriptor.offsets = counters;
  REQUIRE(descriptor.validate(0) == cuco::bulk_build_status::invalid_storage);
  descriptor.offsets = offsets;
  descriptor.records = nullptr;
  REQUIRE(descriptor.validate(0) == cuco::bulk_build_status::success);
  REQUIRE(descriptor.validate(1) == cuco::bulk_build_status::invalid_storage);
  cuco::bulk_multimap_storage<int, int, std::uint32_t> narrow{nullptr, 0, nullptr, 0, nullptr, 0, 1};
  REQUIRE(narrow.validate(std::uint64_t{1} << 32) == cuco::bulk_build_status::invalid_size);
  REQUIRE(cuco::detail::bulk_bucket(constant_hash{}, 0, std::uint64_t{137}) ==
          0xfffffff000000001ULL % 137);
  static_assert(std::is_trivially_copyable_v<decltype(descriptor.ref(
    0, cuda::std::equal_to<int>{}, cuco::default_hash_function<int>{}, cuco::find))>);
}

TEST_CASE("Bulk multimap rejects overflowing capacity before allocation", "[bulk_static_multimap][host]")
{
  using map_type = cuco::bulk_static_multimap<int, int>;
  REQUIRE_THROWS_AS(map_type(std::numeric_limits<std::size_t>::max(), -1), cuco::logic_error);
}

TEST_CASE("Bulk multimap preserves exact occurrences and query semantics", "[bulk_static_multimap][gpu]")
{
  for (auto size : {0, 1, 31, 127, 128, 129, 1025, 65537}) {
    CAPTURE(size);
    std::vector<pair_type> input;
    for (int i = 0; i < size; ++i) input.emplace_back(i % 17, i % 113);
    if (size) input[0] = pair_type{missing_value, missing_value};
    check_queries(input, 137, cuco::default_hash_function<int>{});
  }
}

TEST_CASE("Bulk multimap handles one collision bucket and identical pairs", "[bulk_static_multimap][gpu]")
{
  std::vector<pair_type> input(4097, pair_type{1, 42});
  input.back() = pair_type{8, 99};
  check_queries(input, 17, constant_hash{});
}

TEST_CASE("Bulk multimap supports custom equality and compatible hashing", "[bulk_static_multimap][gpu]")
{
  thrust::device_vector<pair_type> input{pair_type{-1, 10}, pair_type{6, 20}, pair_type{13, 30}};
  cuco::bulk_static_multimap<int, int, modulo_equal, modulo_hash> map{3, -999, 3};
  map.build(ptr(input), ptr(input) + input.size());
  thrust::device_vector<int> probes{20, -8, 0};
  REQUIRE(map.count(ptr(probes), ptr(probes) + probes.size()) == 6);
  thrust::device_vector<int> output_probes(6);
  thrust::device_vector<pair_type> output_pairs(6);
  map.retrieve(ptr(probes), ptr(probes) + probes.size(), ptr(output_probes), ptr(output_pairs));
  auto const actual_probes = download(output_probes);
  auto const actual_pairs = download(output_pairs);
  std::vector<std::pair<int, int>> actual;
  for (std::size_t i = 0; i < actual_pairs.size(); ++i) {
    REQUIRE(modulo_equal{}(actual_probes[i], actual_pairs[i].first));
    actual.emplace_back(actual_probes[i], actual_pairs[i].second);
  }
  std::sort(actual.begin(), actual.end());
  std::vector<std::pair<int, int>> const expected{{-8, 10}, {-8, 20}, {-8, 30},
                                                {20, 10}, {20, 20}, {20, 30}};
  REQUIRE(actual == expected);
}

TEST_CASE("Bulk multimap collective builds preserve exact CSR", "[bulk_static_multimap][gpu]")
{
  check_collective_builds<std::uint32_t>();
  check_collective_builds<std::uint64_t>();
}

TEST_CASE("Bulk multimap allows large payloads and wide keys", "[bulk_static_multimap][gpu]")
{
  using payload = large_payload;
  using record = cuco::pair<std::uint64_t, payload>;
  std::vector<record> const input{{0xffffffffffffffffULL, {{1, 2, 3, 4, 5}}},
                                 {0xffffffffffffffffULL, {{6, 7, 8, 9, 10}}}};
  thrust::device_vector<record> records(input.begin(), input.end());
  cuco::bulk_static_multimap<std::uint64_t, payload> map{2, payload{}, 3};
  map.build(ptr(records), ptr(records) + 2);
  thrust::device_vector<std::uint64_t> probes(1, 0xffffffffffffffffULL);
  REQUIRE(map.count(ptr(probes), ptr(probes) + 1) == 2);
  thrust::device_vector<std::uint64_t> keys(2);
  thrust::device_vector<payload> values(2);
  map.retrieve_all(ptr(keys), ptr(values));
  auto const result = download(values);
  std::vector<std::uint64_t> flattened;
  for (auto const& value : result) flattened.insert(flattened.end(), value.words, value.words + 5);
  std::sort(flattened.begin(), flattened.end());
  REQUIRE(flattened == std::vector<std::uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  REQUIRE_THROWS_AS(map.build(ptr(records), ptr(records) + 3), cuco::logic_error);
}
