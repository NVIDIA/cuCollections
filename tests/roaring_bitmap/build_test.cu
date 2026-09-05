/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.cuh"

#include <cuco/detail/error.hpp>
#include <cuco/roaring_bitmap.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/stream_ref>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

using index_type  = cuda::std::uint32_t;
using bitmap_type = cuco::experimental::roaring_bitmap<index_type>;

std::vector<cuda::std::byte> copy_serialized(bitmap_type const& bitmap)
{
  std::vector<cuda::std::byte> bytes(bitmap.size_bytes());
  CUCO_CUDA_TRY(cudaMemcpy(bytes.data(), bitmap.data(), bytes.size(), cudaMemcpyDeviceToHost));
  return bytes;
}

void require_contains(bitmap_type const& bitmap,
                      std::vector<index_type> const& queries,
                      std::vector<bool> const& expected)
{
  thrust::device_vector<index_type> query_indices(queries);
  thrust::device_vector<bool> results(queries.size());
  bitmap.contains(query_indices.begin(), query_indices.end(), results.begin());
  thrust::host_vector<bool> host_results = results;

  REQUIRE(host_results.size() == expected.size());
  for (cuda::std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(host_results[i] == expected[i]);
  }
}

struct allocation_counts {
  cuda::std::size_t allocations   = 0;
  cuda::std::size_t deallocations = 0;
};

template <class T>
class tracking_allocator {
 public:
  using value_type = T;

  tracking_allocator() : counts_{std::make_shared<allocation_counts>()} {}

  explicit tracking_allocator(std::shared_ptr<allocation_counts> counts) : counts_{counts} {}

  template <class U>
  tracking_allocator(tracking_allocator<U> const& other) noexcept : counts_{other.counts()}
  {
  }

  value_type* allocate(cuda::std::size_t size, cuda::stream_ref stream)
  {
    ++counts_->allocations;
    value_type* data;
    CUCO_CUDA_TRY(cudaMallocAsync(&data, size * sizeof(value_type), stream.get()));
    return data;
  }

  void deallocate(value_type* data, cuda::std::size_t, cuda::stream_ref stream)
  {
    ++counts_->deallocations;
    CUCO_CUDA_TRY(cudaFreeAsync(data, stream.get()));
  }

  [[nodiscard]] std::shared_ptr<allocation_counts> counts() const noexcept { return counts_; }

 private:
  template <class>
  friend class tracking_allocator;

  std::shared_ptr<allocation_counts> counts_;
};

template <class T, class U>
bool operator==(tracking_allocator<T> const& lhs, tracking_allocator<U> const& rhs) noexcept
{
  return lhs.counts() == rhs.counts();
}

template <class T, class U>
bool operator!=(tracking_allocator<T> const& lhs, tracking_allocator<U> const& rhs) noexcept
{
  return not(lhs == rhs);
}

}  // namespace

TEST_CASE("roaring_bitmap builds an empty bitmap", "[roaring_bitmap]")
{
  thrust::device_vector<index_type> indices;
  auto bitmap        = bitmap_type::from_indices(indices.begin(), indices.end());
  auto sorted_bitmap = bitmap_type::from_sorted_indices(indices.begin(), indices.end());
  auto sorted_unique_bitmap =
    bitmap_type::from_sorted_unique_indices(indices.begin(), indices.end());

  REQUIRE(bitmap.empty());
  REQUIRE(bitmap.size() == 0);
  REQUIRE(bitmap.size_bytes() == 2 * sizeof(cuda::std::uint32_t));
  REQUIRE(copy_serialized(bitmap) == copy_serialized(sorted_bitmap));
  REQUIRE(copy_serialized(bitmap) == copy_serialized(sorted_unique_bitmap));
}

TEST_CASE("roaring_bitmap matches RoaringFormatSpec no-run serialization", "[roaring_bitmap]")
{
#ifndef CUCO_ROARING_DATA_DIR
  SKIP(
    "CUCO_ROARING_DATA_DIR is not defined. Configure with -DCUCO_DOWNLOAD_ROARING_TESTDATA=ON to "
    "run this test.");
#else
  auto const host_indices = cuco::test::make_roaring_bitmap_without_runs_indices();

  thrust::device_vector<index_type> indices{host_indices};
  auto bitmap = bitmap_type::from_sorted_unique_indices(indices.begin(), indices.end());

  std::string const path = std::string{CUCO_ROARING_DATA_DIR} + "/bitmapwithoutruns.bin";
  REQUIRE(std::filesystem::exists(path));
  std::ifstream file{path, std::ios::binary | std::ios::ate};
  REQUIRE(file.is_open());
  auto const size = static_cast<cuda::std::size_t>(file.tellg());
  std::vector<cuda::std::byte> expected(size);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(expected.data()), size);

  REQUIRE(bitmap.size() == host_indices.size());
  REQUIRE(copy_serialized(bitmap) == expected);
#endif
}

TEST_CASE("roaring_bitmap rejects reversed input ranges", "[roaring_bitmap]")
{
  thrust::device_vector<index_type> indices{1, 2, 3};

  REQUIRE_THROWS_AS(bitmap_type::from_indices(indices.end(), indices.begin()), cuco::logic_error);
  REQUIRE_THROWS_AS(bitmap_type::from_sorted_indices(indices.end(), indices.begin()),
                    cuco::logic_error);
  REQUIRE_THROWS_AS(bitmap_type::from_sorted_unique_indices(indices.end(), indices.begin()),
                    cuco::logic_error);
}

TEST_CASE("roaring_bitmap builds format boundary indices", "[roaring_bitmap]")
{
  auto constexpr max_index = cuda::std::numeric_limits<index_type>::max();
  thrust::device_vector<index_type> indices{0, max_index};

  auto bitmap = bitmap_type::from_sorted_unique_indices(indices.begin(), indices.end());

  REQUIRE(bitmap.size() == 2);
  require_contains(bitmap, {0, 1, max_index - 1, max_index}, {true, false, false, true});
}

TEST_CASE("roaring_bitmap removes duplicates across a container boundary", "[roaring_bitmap]")
{
  thrust::device_vector<index_type> indices{
    0x0000FFFE, 0x0000FFFF, 0x0000FFFF, 0x00010000, 0x00010000, 0x00010001};

  auto bitmap = bitmap_type::from_sorted_indices(indices.begin(), indices.end());

  REQUIRE(bitmap.size() == 4);
  require_contains(bitmap,
                   {0x0000FFFD, 0x0000FFFE, 0x0000FFFF, 0x00010000, 0x00010001, 0x00010002},
                   {false, true, true, true, true, false});
}

TEST_CASE("roaring_bitmap builds a full container", "[roaring_bitmap]")
{
  constexpr cuda::std::uint32_t num_indices = 1 << 16;
  thrust::device_vector<index_type> indices(num_indices);
  thrust::sequence(indices.begin(), indices.end(), index_type{0x12340000});

  auto bitmap = bitmap_type::from_sorted_unique_indices(indices.begin(), indices.end());

  REQUIRE(bitmap.size() == num_indices);
  require_contains(
    bitmap, {0x1233FFFF, 0x12340000, 0x1234FFFF, 0x12350000}, {false, true, true, false});
}

TEST_CASE("roaring_bitmap builds the maximum number of containers", "[roaring_bitmap]")
{
  constexpr cuda::std::uint32_t num_containers = 1 << 16;
  thrust::device_vector<index_type> indices(num_containers);
  thrust::tabulate(indices.begin(), indices.end(), [] __device__(index_type container) {
    return (container << 16) | index_type{7};
  });

  auto bitmap = bitmap_type::from_sorted_unique_indices(indices.begin(), indices.end());

  REQUIRE(bitmap.size() == num_containers);
  require_contains(bitmap,
                   {7, 8, 0x7FFF0007, 0x80000007, 0xFFFF0007, 0xFFFF0008},
                   {true, false, true, true, true, false});
}

TEST_CASE("roaring_bitmap builds multiple array containers per block", "[roaring_bitmap]")
{
  std::vector<cuda::std::uint16_t> const cardinalities{
    1, 31, 32, 33, 255, 256, 257, 4094, 4095, 4096};
  std::vector<index_type> host_indices;

  for (cuda::std::size_t container = 0; container < cardinalities.size(); ++container) {
    for (cuda::std::uint32_t lower = 0; lower < cardinalities[container]; ++lower) {
      host_indices.push_back((static_cast<index_type>(container) << 16) | lower);
    }
  }

  thrust::device_vector<index_type> indices(host_indices);
  auto bitmap = bitmap_type::from_sorted_unique_indices(indices.begin(), indices.end());

  auto const expected_size =
    2 * sizeof(cuda::std::uint32_t) +
    cardinalities.size() * (2 * sizeof(cuda::std::uint16_t) + sizeof(cuda::std::uint32_t)) +
    host_indices.size() * sizeof(cuda::std::uint16_t);

  REQUIRE(bitmap.size() == host_indices.size());
  REQUIRE(bitmap.size_bytes() == expected_size);
  require_contains(bitmap, host_indices, std::vector<bool>(host_indices.size(), true));

  std::vector<index_type> absent;
  absent.reserve(cardinalities.size());
  for (cuda::std::size_t container = 0; container < cardinalities.size(); ++container) {
    absent.push_back((static_cast<index_type>(container) << 16) | cardinalities[container]);
  }
  require_contains(bitmap, absent, std::vector<bool>(absent.size(), false));
}

TEST_CASE("roaring_bitmap writes array and bitset containers in one grid", "[roaring_bitmap]")
{
  constexpr cuda::std::uint32_t max_array_cardinality = 4096;
  constexpr cuda::std::uint32_t bitset_bytes          = 8192;
  std::vector<cuda::std::uint32_t> const cardinalities{1, 2, 3, 4, 5, 6, 7, 8, 9, 4097, 5000};
  std::vector<index_type> host_indices;

  for (cuda::std::size_t container = 0; container < cardinalities.size(); ++container) {
    for (cuda::std::uint32_t lower = 0; lower < cardinalities[container]; ++lower) {
      host_indices.push_back((static_cast<index_type>(container) << 16) | lower);
    }
  }

  thrust::device_vector<index_type> indices(host_indices);
  auto bitmap = bitmap_type::from_sorted_unique_indices(indices.begin(), indices.end());

  auto expected_size =
    2 * sizeof(cuda::std::uint32_t) +
    cardinalities.size() * (2 * sizeof(cuda::std::uint16_t) + sizeof(cuda::std::uint32_t));
  for (auto const cardinality : cardinalities) {
    expected_size += cardinality <= max_array_cardinality
                       ? cardinality * sizeof(cuda::std::uint16_t)
                       : bitset_bytes;
  }

  REQUIRE(bitmap.size() == host_indices.size());
  REQUIRE(bitmap.size_bytes() == expected_size);
  require_contains(bitmap, host_indices, std::vector<bool>(host_indices.size(), true));

  std::vector<index_type> absent;
  absent.reserve(cardinalities.size());
  for (cuda::std::size_t container = 0; container < cardinalities.size(); ++container) {
    absent.push_back((static_cast<index_type>(container) << 16) | cardinalities[container]);
  }
  require_contains(bitmap, absent, std::vector<bool>(absent.size(), false));
}

TEST_CASE("roaring_bitmap selects array and bitset containers at the format threshold",
          "[roaring_bitmap]")
{
  SECTION("array container")
  {
    std::vector<index_type> host_indices(4096);
    for (index_type i = 0; i < host_indices.size(); ++i) {
      host_indices[i] = 4095 - i;
    }
    thrust::device_vector<index_type> indices(host_indices);
    auto bitmap = bitmap_type::from_indices(indices.begin(), indices.end());

    REQUIRE(bitmap.size() == 4096);
    require_contains(bitmap, {0, 4095, 4096}, {true, true, false});
  }

  SECTION("bitset container")
  {
    std::vector<index_type> host_indices(4097);
    for (index_type i = 0; i < host_indices.size(); ++i) {
      host_indices[i] = 4096 - i;
    }
    thrust::device_vector<index_type> indices(host_indices);
    auto bitmap = bitmap_type::from_indices(indices.begin(), indices.end());

    REQUIRE(bitmap.size() == 4097);
    require_contains(bitmap, {0, 4096, 4097}, {true, true, false});
  }

  SECTION("sorted bitset container with duplicates")
  {
    std::vector<index_type> host_indices;
    host_indices.reserve(4099);
    for (index_type i = 0; i < 4097; ++i) {
      host_indices.push_back(i);
      if (i == 2048 || i == 4096) { host_indices.push_back(i); }
    }
    thrust::device_vector<index_type> indices(host_indices);
    auto bitmap = bitmap_type::from_sorted_indices(indices.begin(), indices.end());

    REQUIRE(bitmap.size() == 4097);
    require_contains(bitmap, {0, 2048, 4096, 4097}, {true, true, true, false});
  }
}

TEST_CASE("roaring_bitmap writes a bitset after an odd-sized array container", "[roaring_bitmap]")
{
  std::vector<index_type> host_indices;
  host_indices.reserve(4098);
  host_indices.push_back(1);
  for (index_type i = 0; i < 4097; ++i) {
    host_indices.push_back(0x00010000 + i);
  }
  std::reverse(host_indices.begin(), host_indices.end());

  thrust::device_vector<index_type> indices(host_indices);
  auto bitmap = bitmap_type::from_indices(indices.begin(), indices.end());

  auto const bytes = copy_serialized(bitmap);
  cuda::std::uint32_t second_offset;
  std::memcpy(&second_offset, bytes.data() + 20, sizeof(second_offset));
  REQUIRE(second_offset % alignof(cuda::std::uint64_t) == 2);
  require_contains(
    bitmap, {1, 2, 0x00010000, 0x00011000, 0x00011001}, {true, false, true, true, false});
}

TEST_CASE("roaring_bitmap accepts transformed input and removes duplicates", "[roaring_bitmap]")
{
  auto const first = cuda::make_transform_iterator(
    cuda::counting_iterator<cuda::std::uint64_t>{0},
    cuda::proclaim_return_type<index_type>([] __device__(cuda::std::uint64_t index) {
      return static_cast<index_type>((31 - index) / 2);
    }));

  auto bitmap = bitmap_type::from_indices(first, first + 32);
  REQUIRE(bitmap.size() == 16);
  require_contains(bitmap, {0, 15, 16}, {true, true, false});
}

TEST_CASE("roaring_bitmap factories produce identical serialized bytes", "[roaring_bitmap]")
{
  thrust::device_vector<index_type> unordered{0x00010002, 7, 1, 0x00010000, 7, 3, 1};
  thrust::device_vector<index_type> sorted{1, 1, 3, 7, 7, 0x00010000, 0x00010002};
  thrust::device_vector<index_type> sorted_unique{1, 3, 7, 0x00010000, 0x00010002};
  auto const original = thrust::host_vector<index_type>{unordered};

  auto from_indices = bitmap_type::from_indices(unordered.begin(), unordered.end());
  auto from_sorted  = bitmap_type::from_sorted_indices(sorted.begin(), sorted.end());
  auto from_sorted_unique =
    bitmap_type::from_sorted_unique_indices(sorted_unique.begin(), sorted_unique.end());

  REQUIRE(copy_serialized(from_indices) == copy_serialized(from_sorted));
  REQUIRE(copy_serialized(from_indices) == copy_serialized(from_sorted_unique));
  REQUIRE(thrust::host_vector<index_type>{unordered} == original);
}

TEST_CASE("roaring_bitmap accepts a transformed sorted unique range", "[roaring_bitmap]")
{
  auto const first = cuda::make_transform_iterator(
    cuda::counting_iterator<cuda::std::uint64_t>{0},
    cuda::proclaim_return_type<index_type>(
      [] __device__(cuda::std::uint64_t index) { return static_cast<index_type>(2 * index); }));

  auto bitmap = bitmap_type::from_sorted_unique_indices(first, first + 16);
  REQUIRE(bitmap.size() == 16);
  require_contains(bitmap, {0, 2, 30, 31}, {true, true, true, false});
}

TEST_CASE("roaring_bitmap factories use the supplied allocator", "[roaring_bitmap]")
{
  SECTION("from_indices")
  {
    auto counts = std::make_shared<allocation_counts>();
    tracking_allocator<cuda::std::byte> allocator{counts};
    using tracked_bitmap = cuco::experimental::roaring_bitmap<index_type, decltype(allocator)>;
    thrust::device_vector<index_type> indices{9, 4, 9, 1, 7};

    {
      auto bitmap = tracked_bitmap::from_indices(indices.begin(), indices.end(), allocator);
      REQUIRE(bitmap.size() == 4);
      REQUIRE(bitmap.allocator() == allocator);
      REQUIRE(counts->allocations == 6);
      REQUIRE(counts->deallocations == 5);
    }
    REQUIRE(counts->deallocations == counts->allocations);
  }

  SECTION("from_sorted_indices")
  {
    auto counts = std::make_shared<allocation_counts>();
    tracking_allocator<cuda::std::byte> allocator{counts};
    using tracked_bitmap = cuco::experimental::roaring_bitmap<index_type, decltype(allocator)>;
    thrust::device_vector<index_type> indices{1, 4, 7, 9, 9};

    {
      auto bitmap = tracked_bitmap::from_sorted_indices(indices.begin(), indices.end(), allocator);
      REQUIRE(bitmap.size() == 4);
      REQUIRE(bitmap.allocator() == allocator);
      REQUIRE(counts->allocations == 6);
      REQUIRE(counts->deallocations == 5);
    }
    REQUIRE(counts->deallocations == counts->allocations);
  }

  SECTION("from_sorted_unique_indices")
  {
    auto counts = std::make_shared<allocation_counts>();
    tracking_allocator<cuda::std::byte> allocator{counts};
    using tracked_bitmap = cuco::experimental::roaring_bitmap<index_type, decltype(allocator)>;
    thrust::device_vector<index_type> indices{1, 4, 7, 9};

    {
      auto bitmap =
        tracked_bitmap::from_sorted_unique_indices(indices.begin(), indices.end(), allocator);
      REQUIRE(bitmap.size() == 4);
      REQUIRE(bitmap.allocator() == allocator);
      REQUIRE(counts->allocations == 5);
      REQUIRE(counts->deallocations == 4);
    }
    REQUIRE(counts->deallocations == counts->allocations);
  }
}

TEST_CASE("roaring_bitmap build serialization is stream ordered", "[roaring_bitmap]")
{
  cudaStream_t stream;
  CUCO_CUDA_TRY(cudaStreamCreate(&stream));

  {
    thrust::device_vector<index_type> indices{5, 4, 3, 2, 1};
    auto bitmap =
      bitmap_type::from_indices(indices.begin(), indices.end(), {}, cuda::stream_ref{stream});

    thrust::device_vector<index_type> queries{1, 5, 6};
    thrust::device_vector<bool> results(queries.size());
    bitmap.contains_async(
      queries.begin(), queries.end(), results.begin(), cuda::stream_ref{stream});
    CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

    thrust::host_vector<bool> host_results = results;
    REQUIRE(host_results[0]);
    REQUIRE(host_results[1]);
    REQUIRE_FALSE(host_results[2]);
  }

  CUCO_CUDA_TRY(cudaStreamDestroy(stream));
}

TEST_CASE("roaring_bitmap build supports cross-stream event handoff", "[roaring_bitmap]")
{
  cudaStream_t build_stream;
  cudaStream_t consume_stream;
  cudaEvent_t ready;
  CUCO_CUDA_TRY(cudaStreamCreate(&build_stream));
  CUCO_CUDA_TRY(cudaStreamCreate(&consume_stream));
  CUCO_CUDA_TRY(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));

  {
    thrust::device_vector<index_type> indices{1, 3, 5, 7};
    auto bitmap = bitmap_type::from_sorted_unique_indices(
      indices.begin(), indices.end(), {}, cuda::stream_ref{build_stream});

    CUCO_CUDA_TRY(cudaEventRecord(ready, build_stream));
    CUCO_CUDA_TRY(cudaStreamWaitEvent(consume_stream, ready));

    thrust::device_vector<index_type> queries{1, 2, 7};
    thrust::device_vector<bool> results(queries.size());
    bitmap.contains_async(
      queries.begin(), queries.end(), results.begin(), cuda::stream_ref{consume_stream});
    CUCO_CUDA_TRY(cudaStreamSynchronize(consume_stream));

    thrust::host_vector<bool> host_results = results;
    REQUIRE(host_results[0]);
    REQUIRE_FALSE(host_results[1]);
    REQUIRE(host_results[2]);
  }

  CUCO_CUDA_TRY(cudaStreamSynchronize(build_stream));
  CUCO_CUDA_TRY(cudaEventDestroy(ready));
  CUCO_CUDA_TRY(cudaStreamDestroy(consume_stream));
  CUCO_CUDA_TRY(cudaStreamDestroy(build_stream));
}
