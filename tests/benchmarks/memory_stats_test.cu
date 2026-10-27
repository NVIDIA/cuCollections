/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>
#include <memory_stats.hpp>

#include <cstdint>
#include <memory>
#include <new>

namespace {

// Exercise the accounting without requiring a CUDA device.
template <class T>
struct host_allocator {
  using value_type = T;

  host_allocator() = default;
  template <class U>
  host_allocator(host_allocator<U> const&) noexcept
  {
  }

  T* allocate(std::size_t count, cuda::stream_ref)
  {
    if (fail) { throw std::bad_alloc{}; }
    return std::allocator<T>{}.allocate(count);
  }

  void deallocate(T* pointer, std::size_t count, cuda::stream_ref)
  {
    std::allocator<T>{}.deallocate(pointer, count);
  }

  template <class U>
  bool operator==(host_allocator<U> const&) const noexcept
  {
    return true;
  }

  inline static bool fail = false;
};

template <class T>
using allocator = cuco::benchmark::tracking_allocator<T, host_allocator<T>>;
using cuco::benchmark::memory_stats_logger;

}  // namespace

TEST_CASE("Benchmark memory tracking excludes setup and counts outstanding bytes", "[benchmark]")
{
  auto alloc        = allocator<std::uint32_t>{};
  auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};
  auto* setup       = alloc.allocate(100, stream);

  auto const logger = memory_stats_logger{};
  REQUIRE(logger.peak_memory_usage() == 0);
  alloc.deallocate(setup, 100, stream);

  auto* first  = alloc.allocate(10, stream);
  auto* second = alloc.allocate(20, stream);
  REQUIRE(logger.peak_memory_usage() == 30 * sizeof(std::uint32_t));
  alloc.deallocate(first, 10, stream);
  auto* third = alloc.allocate(5, stream);
  REQUIRE(logger.peak_memory_usage() == 30 * sizeof(std::uint32_t));
  alloc.deallocate(second, 20, stream);
  alloc.deallocate(third, 5, stream);

  // NVBench repeats the measured operation: freed storage must not accumulate.
  for (int i = 0; i < 10; ++i) {
    auto* allocation = alloc.allocate(40, stream);
    alloc.deallocate(allocation, 40, stream);
  }
  REQUIRE(logger.peak_memory_usage() == 40 * sizeof(std::uint32_t));
}

TEST_CASE("Benchmark memory tracking follows copied and rebound allocators", "[benchmark]")
{
  auto alloc        = allocator<std::uint32_t>{};
  auto copy         = alloc;
  auto rebound      = std::allocator_traits<decltype(alloc)>::rebind_alloc<std::uint64_t>{alloc};
  auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};
  REQUIRE(alloc == rebound);
  REQUIRE_FALSE(alloc != rebound);

  auto const logger = memory_stats_logger{};
  auto* first       = copy.allocate(3, stream);
  auto* second      = rebound.allocate(5, stream);
  REQUIRE(logger.peak_memory_usage() == 3 * sizeof(std::uint32_t) + 5 * sizeof(std::uint64_t));
  alloc.deallocate(first, 3, stream);
  rebound.deallocate(second, 5, stream);
}

TEST_CASE("Benchmark memory tracking resets between scopes", "[benchmark]")
{
  auto alloc        = allocator<std::uint32_t>{};
  auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};
  std::uint32_t* previous;
  {
    auto const logger = memory_stats_logger{};
    previous          = alloc.allocate(100, stream);
    REQUIRE(logger.peak_memory_usage() == 100 * sizeof(std::uint32_t));
  }
  {
    auto const logger = memory_stats_logger{};
    alloc.deallocate(previous, 100, stream);
    auto* current = alloc.allocate(2, stream);
    REQUIRE(logger.peak_memory_usage() == 2 * sizeof(std::uint32_t));
    alloc.deallocate(current, 2, stream);
  }
  // Container destruction after the logging scope is also supported.
  auto* outside = alloc.allocate(3, stream);
  alloc.deallocate(outside, 3, stream);
}

TEST_CASE("Failed and zero-sized allocations do not increase benchmark peak memory", "[benchmark]")
{
  auto alloc                          = allocator<std::uint32_t>{};
  auto const stream                   = cuda::stream_ref{cudaStream_t{nullptr}};
  auto const logger                   = memory_stats_logger{};
  host_allocator<std::uint32_t>::fail = true;
  REQUIRE_THROWS_AS(alloc.allocate(10, stream), std::bad_alloc);
  host_allocator<std::uint32_t>::fail = false;
  auto* empty                         = alloc.allocate(0, stream);
  alloc.deallocate(empty, 0, stream);
  REQUIRE(logger.peak_memory_usage() == 0);
}

TEST_CASE("Nested benchmark memory scopes fail without losing the active scope", "[benchmark]")
{
  auto alloc        = allocator<std::uint32_t>{};
  auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};
  auto const logger = memory_stats_logger{};
  REQUIRE_THROWS_AS(memory_stats_logger{}, cuco::logic_error);
  auto* allocation = alloc.allocate(7, stream);
  REQUIRE(logger.peak_memory_usage() == 7 * sizeof(std::uint32_t));
  alloc.deallocate(allocation, 7, stream);
}
