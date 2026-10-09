/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/detail/utility/cuda.hpp>
#include <cuco/static_map.cuh>
#include <cuco/static_multimap.cuh>
#include <cuco/static_multiset.cuh>
#include <cuco/static_set.cuh>

#include <thrust/device_vector.h>

#include <cooperative_groups.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <type_traits>

namespace {

struct query_result {
  int comparisons{};
  int matches{};
};

struct constant_hash {
  std::size_t value;

  __host__ __device__ constexpr constant_hash(std::size_t value = 0) noexcept : value{value} {}

  __host__ __device__ constexpr std::size_t operator()(int) const noexcept { return value; }
};

struct counting_equal {
  query_result* result;

  __device__ bool operator()(int lhs, int rhs) const noexcept
  {
    atomicAdd(&result->comparisons, 1);
    return lhs == rhs;
  }
};

enum class query_op { for_each, count, find, contains };

template <bool IsMap, bool AllowsDuplicates, typename Probe, int BucketSize>
auto make_container(query_result* result)
{
  auto const equal = counting_equal{result};
  if constexpr (IsMap and AllowsDuplicates) {
    return cuco::static_multimap{128,
                                 cuco::empty_key<int>{-1},
                                 cuco::empty_value<int>{-1},
                                 equal,
                                 Probe{},
                                 {},
                                 cuco::storage<BucketSize>{}};
  } else if constexpr (IsMap) {
    return cuco::static_map{128,
                            cuco::empty_key<int>{-1},
                            cuco::empty_value<int>{-1},
                            equal,
                            Probe{},
                            {},
                            cuco::storage<BucketSize>{}};
  } else if constexpr (AllowsDuplicates) {
    return cuco::static_multiset{
      128, cuco::empty_key<int>{-1}, equal, Probe{}, {}, cuco::storage<BucketSize>{}};
  } else {
    return cuco::static_set{
      128, cuco::empty_key<int>{-1}, equal, Probe{}, {}, cuco::storage<BucketSize>{}};
  }
}

template <bool IsMap, bool AllowsDuplicates, class Ref>
__global__ void insert_collisions(Ref ref, int first, int last)
{
  auto const tile =
    cooperative_groups::tiled_partition<Ref::cg_size>(cooperative_groups::this_thread_block());

  // A single group inserts in order, making the first key's position deterministic. In multi
  // containers, the duplicates at offsets 0, 8 and 16 span several probing windows.
  for (int i = first; i < last; ++i) {
    int const key    = AllowsDuplicates and i % 8 == 0 ? 0 : i;
    auto const value = [&] {
      if constexpr (IsMap) {
        return cuco::pair<int, int>{key, i};
      } else {
        return key;
      }
    }();
    if constexpr (Ref::cg_size == 1) {
      ref.insert(value);
    } else {
      ref.insert(tile, value);
    }
  }
}

template <bool Cooperative, class Ref>
__global__ void query_collisions(Ref ref, query_op op, int key, query_result* result)
{
  auto const tile =
    cooperative_groups::tiled_partition<Ref::cg_size>(cooperative_groups::this_thread_block());
  auto const callback = [result] __device__(auto const&) { atomicAdd(&result->matches, 1); };

  switch (op) {
    case query_op::for_each:
      if constexpr (Cooperative) {
        ref.for_each(tile, key, callback);
      } else {
        ref.for_each(key, callback);
      }
      break;
    case query_op::count: {
      auto const count = [&] {
        if constexpr (Cooperative) {
          return ref.count(tile, key);
        } else {
          return ref.count(key);
        }
      }();
      // count returns each thread's contribution, rather than broadcasting the group total.
      atomicAdd(&result->matches, static_cast<int>(count));
      break;
    }
    case query_op::find: {
      auto const found = [&] {
        if constexpr (Cooperative) {
          return ref.find(tile, key) != ref.end();
        } else {
          return ref.find(key) != ref.end();
        }
      }();
      if (tile.thread_rank() == 0) { result->matches = found; }
      break;
    }
    case query_op::contains: {
      auto const found = [&] {
        if constexpr (Cooperative) {
          return ref.contains(tile, key);
        } else {
          return ref.contains(key);
        }
      }();
      if (tile.thread_rank() == 0) { result->matches = found; }
      break;
    }
  }
}

template <bool Cooperative, bool AllowsDuplicates, class Container>
void check_queries(Container& container,
                   thrust::device_vector<query_result>& results,
                   int key,
                   int expected_matches,
                   bool first_window)
{
  for (auto const op : {query_op::for_each, query_op::count, query_op::find, query_op::contains}) {
    auto const name = op == query_op::for_each ? "for_each"
                      : op == query_op::count  ? "count"
                      : op == query_op::find   ? "find"
                                               : "contains";
    CAPTURE(Cooperative, key, name);
    results[0] = query_result{};
    query_collisions<Cooperative><<<1, Container::cg_size>>>(
      container.ref(cuco::for_each, cuco::count, cuco::find, cuco::contains),
      op,
      key,
      results.data().get());
    CUCO_CUDA_TRY(cudaGetLastError());
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
    query_result const result = results[0];

    auto const single_match = op == query_op::find or op == query_op::contains;
    CHECK(result.matches == (single_match ? int{expected_matches != 0} : expected_matches));
    CHECK(result.comparisons > 0);
    if (first_window and (not AllowsDuplicates or single_match)) {
      // Ignore exact predicate counts within a bucket/group window. Appending collisions after
      // a unique match must not make any of these operations scan subsequent windows.
      CHECK(result.comparisons <= Container::cg_size * Container::bucket_size);
    }
  }
}

template <bool IsMap, bool AllowsDuplicates, typename Probe, int BucketSize>
void test_query_work()
{
  thrust::device_vector<query_result> results(1);
  auto container = make_container<IsMap, AllowsDuplicates, Probe, BucketSize>(results.data().get());
  auto const insert = [&](int first, int last) {
    insert_collisions<IsMap, AllowsDuplicates>
      <<<1, Probe::cg_size>>>(container.ref(cuco::insert), first, last);
    CUCO_CUDA_TRY(cudaGetLastError());
    CUCO_CUDA_TRY(cudaDeviceSynchronize());
  };
  auto const check = [&](int key, int expected_matches, bool first_window) {
    if constexpr (Probe::cg_size == 1) {
      check_queries<false, AllowsDuplicates>(
        container, results, key, expected_matches, first_window);
    }
    // Exercise the CG overload explicitly even with a one-thread group.
    check_queries<true, AllowsDuplicates>(container, results, key, expected_matches, first_window);
  };

  insert(0, 1);
  check(0, 1, true);
  insert(1, 17);
  check(0, AllowsDuplicates ? 3 : 1, true);
  // With bucket size 1 and CG size 2 this hit belongs to a nonzero lane.
  check(1, 1, Probe::cg_size * BucketSize > 1);
  check(AllowsDuplicates ? 15 : 16, 1, false);
  check(99, 0, false);
}

}  // namespace

TEMPLATE_TEST_CASE_SIG(
  "Query work with colliding keys",
  "",
  ((bool DoubleHashing, int CGSize, int BucketSize), DoubleHashing, CGSize, BucketSize),
  (false, 1, 1),
  (false, 1, 2),
  (false, 2, 1),
  (false, 2, 2),
  (true, 1, 1),
  (true, 1, 2),
  (true, 2, 1),
  (true, 2, 2))
{
  using probe = std::conditional_t<DoubleHashing,
                                   cuco::double_hashing<CGSize, constant_hash, constant_hash>,
                                   cuco::linear_probing<CGSize, constant_hash>>;
  SECTION("static_map") { test_query_work<true, false, probe, BucketSize>(); }
  SECTION("static_set") { test_query_work<false, false, probe, BucketSize>(); }
  SECTION("static_multimap") { test_query_work<true, true, probe, BucketSize>(); }
  SECTION("static_multiset") { test_query_work<false, true, probe, BucketSize>(); }
}
