/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/static_map.cuh>

#include <cuda/std/utility>
#include <thrust/device_vector.h>

#include <catch2/catch_template_test_macros.hpp>

template <std::size_t CGSize, typename MapRef>
__global__ void insert_test_kernel(MapRef map_ref, bool* inserted)
{
  auto const g = cuco::test::cg::this_thread_block();

  if constexpr (CGSize == 1) {
    *inserted = map_ref.insert({42, 123});
  } else {
    auto const tile = cuco::test::cg::tiled_partition<CGSize>(g);

    auto const success = map_ref.insert(tile, {42, 123});

    if (tile.thread_rank() == 0) {
      *inserted = success;
    }
  }
}

template <std::size_t CGSize, typename MapRef>
__global__ void insert_and_find_test_kernel(MapRef map_ref, bool* inserted, bool* value_correct)
{
  auto const g = cuco::test::cg::this_thread_block();

  if constexpr (CGSize == 1) {
    auto [iter, success] = map_ref.insert_and_find({42, 123});

    *inserted      = success;
    *value_correct = iter->first == 42 && iter->second == 123;
  } else {
    auto const tile = cuco::test::cg::tiled_partition<CGSize>(g);

    auto [iter, success] = map_ref.insert_and_find(tile, {42, 123});

    if (tile.thread_rank() == 0) {
      *inserted      = success;
      *value_correct = iter->first == 42 && iter->second == 123;
    }
  }
}

TEMPLATE_TEST_CASE_SIG("static_map insert and insert_and_find brace-initializer overloads",
                       "",
                       ((std::size_t CGSize), CGSize),
                       (1),
                       (2))
{
  using key_type   = int32_t;
  using value_type = int32_t;

  using probing_scheme =
    cuco::linear_probing<CGSize, cuco::default_hash_function<key_type>>;

  using map_type = cuco::static_map<key_type,
                                    value_type,
                                    cuco::extent<std::size_t>,
                                    cuda::thread_scope_device,
                                    cuda::std::equal_to<key_type>,
                                    probing_scheme>;

  map_type map{10, cuco::empty_key<key_type>{-1}, cuco::empty_value<value_type>{-1}};

  thrust::device_vector<bool> inserted(1, false);
  thrust::device_vector<bool> value_correct(1, false);

  auto insert_ref = map.ref(cuco::op::insert);

  SECTION("insert accepts a brace-initialized value_type")
  {
    insert_test_kernel<CGSize><<<1, CGSize>>>(
      insert_ref, inserted.data().get());

    REQUIRE(cuco::test::all_of(
      inserted.begin(), inserted.end(), cuda::std::identity{}));

    // The same key should not be inserted twice.
    insert_test_kernel<CGSize><<<1, CGSize>>>(
      insert_ref, inserted.data().get());

    REQUIRE(cuco::test::none_of(
      inserted.begin(), inserted.end(), cuda::std::identity{}));
  }

  SECTION("insert_and_find accepts a brace-initialized value_type")
  {
    auto insert_and_find_ref = map.ref(cuco::op::insert_and_find);

    insert_and_find_test_kernel<CGSize><<<1, CGSize>>>(
      insert_and_find_ref,
      inserted.data().get(),
      value_correct.data().get());

    REQUIRE(cuco::test::all_of(
      inserted.begin(), inserted.end(), cuda::std::identity{}));
    REQUIRE(cuco::test::all_of(
      value_correct.begin(), value_correct.end(), cuda::std::identity{}));

    // The second insertion should find the existing element.
    insert_and_find_test_kernel<CGSize><<<1, CGSize>>>(
      insert_and_find_ref,
      inserted.data().get(),
      value_correct.data().get());

    REQUIRE(cuco::test::none_of(
      inserted.begin(), inserted.end(), cuda::std::identity{}));
    REQUIRE(cuco::test::all_of(
      value_correct.begin(), value_correct.end(), cuda::std::identity{}));
  }
}
