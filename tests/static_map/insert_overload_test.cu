/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>

#include <catch2/catch_template_test_macros.hpp>

template <typename MapRef, typename Key, typename Value>
__global__ void insert_test_kernel(MapRef map_ref, Key key, Value value, bool* inserted)
{
  *inserted = map_ref.insert({key, value});
}

template <typename MapRef, typename Key, typename Value>
__global__ void insert_and_find_test_kernel(
  MapRef map_ref, Key key, Value value, bool* inserted, Key* found_key, Value* found_value)
{
  auto [iter, success] = map_ref.insert_and_find({key, value});

  *inserted    = success;
  *found_key   = iter->first;
  *found_value = iter->second;
}

TEMPLATE_TEST_CASE_SIG("static_map insert and insert_and_find value_type overloads",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t),
                       (int32_t, int64_t),
                       (int64_t, int32_t),
                       (int64_t, int64_t))
{
  using probing_scheme = cuco::linear_probing<1, cuco::default_hash_function<Key>>;

  using map_type = cuco::static_map<Key,
                                    Value,
                                    cuco::extent<std::size_t>,
                                    cuda::thread_scope_device,
                                    cuda::std::equal_to<Key>,
                                    probing_scheme>;

  map_type map{10, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};

  auto map_ref = map.ref(cuco::insert);

  thrust::device_vector<bool> inserted(1, false);

  SECTION("insert accepts a brace-initialized value_type")
  {
    insert_test_kernel<<<1, 1>>>(map_ref, Key{42}, Value{123}, inserted.data().get());

    REQUIRE(cuco::test::all_of(inserted.begin(), inserted.end(), cuda::std::identity{}));

    // The same key should not be inserted twice.
    insert_test_kernel<<<1, 1>>>(map_ref, Key{42}, Value{456}, inserted.data().get());

    REQUIRE(cuco::test::none_of(inserted.begin(), inserted.end(), cuda::std::identity{}));
  }

  SECTION("insert_and_find accepts a brace-initialized value_type")
  {
    auto insert_and_find_ref = map.ref(cuco::insert_and_find);

    thrust::device_vector<Key> found_key(1);
    thrust::device_vector<Value> found_value(1);

    insert_and_find_test_kernel<<<1, 1>>>(insert_and_find_ref,
                                          Key{42},
                                          Value{123},
                                          inserted.data().get(),
                                          found_key.data().get(),
                                          found_value.data().get());

    REQUIRE(cuco::test::all_of(inserted.begin(), inserted.end(), cuda::std::identity{}));

    REQUIRE(found_key[0] == Key{42});
    REQUIRE(found_value[0] == Value{123});

    // The second insertion should find the existing element.
    insert_and_find_test_kernel<<<1, 1>>>(insert_and_find_ref,
                                          Key{42},
                                          Value{456},
                                          inserted.data().get(),
                                          found_key.data().get(),
                                          found_value.data().get());

    REQUIRE(cuco::test::none_of(inserted.begin(), inserted.end(), cuda::std::identity{}));

    // The existing value should not have been replaced.
    REQUIRE(found_key[0] == Key{42});
    REQUIRE(found_value[0] == Value{123});
  }
}
