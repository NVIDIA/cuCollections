/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/static_set.cuh>

#include <cuda/std/functional>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("static_set capacity test", "")
{
  using Key        = int32_t;
  using ProbeT     = cuco::double_hashing<1, cuco::default_hash_function<Key>>;
  using Equal      = cuda::std::equal_to<Key>;
  using AllocatorT = cuco::cuda_allocator<cuda::std::byte>;
  using StorageT   = cuco::storage<2>;

  SECTION("zero capacity is allowed.")
  {
    auto constexpr gold_capacity = 4;

    using extent_type = cuco::extent<std::size_t, 0>;
    cuco::
      static_set<Key, extent_type, cuda::thread_scope_device, Equal, ProbeT, AllocatorT, StorageT>
        set{extent_type{}, cuco::empty_key<Key>{-1}};
    auto const capacity = set.capacity();
    REQUIRE(capacity == gold_capacity);

    auto ref                = set.ref(cuco::insert);
    auto const ref_capacity = ref.capacity();
    REQUIRE(ref_capacity == gold_capacity);
  }

  SECTION("negative capacity (ikr -_-||) is also allowed.")
  {
    auto constexpr gold_capacity = 4;

    using extent_type = cuco::extent<int32_t>;
    cuco::
      static_set<Key, extent_type, cuda::thread_scope_device, Equal, ProbeT, AllocatorT, StorageT>
        set{extent_type{-10}, cuco::empty_key<Key>{-1}};
    auto const capacity = set.capacity();
    REQUIRE(capacity == gold_capacity);

    auto ref                = set.ref(cuco::insert);
    auto const ref_capacity = ref.capacity();
    REQUIRE(ref_capacity == gold_capacity);
  }

  constexpr std::size_t num_keys{400};

  SECTION("Static bucket extent can be evaluated at build time.")
  {
    std::size_t constexpr gold_extent = 422;  // 211 x 2

    using extent_type = cuco::extent<std::size_t, num_keys>;
    cuco::
      static_set<Key, extent_type, cuda::thread_scope_device, Equal, ProbeT, AllocatorT, StorageT>
        set{extent_type{}, cuco::empty_key<Key>{-1}};

    auto ref               = set.ref(cuco::insert);
    auto const num_buckets = ref.bucket_extent();
    STATIC_REQUIRE(static_cast<std::size_t>(num_buckets) == gold_extent);
  }

  SECTION("Dynamic extent is evaluated at run time.")
  {
    auto constexpr gold_capacity = 422;  // 211 x 2

    using extent_type = cuco::extent<std::size_t>;
    cuco::
      static_set<Key, extent_type, cuda::thread_scope_device, Equal, ProbeT, AllocatorT, StorageT>
        set{num_keys, cuco::empty_key<Key>{-1}};
    auto const capacity = set.capacity();
    REQUIRE(capacity == gold_capacity);

    auto ref                = set.ref(cuco::insert);
    auto const ref_capacity = ref.capacity();
    REQUIRE(ref_capacity == gold_capacity);
  }

  SECTION("Set can be constructed from plain integer.")
  {
    auto constexpr gold_capacity = 422;  // 211 x 2

    cuco::
      static_set<Key, std::size_t, cuda::thread_scope_device, Equal, ProbeT, AllocatorT, StorageT>
        set{num_keys, cuco::empty_key<Key>{-1}};
    auto const capacity = set.capacity();
    REQUIRE(capacity == gold_capacity);

    auto ref                = set.ref(cuco::insert);
    auto const ref_capacity = ref.capacity();
    REQUIRE(ref_capacity == gold_capacity);
  }

  SECTION("Set can be constructed from plain integer and load factor.")
  {
    auto constexpr gold_capacity = 502;  // 251 x 2

    cuco::
      static_set<Key, std::size_t, cuda::thread_scope_device, Equal, ProbeT, AllocatorT, StorageT>
        set{num_keys, 0.8, cuco::empty_key<Key>{-1}};
    auto const capacity = set.capacity();
    REQUIRE(capacity == gold_capacity);

    auto ref                = set.ref(cuco::insert);
    auto const ref_capacity = ref.capacity();
    REQUIRE(ref_capacity == gold_capacity);
  }

  SECTION("Dynamic extent of linear probing is evaluated at run time.")
  {
    auto constexpr gold_capacity = 400;

    using probe = cuco::linear_probing<2, cuco::default_hash_function<Key>>;
    auto set    = cuco::static_set<Key,
                                   cuco::extent<std::size_t>,
                                   cuda::thread_scope_device,
                                   Equal,
                                   probe,
                                   AllocatorT,
                                   StorageT>{num_keys, cuco::empty_key<Key>{-1}};

    auto const capacity = set.capacity();
    REQUIRE(capacity == gold_capacity);

    auto ref                = set.ref(cuco::insert);
    auto const ref_capacity = ref.capacity();
    REQUIRE(ref_capacity == gold_capacity);
  }
}
