/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/static_map.cuh>
#include <cuco/static_multimap.cuh>
#include <cuco/static_multiset.cuh>
#include <cuco/static_set.cuh>

#include <cuda/std/functional>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>

namespace {
using key_type       = int;
using mapped_type    = int;
using hasher         = cuco::default_hash_function<key_type>;
using probing_scheme = cuco::linear_probing<1, hasher>;
using key_equal      = cuda::std::equal_to<key_type>;

constexpr key_type empty_key       = -1;
constexpr key_type erased_key      = -2;
constexpr mapped_type empty_mapped = -3;

template <int BucketSize, typename Container>
auto make_storage_ref()
{
  using storage_ref_type = typename Container::storage_ref_type;
  return storage_ref_type{
    cuco::make_valid_extent<probing_scheme, cuco::storage<BucketSize>>(std::size_t{1024}), nullptr};
}

template <typename Ref>
void check_rebinds_preserve_sentinels(Ref const& ref)
{
  REQUIRE(ref.empty_key_sentinel() == empty_key);
  REQUIRE(ref.erased_key_sentinel() == erased_key);

  auto const with_operators = ref.rebind_operators(cuco::op::insert);
  REQUIRE(with_operators.empty_key_sentinel() == empty_key);
  REQUIRE(with_operators.erased_key_sentinel() == erased_key);

  auto const with_key_eq = ref.rebind_key_eq(key_equal{});
  REQUIRE(with_key_eq.empty_key_sentinel() == empty_key);
  REQUIRE(with_key_eq.erased_key_sentinel() == erased_key);

  auto const with_hash = ref.rebind_hash_function(hasher{});
  REQUIRE(with_hash.empty_key_sentinel() == empty_key);
  REQUIRE(with_hash.erased_key_sentinel() == erased_key);
}
}  // namespace

TEST_CASE("static_set_ref rebinds preserve sentinels", "")
{
  using set_type = cuco::static_set<key_type,
                                    cuco::extent<std::size_t>,
                                    cuda::thread_scope_device,
                                    key_equal,
                                    probing_scheme>;
  typename set_type::template ref_type<> ref{cuco::empty_key<key_type>{empty_key},
                                             cuco::erased_key<key_type>{erased_key},
                                             key_equal{},
                                             probing_scheme{},
                                             {},
                                             make_storage_ref<1, set_type>()};
  check_rebinds_preserve_sentinels(ref);
}

TEST_CASE("static_map_ref rebinds preserve sentinels", "")
{
  using map_type = cuco::static_map<key_type,
                                    mapped_type,
                                    cuco::extent<std::size_t>,
                                    cuda::thread_scope_device,
                                    key_equal,
                                    probing_scheme>;
  typename map_type::template ref_type<> ref{cuco::empty_key<key_type>{empty_key},
                                             cuco::empty_value<mapped_type>{empty_mapped},
                                             cuco::erased_key<key_type>{erased_key},
                                             key_equal{},
                                             probing_scheme{},
                                             {},
                                             make_storage_ref<1, map_type>()};
  check_rebinds_preserve_sentinels(ref);
  REQUIRE(ref.rebind_key_eq(key_equal{}).empty_value_sentinel() == empty_mapped);
}

TEST_CASE("static_multiset_ref rebinds preserve sentinels", "")
{
  using set_type = cuco::static_multiset<key_type,
                                         cuco::extent<std::size_t>,
                                         cuda::thread_scope_device,
                                         key_equal,
                                         probing_scheme>;
  typename set_type::template ref_type<> ref{cuco::empty_key<key_type>{empty_key},
                                             cuco::erased_key<key_type>{erased_key},
                                             key_equal{},
                                             probing_scheme{},
                                             {},
                                             make_storage_ref<2, set_type>()};
  check_rebinds_preserve_sentinels(ref);
}

TEST_CASE("static_multimap_ref rebinds preserve sentinels", "")
{
  using map_type = cuco::static_multimap<key_type,
                                         mapped_type,
                                         cuco::extent<std::size_t>,
                                         cuda::thread_scope_device,
                                         key_equal,
                                         probing_scheme>;
  typename map_type::template ref_type<> ref{cuco::empty_key<key_type>{empty_key},
                                             cuco::empty_value<mapped_type>{empty_mapped},
                                             cuco::erased_key<key_type>{erased_key},
                                             key_equal{},
                                             probing_scheme{},
                                             {},
                                             make_storage_ref<2, map_type>()};
  check_rebinds_preserve_sentinels(ref);
  REQUIRE(ref.rebind_key_eq(key_equal{}).empty_value_sentinel() == empty_mapped);
}
