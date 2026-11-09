/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/static_map.cuh>
#include <cuco/static_multimap.cuh>
#include <cuco/static_multiset.cuh>
#include <cuco/static_set.cuh>

#include <cuda/std/functional>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <type_traits>

namespace {
using key_type       = int;
using mapped_type    = int;
using hasher         = cuco::default_hash_function<key_type>;
using probing_scheme = cuco::linear_probing<1, hasher>;
using key_equal      = cuda::std::equal_to<key_type>;

constexpr key_type empty_key       = -1;
constexpr key_type erased_key      = -2;
constexpr mapped_type empty_mapped = -3;

template <typename Container, typename = void>
struct is_map : std::false_type {};

template <typename Container>
struct is_map<Container, std::void_t<typename Container::mapped_type>> : std::true_type {};

template <typename Container>
auto make_ref()
{
  using storage_ref_type = typename Container::storage_ref_type;
  auto const storage_ref =
    storage_ref_type{cuco::make_valid_extent<probing_scheme, cuco::storage<Container::bucket_size>>(
                       std::size_t{1024}),
                     nullptr};
  using ref_type = typename Container::template ref_type<>;
  if constexpr (is_map<Container>::value) {
    return ref_type{cuco::empty_key<key_type>{empty_key},
                    cuco::empty_value<mapped_type>{empty_mapped},
                    cuco::erased_key<key_type>{erased_key},
                    key_equal{},
                    probing_scheme{},
                    {},
                    storage_ref};
  } else {
    return ref_type{cuco::empty_key<key_type>{empty_key},
                    cuco::erased_key<key_type>{erased_key},
                    key_equal{},
                    probing_scheme{},
                    {},
                    storage_ref};
  }
}

template <typename Container, typename Ref>
void check_sentinels(Ref const& ref)
{
  REQUIRE(ref.empty_key_sentinel() == empty_key);
  REQUIRE(ref.erased_key_sentinel() == erased_key);
  if constexpr (is_map<Container>::value) { REQUIRE(ref.empty_value_sentinel() == empty_mapped); }
}
}  // namespace

TEMPLATE_TEST_CASE("Ref rebinds preserve sentinels",
                   "",
                   (cuco::static_set<key_type,
                                     cuco::extent<std::size_t>,
                                     cuda::thread_scope_device,
                                     key_equal,
                                     probing_scheme>),
                   (cuco::static_map<key_type,
                                     mapped_type,
                                     cuco::extent<std::size_t>,
                                     cuda::thread_scope_device,
                                     key_equal,
                                     probing_scheme>),
                   (cuco::static_multiset<key_type,
                                          cuco::extent<std::size_t>,
                                          cuda::thread_scope_device,
                                          key_equal,
                                          probing_scheme>),
                   (cuco::static_multimap<key_type,
                                          mapped_type,
                                          cuco::extent<std::size_t>,
                                          cuda::thread_scope_device,
                                          key_equal,
                                          probing_scheme>))
{
  auto const ref = make_ref<TestType>();
  check_sentinels<TestType>(ref);
  check_sentinels<TestType>(ref.rebind_operators(cuco::op::insert));
  check_sentinels<TestType>(ref.rebind_key_eq(key_equal{}));
  check_sentinels<TestType>(ref.rebind_hash_function(hasher{}));
}
