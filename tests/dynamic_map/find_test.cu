/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/__config>
#include <cuco/dynamic_map.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE_SIG("dynamic_map find tests",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       // Robin Hood (hard-wired into static_map, which dynamic_map's submaps are)
                       // needs a single-CAS slot: padded int32/int64 and int64/int32 (12B) are
                       // unsupported; int64/int64 (16B) needs 128-bit atomics.
                       (int32_t, int32_t)
// (int32_t, int64_t),
// (int64_t, int32_t),
#if defined(CUCO_HAS_128BIT_ATOMICS)
                         ,
                       (int64_t, int64_t)
#endif
)
{
  constexpr std::size_t num_keys = 1'000'000;
  cuco::dynamic_map<Key, Value> map{num_keys * 2,
                                    cuco::empty_key<Key>{-1},
                                    cuco::empty_value<Value>{-1},
                                    cuco::erased_key<Key>{-2}};

  SECTION("Check single submap insert/find")
  {
    thrust::device_vector<Key> d_keys(num_keys);
    thrust::device_vector<Value> d_values(num_keys);
    thrust::device_vector<Value> d_found_values(num_keys);

    thrust::sequence(thrust::device, d_keys.begin(), d_keys.end(), 1);
    thrust::sequence(thrust::device, d_values.begin(), d_values.end(), 1);

    auto pairs_begin = cuda::make_transform_iterator(
      cuda::make_counting_iterator<std::size_t>(0),
      cuda::proclaim_return_type<cuco::pair<Key, Value>>(
        [keys = d_keys.begin(), values = d_values.begin()] __device__(auto i) {
          return cuco::pair<Key, Value>{keys[i], values[i]};
        }));

    map.insert(pairs_begin, pairs_begin + num_keys);

    REQUIRE(map.size() == num_keys);

    map.find(d_keys.begin(), d_keys.end(), d_found_values.begin());

    auto zip_equal = cuda::proclaim_return_type<bool>(
      [] __device__(auto const& p) { return cuda::std::get<0>(p) == cuda::std::get<1>(p); });
    auto zip =
      thrust::make_zip_iterator(cuda::std::tuple{d_values.begin(), d_found_values.begin()});
    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));

    thrust::device_vector<Key> d_nonexistent_keys(100);
    thrust::device_vector<Value> d_nonexistent_values(100);

    thrust::sequence(thrust::device,
                     d_nonexistent_keys.begin(),
                     d_nonexistent_keys.end(),
                     static_cast<Key>(num_keys + 1));

    map.find(d_nonexistent_keys.begin(), d_nonexistent_keys.end(), d_nonexistent_values.begin());

    auto empty_zip = thrust::make_zip_iterator(
      cuda::std::tuple{d_nonexistent_values.begin(),
                       cuda::constant_iterator<Value>{cuco::empty_value<Value>{-1}.value}});
    REQUIRE(cuco::test::all_of(empty_zip, empty_zip + 100, zip_equal));

    thrust::device_vector<Key> d_mixed_keys(200);
    thrust::device_vector<Value> d_mixed_values(200);

    thrust::copy(d_keys.begin(), d_keys.begin() + 100, d_mixed_keys.begin());
    thrust::sequence(thrust::device,
                     d_mixed_keys.begin() + 100,
                     d_mixed_keys.end(),
                     static_cast<Key>(num_keys + 1));

    map.find(d_mixed_keys.begin(), d_mixed_keys.end(), d_mixed_values.begin());

    auto first_half_zip =
      thrust::make_zip_iterator(cuda::std::tuple{d_values.begin(), d_mixed_values.begin()});
    REQUIRE(cuco::test::all_of(first_half_zip, first_half_zip + 100, zip_equal));

    auto second_half_empty_zip = thrust::make_zip_iterator(
      cuda::std::tuple{d_mixed_values.begin() + 100,
                       cuda::constant_iterator<Value>{cuco::empty_value<Value>{-1}.value}});
    REQUIRE(cuco::test::all_of(second_half_empty_zip, second_half_empty_zip + 100, zip_equal));
  }

  SECTION("Check find after erase")
  {
    thrust::device_vector<Key> d_keys(num_keys);
    thrust::device_vector<Value> d_values(num_keys);
    thrust::device_vector<Value> d_found_values(num_keys);

    thrust::sequence(thrust::device, d_keys.begin(), d_keys.end(), 1);
    thrust::sequence(thrust::device, d_values.begin(), d_values.end(), 1);

    auto pairs_begin = cuda::make_transform_iterator(
      cuda::make_counting_iterator<std::size_t>(0),
      cuda::proclaim_return_type<cuco::pair<Key, Value>>(
        [keys = d_keys.begin(), values = d_values.begin()] __device__(auto i) {
          return cuco::pair<Key, Value>{keys[i], values[i]};
        }));

    map.insert(pairs_begin, pairs_begin + num_keys);

    REQUIRE(map.size() == num_keys);

    map.find(d_keys.begin(), d_keys.end(), d_found_values.begin());

    auto zip_equal = cuda::proclaim_return_type<bool>(
      [] __device__(auto const& p) { return cuda::std::get<0>(p) == cuda::std::get<1>(p); });
    auto zip =
      thrust::make_zip_iterator(cuda::std::tuple{d_values.begin(), d_found_values.begin()});
    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));

    map.erase(d_keys.begin(), d_keys.begin() + num_keys / 2);

    REQUIRE(map.size() == num_keys / 2);

    map.find(d_keys.begin(), d_keys.end(), d_found_values.begin());

    auto first_half_empty_zip = thrust::make_zip_iterator(cuda::std::tuple{
      d_found_values.begin(), cuda::constant_iterator<Value>{cuco::empty_value<Value>{-1}.value}});
    REQUIRE(
      cuco::test::all_of(first_half_empty_zip, first_half_empty_zip + num_keys / 2, zip_equal));

    auto second_half_zip = thrust::make_zip_iterator(
      cuda::std::tuple{d_values.begin() + num_keys / 2, d_found_values.begin() + num_keys / 2});
    REQUIRE(cuco::test::all_of(second_half_zip, second_half_zip + num_keys / 2, zip_equal));
  }
}
