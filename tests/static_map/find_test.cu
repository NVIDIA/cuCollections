/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <test_utils.hpp>

#include <cuco/static_map.cuh>

#include <cuda/functional>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <catch2/catch_template_test_macros.hpp>

using size_type = int32_t;

int32_t constexpr SENTINEL = -1;

template <typename Map>
void test_unique_sequence(Map& map, size_type num_keys)
{
  using Key   = typename Map::key_type;
  using Value = typename Map::mapped_type;

  auto keys_begin  = thrust::counting_iterator<Key>{0};
  auto pairs_begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0),
    cuda::proclaim_return_type<cuco::pair<Key, Value>>(
      [] __device__(auto i) { return cuco::pair<Key, Value>{i, i}; }));

  auto zip_equal = cuda::proclaim_return_type<bool>(
    [] __device__(auto const& p) { return thrust::get<0>(p) == thrust::get<1>(p); });
  auto is_even =
    cuda::proclaim_return_type<bool>([] __device__(auto const& i) { return i % 2 == 0; });

  SECTION("Non-inserted keys have no matches")
  {
    thrust::device_vector<Value> d_results(num_keys);

    map.find(keys_begin, keys_begin + num_keys, d_results.begin());
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(
      d_results.begin(), thrust::constant_iterator<Key>{map.empty_key_sentinel()}));

    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));
  }

  map.insert(pairs_begin, pairs_begin + num_keys);

  SECTION("All inserted keys should be correctly recovered during find")
  {
    thrust::device_vector<Value> d_results(num_keys);

    map.find(keys_begin, keys_begin + num_keys, d_results.begin());
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(d_results.begin(), keys_begin));

    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));
  }

  SECTION("Conditional find should return valid values on even inputs.")
  {
    auto found_results = thrust::device_vector<Key>(num_keys);
    auto gold_fn       = cuda::proclaim_return_type<Value>([] __device__(auto const& i) {
      return i % 2 == 0 ? static_cast<Value>(i) : Value{SENTINEL};
    });

    map.find_if(keys_begin,
                keys_begin + num_keys,
                thrust::counting_iterator<std::size_t>{0},
                is_even,
                found_results.begin());

    REQUIRE(cuco::test::equal(
      found_results.begin(),
      found_results.end(),
      thrust::make_transform_iterator(thrust::counting_iterator<Key>{0}, gold_fn),
      cuda::proclaim_return_type<bool>(
        [] __device__(auto const& found, auto const& gold) { return found == gold; })));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_map: find tests",
  "",
  ((typename Key, typename Value, cuco::test::probe_sequence Probe, int CGSize),
   Key,
   Value,
   Probe,
   CGSize),
  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int32_t, int64_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, int64_t, cuco::test::probe_sequence::linear_probing, 2))
{
  constexpr size_type num_keys{400};
  constexpr size_type gold_capacity = CGSize == 1 ? 422   // 211 x 1 x 2
                                                  : 412;  // 103 x 2 x 2

  // XXX: testing static extent is intended, DO NOT CHANGE
  using extent_type = cuco::extent<size_type, num_keys>;
  using probe       = std::conditional_t<
          Probe == cuco::test::probe_sequence::linear_probing,
          cuco::linear_probing<CGSize, cuco::murmurhash3_32<Key>>,
          cuco::double_hashing<CGSize, cuco::murmurhash3_32<Key>, cuco::murmurhash3_32<Key>>>;

  auto map = cuco::static_map<Key,
                              Value,
                              extent_type,
                              cuda::thread_scope_device,
                              thrust::equal_to<Key>,
                              probe,
                              cuco::cuda_allocator<cuda::std::byte>,
                              cuco::storage<2>>{
    extent_type{}, cuco::empty_key<Key>{SENTINEL}, cuco::empty_value<Value>{SENTINEL}};

  REQUIRE(map.capacity() == gold_capacity);

  test_unique_sequence(map, num_keys);
}
