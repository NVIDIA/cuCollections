/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <utils.hpp>

#include <cuco/static_set.cuh>

#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cuda/functional>

#include <catch2/catch_template_test_macros.hpp>

using size_type = int32_t;

template <typename Set>
__inline__ void test_unique_sequence(Set& set, size_type num_keys)
{
  using Key = typename Set::key_type;

  thrust::device_vector<Key> d_keys(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());

  auto keys_begin = d_keys.begin();
  thrust::device_vector<bool> d_contained(num_keys);

  auto zip_equal = cuda::proclaim_return_type<bool>(
    [] __device__(auto const& p) { return thrust::get<0>(p) == thrust::get<1>(p); });
  auto is_even =
    cuda::proclaim_return_type<bool>([] __device__(auto const& i) { return i % 2 == 0; });

  SECTION("Non-inserted keys should not be contained.")
  {
    REQUIRE(set.size() == 0);

    set.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::none_of(d_contained.begin(), d_contained.end(), thrust::identity{}));
  }

  SECTION("Non-inserted keys have no matches")
  {
    thrust::device_vector<Key> d_results(num_keys);

    set.find(keys_begin, keys_begin + num_keys, d_results.begin());
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(
      d_results.begin(), thrust::constant_iterator<Key>{set.empty_key_sentinel()}));

    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));
  }

  SECTION("All conditionally inserted keys should be contained")
  {
    auto const inserted = set.insert_if(
      keys_begin, keys_begin + num_keys, thrust::counting_iterator<std::size_t>(0), is_even);
    REQUIRE(inserted == num_keys / 2);
    REQUIRE(set.size() == num_keys / 2);

    set.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::equal(
      d_contained.begin(),
      d_contained.end(),
      thrust::counting_iterator<std::size_t>(0),
      cuda::proclaim_return_type<bool>([] __device__(auto const& idx_contained, auto const& idx) {
        return ((idx % 2) == 0) == idx_contained;
      })));
  }

  set.insert(keys_begin, keys_begin + num_keys);
  REQUIRE(set.size() == num_keys);

  SECTION("All inserted keys should be contained.")
  {
    set.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), thrust::identity{}));
  }

  SECTION("Conditional contains should return true on even inputs.")
  {
    set.contains_if(keys_begin,
                    keys_begin + num_keys,
                    thrust::counting_iterator<std::size_t>(0),
                    is_even,
                    d_contained.begin());
    auto gold_iter =
      thrust::make_transform_iterator(thrust::counting_iterator<std::size_t>(0), is_even);
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(d_contained.begin(), gold_iter));
    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));
  }

  SECTION("All inserted keys should be correctly recovered during find")
  {
    thrust::device_vector<Key> d_results(num_keys);

    set.find(keys_begin, keys_begin + num_keys, d_results.begin());
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(d_results.begin(), keys_begin));

    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Unique sequence",
  "",
  ((typename Key, cuco::test::probe_sequence Probe, int CGSize), Key, Probe, CGSize),
  (int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, cuco::test::probe_sequence::linear_probing, 2))
{
  constexpr size_type num_keys{400};
  constexpr size_type gold_capacity = CGSize == 1 ? 422  // 211 x 1 x 2
                                                  : 412  // 103 x 2 x 2
    ;

  using probe = std::conditional_t<
    Probe == cuco::test::probe_sequence::linear_probing,
    cuco::experimental::linear_probing<CGSize, cuco::default_hash_function<Key>>,
    cuco::experimental::double_hashing<CGSize, cuco::default_hash_function<Key>>>;

  auto set = cuco::experimental::static_set<Key,
                                            cuco::experimental::extent<size_type>,
                                            cuda::thread_scope_device,
                                            thrust::equal_to<Key>,
                                            probe,
                                            cuco::cuda_allocator<std::byte>,
                                            cuco::experimental::aow_storage<2>>{
    num_keys, cuco::empty_key<Key>{-1}};

  REQUIRE(set.capacity() == gold_capacity);

  test_unique_sequence(set, num_keys);
}
