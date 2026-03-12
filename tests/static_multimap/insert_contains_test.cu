/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include <cuco/static_multimap.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <catch2/catch_template_test_macros.hpp>

template <typename Map>
void test_insert(Map& map, std::size_t num_keys)
{
  using Key   = typename Map::key_type;
  using Value = typename Map::mapped_type;

  auto const keys_begin = cuda::counting_iterator<Key>{0};
  auto pairs_begin      = cuda::make_transform_iterator(
    cuda::make_counting_iterator(0),
    cuda::proclaim_return_type<cuco::pair<Key, Value>>(
      [] __device__(auto i) { return cuco::pair<Key, Value>{i, i}; }));
  thrust::device_vector<bool> d_contained(num_keys);

  SECTION("Non-inserted keys should not be contained.")
  {
    map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::none_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  }

  map.insert(pairs_begin, pairs_begin + num_keys);

  SECTION("All inserted keys should be contained.")
  {
    map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  }

  SECTION("Conditional contains should return true on even inputs.")
  {
    auto is_even =
      cuda::proclaim_return_type<bool>([] __device__(auto const& i) { return i % 2 == 0; });
    auto zip_equal = cuda::proclaim_return_type<bool>(
      [] __device__(auto const& p) { return cuda::std::get<0>(p) == cuda::std::get<1>(p); });

    map.contains_if(keys_begin,
                    keys_begin + num_keys,
                    cuda::counting_iterator<std::size_t>(0),
                    is_even,
                    d_contained.begin());
    auto gold_iter =
      cuda::make_transform_iterator(cuda::counting_iterator<std::size_t>(0), is_even);
    auto zip = thrust::make_zip_iterator(cuda::std::tuple{d_contained.begin(), gold_iter});
    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_multimap insert/contains test",
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
  constexpr std::size_t num_keys{4'000};

  using extent_type = cuco::extent<std::size_t>;
  using probe       = std::conditional_t<
          Probe == cuco::test::probe_sequence::linear_probing,
          cuco::linear_probing<CGSize, cuco::murmurhash3_32<Key>>,
          cuco::double_hashing<CGSize, cuco::murmurhash3_32<Key>, cuco::murmurhash3_32<Key>>>;

  auto map = cuco::static_multimap<Key,
                                   Value,
                                   extent_type,
                                   cuda::thread_scope_device,
                                   cuda::std::equal_to<Key>,
                                   probe,
                                   cuco::cuda_allocator<cuda::std::byte>,
                                   cuco::storage<2>>{
    extent_type{num_keys}, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};

  test_insert(map, num_keys);
}
