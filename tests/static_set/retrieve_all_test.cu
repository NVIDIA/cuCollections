/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <catch2/catch_template_test_macros.hpp>

template <typename Set>
__inline__ void test_unique_sequence(Set& set, std::size_t num_keys)
{
  using Key = typename Set::key_type;

  thrust::device_vector<Key> d_keys(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());

  auto key_begin = d_keys.begin();

  SECTION("Non-inserted keys should not be contained.")
  {
    REQUIRE(set.size() == 0);

    auto key_end = set.retrieve_all(key_begin);
    REQUIRE(std::distance(key_begin, key_end) == 0);
  }

  set.insert(key_begin, key_begin + num_keys);
  REQUIRE(set.size() == num_keys);

  SECTION("All inserted key/value pairs should be contained.")
  {
    thrust::device_vector<Key> d_res(num_keys);
    auto d_res_end = set.retrieve_all(d_res.begin());
    thrust::sort(thrust::device, d_res.begin(), d_res_end);
    REQUIRE(cuco::test::equal(
      d_res.begin(), d_res_end, thrust::counting_iterator<Key>(0), thrust::equal_to<Key>{}));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Retrieve all",
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
  constexpr std::size_t num_keys{400};
  auto constexpr gold_capacity = CGSize == 1 ? 409  // 409 x 1 x 1
                                             : 422  // 211 x 2 x 1
    ;

  using extent_type    = cuco::experimental::extent<std::size_t>;
  using allocator_type = cuco::cuda_allocator<std::byte>;
  using storage_type   = cuco::experimental::aow_storage<1>;

  if constexpr (Probe == cuco::test::probe_sequence::linear_probing) {
    using probe = cuco::experimental::linear_probing<CGSize, cuco::murmurhash3_32<Key>>;
    auto set    = cuco::experimental::static_set<Key,
                                              extent_type,
                                              cuda::thread_scope_device,
                                              thrust::equal_to<Key>,
                                              probe,
                                              allocator_type,
                                              storage_type>{num_keys, cuco::empty_key<Key>{-1}};

    REQUIRE(set.capacity() == gold_capacity);

    test_unique_sequence(set, num_keys);
  }

  if constexpr (Probe == cuco::test::probe_sequence::double_hashing) {
    using probe = cuco::experimental::
      double_hashing<CGSize, cuco::murmurhash3_32<Key>, cuco::murmurhash3_32<Key>>;
    auto set = cuco::experimental::static_set<Key,
                                              extent_type,
                                              cuda::thread_scope_device,
                                              thrust::equal_to<Key>,
                                              probe,
                                              allocator_type,
                                              storage_type>{num_keys, cuco::empty_key<Key>{-1}};

    REQUIRE(set.capacity() == gold_capacity);

    test_unique_sequence(set, num_keys);
  }
}
