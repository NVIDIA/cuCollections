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

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <catch2/catch.hpp>

// Custom pair equal
template <typename Key, typename Value>
struct pair_equal {
  __device__ bool operator()(const cuco::pair_type<Key, Value>& lhs,
                             const cuco::pair_type<Key, Value>& rhs) const
  {
    return lhs.first == rhs.first;
  }
};

template <typename Key, typename Value, typename Map, typename PairIt>
__inline__ void test_pair_functions(Map& map, PairIt pair_begin, std::size_t num_pairs)
{
  map.insert(pair_begin, pair_begin + num_pairs);
  cudaStreamSynchronize(0);

  auto res = map.get_size();
  REQUIRE(res == num_pairs / 2);  // since multiplicity = 2

  // query pair matching rate = 50%
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    pair_begin,
                    [] __device__(auto i) {
                      return cuco::pair_type<Key, Value>{i, i};
                    });

  SECTION("pair_contains returns true for all inserted pairs and false for non-inserted ones.")
  {
    thrust::device_vector<bool> result(num_pairs);
    auto res_begin = result.begin();
    map.pair_contains(pair_begin,
                      pair_begin + num_pairs,
                      res_begin,
                      cuco::detail::MurmurHash3_32<Key>{},
                      pair_equal<Key, Value>{});

    auto true_iter  = thrust::make_constant_iterator(true);
    auto false_iter = thrust::make_constant_iterator(false);

    REQUIRE(
      cuco::test::equal(res_begin, res_begin + num_pairs / 2, true_iter, thrust::equal_to<bool>{}));
    REQUIRE(cuco::test::equal(
      res_begin + num_pairs / 2, res_begin + num_pairs, false_iter, thrust::equal_to<bool>{}));
  }
}

TEMPLATE_TEST_CASE_SIG("Tests of pair functions",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t),
                       (int32_t, int64_t),
                       (int64_t, int32_t),
                       (int64_t, int64_t))
{
  constexpr std::size_t num_pairs{4};
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(num_pairs);

  // pair multiplicity = 2
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    d_pairs.begin(),
                    [] __device__(auto i) {
                      return cuco::pair_type<Key, Value>{i / 2, i};
                    });

  cuco::static_map<Key, Value> map{
    num_pairs * 2, cuco::sentinel::empty_key<Key>{-1}, cuco::sentinel::empty_value<Value>{-1}};
  test_pair_functions<Key, Value>(map, d_pairs.begin(), num_pairs);
}
