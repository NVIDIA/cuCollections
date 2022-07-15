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

#include <cuco/static_multimap.cuh>

#include <catch2/catch.hpp>

#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>

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

  // query pair matching rate = 50%
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    pair_begin,
                    [] __device__(auto i) {
                      return cuco::pair_type<Key, Value>{i, i};
                    });

  // 50% `pred == true`
  auto pred    = [] __device__(int32_t k) { return k % 2 == 0; };
  auto stencil = thrust::make_counting_iterator<int32_t>(0);

  SECTION("Output of pair_count_if and pair_retrieve_if should be coherent.")
  {
    auto const count = map.pair_count_if(
      pair_begin, pair_begin + num_pairs, stencil, pred, pair_equal<Key, Value>{});

    REQUIRE(count * 2 == num_pairs);

    auto out1_begin = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
    auto out2_begin = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

    auto [out1_end, out2_end] = map.pair_retrieve_if(pair_begin,
                                                     pair_begin + num_pairs,
                                                     stencil,
                                                     pred,
                                                     out1_begin,
                                                     out2_begin,
                                                     pair_equal<Key, Value>{});

    REQUIRE((out1_end - out1_begin) * 2 == num_pairs);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Tests of pair_retrieve_if functions",
  "",
  ((typename Key, typename Value, cuco::test::probe_sequence Probe), Key, Value, Probe),
  (int32_t, int32_t, cuco::test::probe_sequence::linear_probing),
  (int32_t, int64_t, cuco::test::probe_sequence::linear_probing),
  (int64_t, int64_t, cuco::test::probe_sequence::linear_probing),
  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing),
  (int32_t, int64_t, cuco::test::probe_sequence::double_hashing),
  (int64_t, int64_t, cuco::test::probe_sequence::double_hashing))
{
  constexpr std::size_t num_pairs{200};
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(num_pairs);

  // pair multiplicity = 2
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    d_pairs.begin(),
                    [] __device__(auto i) {
                      return cuco::pair_type<Key, Value>{i / 2, i};
                    });

  if constexpr (Probe == cuco::test::probe_sequence::linear_probing) {
    cuco::static_multimap<Key,
                          Value,
                          cuda::thread_scope_device,
                          cuco::cuda_allocator<char>,
                          cuco::linear_probing<1, cuco::detail::MurmurHash3_32<Key>>>
      map{
        num_pairs * 2, cuco::sentinel::empty_key<Key>{-1}, cuco::sentinel::empty_value<Value>{-1}};
    test_pair_functions<Key, Value>(map, d_pairs.begin(), num_pairs);
  }
  if constexpr (Probe == cuco::test::probe_sequence::double_hashing) {
    cuco::static_multimap<Key, Value> map{
      num_pairs * 2, cuco::sentinel::empty_key<Key>{-1}, cuco::sentinel::empty_value<Value>{-1}};
    test_pair_functions<Key, Value>(map, d_pairs.begin(), num_pairs);
  }
}
