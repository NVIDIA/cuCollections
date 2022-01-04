/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <catch2/catch.hpp>

#include <cuco/static_multimap.cuh>

#include <util.hpp>

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
  REQUIRE(res == num_pairs);

  // query pair matching rate = 50%
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    pair_begin,
                    [] __device__(auto i) {
                      return cuco::pair_type<Key, Value>{i, i};
                    });

  SECTION("Output of pair_count and pair_retrieve should be coherent.")
  {
    auto num = map.pair_count(pair_begin, pair_begin + num_pairs, pair_equal<Key, Value>{});

    auto out1_begin = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
    auto out2_begin = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

    REQUIRE(num == num_pairs);

    auto [out1_end, out2_end] = map.pair_retrieve(
      pair_begin, pair_begin + num_pairs, out1_begin, out2_begin, pair_equal<Key, Value>{});

    REQUIRE((out1_end - out1_begin) == num_pairs);
  }

  SECTION("Output of pair_count_outer and pair_retrieve_outer should be coherent.")
  {
    auto num = map.pair_count_outer(pair_begin, pair_begin + num_pairs, pair_equal<Key, Value>{});

    auto out1_begin = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
    auto out2_begin = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

    REQUIRE(num == (num_pairs + num_pairs / 2));

    auto [out1_end, out2_end] = map.pair_retrieve_outer(
      pair_begin, pair_begin + num_pairs, out1_begin, out2_begin, pair_equal<Key, Value>{});

    REQUIRE((out1_end - out1_begin) == (num_pairs + num_pairs / 2));
  }
}

TEMPLATE_TEST_CASE_SIG("Tests of pair functions",
                       "",
                       ((typename Key, typename Value, probe_sequence Probe), Key, Value, Probe),
                       (int32_t, int32_t, probe_sequence::linear_probing),
                       (int32_t, int64_t, probe_sequence::linear_probing),
                       (int64_t, int64_t, probe_sequence::linear_probing),
                       (int32_t, int32_t, probe_sequence::double_hashing),
                       (int32_t, int64_t, probe_sequence::double_hashing),
                       (int64_t, int64_t, probe_sequence::double_hashing))
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

  if constexpr (Probe == probe_sequence::linear_probing) {
    cuco::static_multimap<Key,
                          Value,
                          cuda::thread_scope_device,
                          cuco::cuda_allocator<char>,
                          cuco::linear_probing<1, cuco::detail::MurmurHash3_32<Key>>>
      map{num_pairs * 2, -1, -1};
    test_pair_functions<Key, Value>(map, d_pairs.begin(), num_pairs);
  }
  if constexpr (Probe == probe_sequence::double_hashing) {
    cuco::static_multimap<Key, Value> map{num_pairs * 2, -1, -1};
    test_pair_functions<Key, Value>(map, d_pairs.begin(), num_pairs);
  }
}
