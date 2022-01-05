/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <catch2/catch.hpp>

#include <cuco/static_map.cuh>

#include <utils.hpp>

#define SIZE 10
__device__ int A[SIZE];

template <typename T>
struct custom_equals {
  __device__ bool operator()(T lhs, T rhs) { return A[lhs] == A[rhs]; }
};

TEMPLATE_TEST_CASE_SIG(
  "Key comparison against sentinel", "", ((typename T), T), (int32_t), (int64_t))
{
  using Key   = T;
  using Value = T;

  constexpr std::size_t num_keys{SIZE};
  cuco::static_map<Key, Value> map{SIZE * 2, -1, -1};

  auto m_view = map.get_device_mutable_view();
  auto view   = map.get_device_view();

  int h_A[SIZE];
  for (int i = 0; i < SIZE; i++) {
    h_A[i] = i;
  }
  cudaMemcpyToSymbol(A, h_A, SIZE * sizeof(int));

  auto pairs_begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator<T>(0),
    [] __device__(auto i) { return cuco::pair_type<Key, Value>(i, i); });

  SECTION(
    "Tests of non-CG insert: The custom `key_equal` can never be used to compare against sentinel")
  {
    REQUIRE(cuco::test::all_of(
      pairs_begin,
      pairs_begin + num_keys,
      [m_view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
        return m_view.insert(pair, cuco::detail::MurmurHash3_32<Key>{}, custom_equals<Key>{});
      }));
  }

  SECTION(
    "Tests of CG insert: The custom `key_equal` can never be used to compare against sentinel")
  {
    map.insert(pairs_begin,
               pairs_begin + num_keys,
               cuco::detail::MurmurHash3_32<Key>{},
               custom_equals<Key>{});
    // All keys inserted via custom `key_equal` should be found
    REQUIRE(cuco::test::all_of(pairs_begin,
                               pairs_begin + num_keys,
                               [view] __device__(cuco::pair_type<Key, Value> const& pair) {
                                 auto const found = view.find(pair.first);
                                 return (found != view.end()) and
                                        (found->first.load() == pair.first and
                                         found->second.load() == pair.second);
                               }));
  }
}
