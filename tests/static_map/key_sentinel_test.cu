/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/__config>
#include <cuco/static_map.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/device_vector.h>

#include <catch2/catch_template_test_macros.hpp>

#define SIZE 10
__device__ int A[SIZE];

template <typename T>
struct custom_equals {
  __device__ bool operator()(T lhs, T rhs) const { return A[lhs] == A[rhs]; }
};

TEMPLATE_TEST_CASE_SIG("static_map key sentinel tests",
                       "",
                       ((typename T), T),
                       (int32_t),
                       (int64_t)
#if defined(CUCO_HAS_128BIT_ATOMICS)
                         ,
                       (__int128_t)
#endif
)
{
  using Key   = T;
  using Value = T;

  constexpr std::size_t num_keys{SIZE};
  auto map = cuco::static_map{SIZE * 2,
                              cuco::empty_key<Key>{-1},
                              cuco::empty_value<Value>{-1},
                              custom_equals<Key>{},
                              cuco::linear_probing<1, cuco::default_hash_function<Key>>{}};

  auto insert_ref = map.ref(cuco::op::insert);
  auto find_ref   = map.ref(cuco::op::find);

  int h_A[SIZE];
  for (int i = 0; i < SIZE; i++) {
    h_A[i] = i;
  }
  CUCO_CUDA_TRY(cudaMemcpyToSymbol(A, h_A, SIZE * sizeof(int)));

  auto pairs_begin = cuda::make_transform_iterator(
    cuda::make_counting_iterator<T>(0),
    cuda::proclaim_return_type<cuco::pair<Key, Value>>(
      [] __device__(auto i) { return cuco::pair<Key, Value>(i, i); }));

  SECTION(
    "Tests of non-CG insert: The custom `key_equal` can never be used to compare against sentinel")
  {
    REQUIRE(
      cuco::test::all_of(pairs_begin,
                         pairs_begin + num_keys,
                         cuda::proclaim_return_type<bool>(
                           [insert_ref] __device__(cuco::pair<Key, Value> const& pair) mutable {
                             return insert_ref.insert(pair);
                           })));
  }

  SECTION(
    "Tests of CG insert: The custom `key_equal` can never be used to compare against sentinel")
  {
    map.insert(pairs_begin, pairs_begin + num_keys);
    // All keys inserted via custom `key_equal` should be found
    REQUIRE(cuco::test::all_of(
      pairs_begin,
      pairs_begin + num_keys,
      cuda::proclaim_return_type<bool>([find_ref] __device__(cuco::pair<Key, Value> const& pair) {
        auto const found = find_ref.find(pair.first);
        return (found != find_ref.end()) and
               (found->first == pair.first and found->second == pair.second);
      })));
  }
}
