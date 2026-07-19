/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/__config>
#include <cuco/hash_functions.cuh>
#include <cuco/static_map.cuh>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/device_vector.h>

#include <catch2/catch_template_test_macros.hpp>

using size_type = std::size_t;

template <typename Key, typename Hash>
void test_hash_function()
{
  using Value = Key;

  constexpr size_type num_keys{400};

  auto map = cuco::static_map<Key,
                              Value,
                              cuco::extent<size_type>,
                              cuda::thread_scope_device,
                              cuda::std::equal_to<Key>,
                              cuco::linear_probing<1, Hash>,
                              cuco::cuda_allocator<cuda::std::byte>,
                              cuco::storage<2>>{
    num_keys, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};

  auto keys_begin = cuda::counting_iterator<Key>(1);

  auto pairs_begin = cuda::make_transform_iterator(
    keys_begin, cuda::proclaim_return_type<cuco::pair<Key, Value>>([] __device__(auto i) {
      return cuco::pair<Key, Value>(i, i);
    }));

  thrust::device_vector<bool> d_keys_exist(num_keys);

  map.insert(pairs_begin, pairs_begin + num_keys);

  REQUIRE(map.size() == num_keys);

  map.contains(keys_begin, keys_begin + num_keys, d_keys_exist.begin());

  REQUIRE(cuco::test::all_of(d_keys_exist.begin(), d_keys_exist.end(), cuda::std::identity{}));
}

// Robin Hood is linear-probing + single-CAS only; unsupported variants (double_hashing,
// padded/oversized slots) are commented; 16B int64/int64 needs 128-bit atomics.
TEMPLATE_TEST_CASE_SIG("static_map hash tests",
                       "",
                       ((typename Key)),
                       (int32_t)
#if defined(CUCO_HAS_128BIT_ATOMICS)
                         ,
                       (int64_t)
#endif
                       //  (__int128_t)  // 32B slot: oversized for single-CAS Robin Hood
)
{
  test_hash_function<Key, cuco::murmurhash3_32<Key>>();
  test_hash_function<Key, cuco::murmurhash3_x64_128<Key>>();
  test_hash_function<Key, cuco::xxhash_32<Key>>();
  test_hash_function<Key, cuco::xxhash_64<Key>>();
}
