/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/__config>
#include <cuco/static_map.cuh>

#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/device_vector.h>

#include <catch2/catch_template_test_macros.hpp>

using size_type = std::size_t;

template <typename Map>
void test_for_each(Map& map, size_type num_keys)
{
  using Key   = typename Map::key_type;
  using Value = typename Map::mapped_type;

  REQUIRE(num_keys % 2 == 0);

  // Insert pairs
  auto pairs_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>(0),
    cuda::proclaim_return_type<cuco::pair<Key, Value>>([] __device__(auto i) {
      // use payload as 1 for even keys and 2 for odd keys
      return cuco::pair<Key, Value>{i, i % 2 + 1};
    }));

  cuda::stream_ref stream{cudaStream_t{nullptr}};

  map.insert(pairs_begin, pairs_begin + num_keys, stream);

  using Allocator = cuco::cuda_allocator<cuda::atomic<size_type, cuda::thread_scope_device>>;
  cuco::detail::counter_storage<size_type, cuda::thread_scope_device, Allocator> counter_storage(
    Allocator{}, stream);
  counter_storage.reset(stream);

  // count all the keys which are even and whose payload has value 1
  map.for_each(
    [counter = counter_storage.data()] __device__(auto const slot) {
      auto const& [key, value] = slot;
      if (((key % 2 == 0)) and (value == 1)) { counter->fetch_add(1, cuda::memory_order_relaxed); }
    },
    stream);

  auto const res = counter_storage.load_to_host(stream);
  REQUIRE(res == num_keys / 2);

  counter_storage.reset(stream);

  map.for_each(
    cuda::counting_iterator<size_type>(0),
    cuda::counting_iterator<size_type>(2 * num_keys),  // test for false-positives
    [counter = counter_storage.data()] __device__(auto const slot) {
      auto const& [key, value] = slot;
      if (((key % 2 == 0)) and (value == 1)) { counter->fetch_add(1, cuda::memory_order_relaxed); }
    },
    stream);
  REQUIRE(res == num_keys / 2);
}

TEMPLATE_TEST_CASE_SIG(
  "static_map for_each tests",
  "",
  ((typename Key, typename Value, cuco::test::probe_sequence Probe, int CGSize),
   Key,
   Value,
   Probe,
   CGSize),
  // Robin Hood is linear-probing + single-CAS only; unsupported variants (double_hashing,
  // padded/oversized slots) are commented; 16B int64/int64 needs 128-bit atomics.
  //  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing, 1),
  //  (int32_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),
  //  (int32_t, int32_t, cuco::test::probe_sequence::double_hashing, 2),
  //  (int32_t, int64_t, cuco::test::probe_sequence::double_hashing, 2),
  //  (int64_t, int32_t, cuco::test::probe_sequence::double_hashing, 1),
  //  (int64_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),
  //  (int64_t, int32_t, cuco::test::probe_sequence::double_hashing, 2),
  //  (int64_t, int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, int32_t, cuco::test::probe_sequence::linear_probing, 1),
  //  (int32_t, int64_t, cuco::test::probe_sequence::linear_probing, 1),  // padded 12B slot
  (int32_t, int32_t, cuco::test::probe_sequence::linear_probing, 2)
//  (int32_t, int64_t, cuco::test::probe_sequence::linear_probing, 2),  // padded 12B slot
//  (int64_t, int32_t, cuco::test::probe_sequence::linear_probing, 1),  // padded 12B slot
//  (int64_t, int32_t, cuco::test::probe_sequence::linear_probing, 2),  // padded 12B slot
#if defined(CUCO_HAS_128BIT_ATOMICS)
    ,
  (int64_t, int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, int64_t, cuco::test::probe_sequence::linear_probing, 2)
#endif
  //  (__int128_t, __int128_t, cuco::test::probe_sequence::double_hashing, 2),  // 32B slot
  //  (__int128_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),     // oversized slot
  //  (int32_t, __int128_t, cuco::test::probe_sequence::linear_probing, 2)      // oversized slot
)
{
  constexpr size_type num_keys{100};
  using probe = std::conditional_t<
    Probe == cuco::test::probe_sequence::linear_probing,
    cuco::linear_probing<CGSize, cuco::murmurhash3_32<Key>>,
    cuco::double_hashing<CGSize, cuco::murmurhash3_32<Key>, cuco::murmurhash3_32<Key>>>;

  using map_type = cuco::static_map<Key,
                                    Value,
                                    cuco::extent<size_type>,
                                    cuda::thread_scope_device,
                                    cuda::std::equal_to<Key>,
                                    probe,
                                    cuco::cuda_allocator<cuda::std::byte>,
                                    cuco::storage<2>>;

  auto map = map_type{num_keys, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{0}};
  test_for_each(map, num_keys);
}
