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

#include <cuco/detail/__config>
#include <cuco/static_set.cuh>

#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/device_vector.h>

#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>

using size_type = std::size_t;

template <typename Set>
void test_for_each(Set& set, size_type num_keys, size_type expected_evens, size_type expected_odds)
{
  using Key = typename Set::key_type;

  REQUIRE(num_keys % 2 == 0);

  cuda::stream_ref stream{cudaStream_t{nullptr}};

  // Insert keys
  auto keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>{0}, cuda::proclaim_return_type<Key>([] __device__(auto i) {
      // generates a sequence of 0, 1, 2, ...
      return static_cast<Key>(i);
    }));
  set.insert(keys_begin, keys_begin + num_keys, stream);

  using Allocator = cuco::cuda_allocator<cuda::atomic<size_type, cuda::thread_scope_device>>;
  cuco::detail::counter_storage<size_type, cuda::thread_scope_device, Allocator> counter_storage(
    Allocator{}, stream);
  counter_storage.reset(stream);

  // count the sum of all even keys
  set.for_each(
    [counter = counter_storage.data()] __device__(auto const slot) {
      if (slot % 2 == 0) { counter->fetch_add(slot, cuda::memory_order_relaxed); }
    },
    stream);
  REQUIRE(counter_storage.load_to_host(stream) == expected_evens);

  counter_storage.reset(stream);

  // count the sum of all odd keys
  set.for_each(
    cuda::counting_iterator<size_type>(0),
    cuda::counting_iterator<size_type>(2 * num_keys),  // test for false-positives
    [counter = counter_storage.data()] __device__(auto const slot) {
      if (!(slot % 2 == 0)) { counter->fetch_add(slot, cuda::memory_order_relaxed); }
    },
    stream);
  REQUIRE(counter_storage.load_to_host(stream) == expected_odds);
}

TEMPLATE_TEST_CASE_SIG(
  "static_set for_each tests",
  "",
  ((typename Key, cuco::test::probe_sequence Probe, int CGSize), Key, Probe, CGSize),
  (int16_t, cuco::test::probe_sequence::double_hashing, 1),
  (int16_t, cuco::test::probe_sequence::double_hashing, 1),
  (int16_t, cuco::test::probe_sequence::double_hashing, 2),
  (int16_t, cuco::test::probe_sequence::linear_probing, 1),
  (int16_t, cuco::test::probe_sequence::linear_probing, 2),
  (int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, cuco::test::probe_sequence::linear_probing, 2)
#if defined(CUCO_HAS_128BIT_ATOMICS)
    ,
  (__int128_t, cuco::test::probe_sequence::double_hashing, 1),
  (__int128_t, cuco::test::probe_sequence::double_hashing, 2),
  (__int128_t, cuco::test::probe_sequence::linear_probing, 1),
  (__int128_t, cuco::test::probe_sequence::linear_probing, 2)
#endif
)
{
  // Limit key count for small types: leave room for the -1 sentinel.
  // Expected sums are pre-computed per type class:
  //   int16_t (num_keys=100): sum of evens 0..98 = 2450, sum of odds 1..99 = 2500
  //   int16_t+ (num_keys=1000): sum of evens 0..998 = 249'500, sum of odds 1..999 = 250'000
  constexpr size_type num_keys       = (sizeof(Key) == 1) ? 100 : 1'000;
  constexpr size_type expected_evens = (sizeof(Key) == 1) ? 2'450 : 249'500;
  constexpr size_type expected_odds  = (sizeof(Key) == 1) ? 2'500 : 250'000;

  using probe = std::conditional_t<
    Probe == cuco::test::probe_sequence::linear_probing,
    cuco::linear_probing<CGSize, cuco::murmurhash3_32<Key>>,
    cuco::double_hashing<CGSize, cuco::murmurhash3_32<Key>, cuco::murmurhash3_32<Key>>>;

  using set_t = cuco::static_set<Key,
                                 cuco::extent<size_type>,
                                 cuda::thread_scope_device,
                                 cuda::std::equal_to<Key>,
                                 probe,
                                 cuco::cuda_allocator<cuda::std::byte>,
                                 cuco::storage<2>>;

  auto set = set_t{num_keys, cuco::empty_key<Key>{static_cast<Key>(-1)}};
  test_for_each(set, num_keys, expected_evens, expected_odds);
}
