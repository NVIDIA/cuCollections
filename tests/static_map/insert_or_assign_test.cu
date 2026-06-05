/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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
#include <cuco/static_map.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <catch2/catch_template_test_macros.hpp>
#include <static_map/robin_hood_invariant.cuh>

using size_type = std::size_t;

template <typename Map>
void test_insert_or_assign(Map& map, size_type num_keys)
{
  using Key   = typename Map::key_type;
  using Value = typename Map::mapped_type;

  // Insert pairs
  auto pairs_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>(0),
    cuda::proclaim_return_type<cuco::pair<Key, Value>>(
      [] __device__(auto i) { return cuco::pair<Key, Value>{i, i}; }));

  auto const initial_size = map.insert(pairs_begin, pairs_begin + num_keys);
  REQUIRE(initial_size == num_keys);  // all keys should be inserted

  // Query pairs have the same keys but different payloads
  auto query_pairs_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>(0),
    cuda::proclaim_return_type<cuco::pair<Key, Value>>(
      [] __device__(auto i) { return cuco::pair<Key, Value>(i, i * 2); }));

  map.insert_or_assign(query_pairs_begin, query_pairs_begin + num_keys);

  // Robin Hood-specific: the populated table must satisfy the per-bucket layout invariant.
  if constexpr (cuco::is_robin_hood_probing<
                  typename std::decay_t<decltype(map)>::probing_scheme_type>::value) {
    cuco::test::check_robin_hood_invariant(map);
  }

  auto const updated_size = map.size();
  // all keys are present in the map so the size shouldn't change
  REQUIRE(updated_size == initial_size);

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<Value> d_values(num_keys);
  map.retrieve_all(d_keys.begin(), d_values.begin());

  auto gold_values_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>(0),
    cuda::proclaim_return_type<size_type>([] __device__(auto i) { return i * 2; }));

  thrust::sort(thrust::device, d_values.begin(), d_values.end());
  REQUIRE(cuco::test::equal(
    d_values.begin(), d_values.end(), gold_values_begin, cuda::std::equal_to<Value>{}));
}

TEMPLATE_TEST_CASE_SIG(
  "static_map insert_or_assign tests",
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
  (int64_t, int64_t, cuco::test::probe_sequence::linear_probing, 2),
  // Robin Hood mirrors the linear-probing rows. Only single-CAS (<= 8-byte) slots are
  // unconditional; wider-slot RH displacement needs a packed atom.cas.b128 (gated below).
  (int32_t, int32_t, cuco::test::probe_sequence::robin_hood, 1),
  (int32_t, int32_t, cuco::test::probe_sequence::robin_hood, 2)
#if defined(CUCO_HAS_128BIT_ATOMICS)
    ,
  (__int128_t, __int128_t, cuco::test::probe_sequence::double_hashing, 2),
  (__int128_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, __int128_t, cuco::test::probe_sequence::linear_probing, 2),
  // Wider-slot Robin Hood rows: the packed displacement CAS needs atom.cas.b128.
  (int32_t, int64_t, cuco::test::probe_sequence::robin_hood, 1),
  (int32_t, int64_t, cuco::test::probe_sequence::robin_hood, 2),
  (int64_t, int32_t, cuco::test::probe_sequence::robin_hood, 1),
  (int64_t, int32_t, cuco::test::probe_sequence::robin_hood, 2),
  (int64_t, int64_t, cuco::test::probe_sequence::robin_hood, 1),
  (int64_t, int64_t, cuco::test::probe_sequence::robin_hood, 2)
#endif
)
{
  constexpr size_type num_keys{400};

  using probe = std::conditional_t<
    Probe == cuco::test::probe_sequence::double_hashing,
    cuco::double_hashing<CGSize, cuco::murmurhash3_32<Key>, cuco::murmurhash3_32<Key>>,
    std::conditional_t<
      Probe == cuco::test::probe_sequence::robin_hood,
      cuco::robin_hood_probing<cuco::linear_probing<CGSize, cuco::murmurhash3_32<Key>>>,
      cuco::linear_probing<CGSize, cuco::murmurhash3_32<Key>>>>;

  auto map = cuco::static_map<Key,
                              Value,
                              cuco::extent<size_type>,
                              cuda::thread_scope_device,
                              cuda::std::equal_to<Key>,
                              probe,
                              cuco::cuda_allocator<cuda::std::byte>,
                              cuco::storage<2>>{
    num_keys, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};

  test_insert_or_assign(map, num_keys);
}

// Dedicated Robin Hood coverage for *concurrent* insert + assign. The probe-enum test above is
// phased (insert all keys, then assign in a separate pass), so it never exercises an assign racing
// a displacement -- the hazard RH introduces. Here every occurrence of key k assigns the same value
// (k * 2), so the final value is deterministic regardless of ordering; but a displacement-vs-assign
// race would land an assign on a different key's slot and corrupt it. Runs at ~0.95 load in one
// concurrent pass with duplicates, and verifies each key's value individually (a value-multiset
// check would miss a key<->key value swap from a misplaced assign).
TEMPLATE_TEST_CASE_SIG("static_map robin_hood insert_or_assign (high load)",
                       "",
                       ((typename Key, typename Value, int CGSize), Key, Value, CGSize),
                       (int32_t, int32_t, 1),
                       (int32_t, int32_t, 2)
#if defined(CUCO_HAS_128BIT_ATOMICS)
                         ,
                       (int64_t, int64_t, 1),
                       (int64_t, int64_t, 2)
#endif
)
{
  constexpr size_type num_unique_keys = 5'000;
  constexpr size_type num_keys        = 10'000;  // each unique key assigned twice (same value)

  using probe = cuco::robin_hood_probing<cuco::linear_probing<CGSize, cuco::murmurhash3_32<Key>>>;
  using map_type = cuco::static_map<Key,
                                    Value,
                                    cuco::extent<size_type>,
                                    cuda::thread_scope_device,
                                    cuda::std::equal_to<Key>,
                                    probe,
                                    cuco::cuda_allocator<cuda::std::byte>,
                                    cuco::storage<2>>;

  // ~0.95 load on the unique keys, so the table is nearly full and displacement is stressed.
  constexpr size_type capacity = static_cast<size_type>(num_unique_keys / 0.95);
  auto map = map_type{capacity, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};

  // Every occurrence of key k carries the same value (k * 2).
  auto pairs_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>(0),
    cuda::proclaim_return_type<cuco::pair<Key, Value>>([num_unique_keys] __device__(auto i) {
      auto const k = static_cast<Key>(i % num_unique_keys);
      return cuco::pair<Key, Value>{k, static_cast<Value>(k * 2)};
    }));

  map.insert_or_assign(pairs_begin, pairs_begin + num_keys);

  REQUIRE(map.size() == num_unique_keys);
  cuco::test::check_robin_hood_invariant(map);

  thrust::device_vector<Key> d_keys(num_unique_keys);
  thrust::device_vector<Value> d_values(num_unique_keys);
  map.retrieve_all(d_keys.begin(), d_values.begin());

  auto const zip = thrust::make_zip_iterator(cuda::std::tuple{d_keys.begin(), d_values.begin()});
  REQUIRE(cuco::test::all_of(
    zip, zip + num_unique_keys, cuda::proclaim_return_type<bool>([] __device__(auto const& p) {
      return cuda::std::get<1>(p) == static_cast<Value>(cuda::std::get<0>(p) * 2);
    })));
}
