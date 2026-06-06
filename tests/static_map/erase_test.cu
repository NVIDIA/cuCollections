/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.
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
#include <cuco/utility/reduction_functors.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/device_vector.h>

#include <catch2/catch_template_test_macros.hpp>
#include <static_map/robin_hood_invariant.cuh>

using size_type = int32_t;

template <typename Map>
void test_erase(Map& map, size_type num_keys)
{
  using key_type    = typename Map::key_type;
  using mapped_type = typename Map::mapped_type;

  thrust::device_vector<bool> d_keys_exist(num_keys);

  auto keys_begin = cuda::counting_iterator<key_type>(1);

  auto pairs_begin = cuda::make_transform_iterator(
    keys_begin,
    cuda::proclaim_return_type<cuco::pair<key_type, mapped_type>>([] __device__(key_type const& x) {
      return cuco::pair<key_type, mapped_type>(x, static_cast<mapped_type>(x));
    }));

  SECTION("Check basic insert/erase")
  {
    map.insert(pairs_begin, pairs_begin + num_keys);

    REQUIRE(map.size() == num_keys);

    map.erase(keys_begin, keys_begin + num_keys);

    REQUIRE(map.size() == 0);

    map.contains(keys_begin, keys_begin + num_keys, d_keys_exist.begin());

    REQUIRE(cuco::test::none_of(d_keys_exist.begin(), d_keys_exist.end(), cuda::std::identity{}));

    map.insert(pairs_begin, pairs_begin + num_keys);

    REQUIRE(map.size() == num_keys);

    map.contains(keys_begin, keys_begin + num_keys, d_keys_exist.begin());

    REQUIRE(cuco::test::all_of(d_keys_exist.begin(), d_keys_exist.end(), cuda::std::identity{}));

    map.erase(keys_begin, keys_begin + num_keys / 2);
    map.contains(keys_begin, keys_begin + num_keys, d_keys_exist.begin());

    REQUIRE(cuco::test::none_of(
      d_keys_exist.begin(), d_keys_exist.begin() + num_keys / 2, cuda::std::identity{}));

    REQUIRE(cuco::test::all_of(
      d_keys_exist.begin() + num_keys / 2, d_keys_exist.end(), cuda::std::identity{}));

    map.erase(keys_begin + num_keys / 2, keys_begin + num_keys);
    REQUIRE(map.size() == 0);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_map erase tests",
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
  // Wider-slot Robin Hood rows: RH displacement needs a single packed atom.cas.b128, so the slot
  // must be packable (padding-free). Only int64/int64 qualifies -- int32/int64 and int64/int32 are
  // padded (not is_packable), fall back to a split (back-to-back) CAS, and RH displacement would
  // livelock on an occupied slot.
  (int64_t, int64_t, cuco::test::probe_sequence::robin_hood, 1),
  (int64_t, int64_t, cuco::test::probe_sequence::robin_hood, 2)
#endif
)
{
  constexpr size_type num_keys{1'000'000};

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
                              cuco::storage<2>>{num_keys * 2,
                                                cuco::empty_key<Key>{-1},
                                                cuco::empty_value<Value>{-1},
                                                cuco::erased_key<Key>{-2}};

  test_erase(map, num_keys);
}

// Phase 1 Robin Hood erase: read-path after erase (no re-insertion -- insert-into-tombstone is
// separate). Inserts at high load so clusters are long, erases the first half, then checks
// contains: erased keys absent, the rest still found. This exercises the tombstone path in the
// Robin Hood early-exit -- a tombstone keeps the erased key's age (stored in its payload), so a
// lookup whose probe passes a tombstone neither terminates nor falsely early-exits. A garbage
// tombstone age would make *present* keys disappear (the all_of below would fail).
TEMPLATE_TEST_CASE_SIG("static_map robin_hood erase read-path",
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
  constexpr size_type num_keys = 5'000;

  using probe = cuco::robin_hood_probing<cuco::linear_probing<CGSize, cuco::murmurhash3_32<Key>>>;
  using map_type = cuco::static_map<Key,
                                    Value,
                                    cuco::extent<size_type>,
                                    cuda::thread_scope_device,
                                    cuda::std::equal_to<Key>,
                                    probe,
                                    cuco::cuda_allocator<cuda::std::byte>,
                                    cuco::storage<2>>;

  constexpr size_type capacity = static_cast<size_type>(num_keys / 0.85);
  auto map                     = map_type{
    capacity, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}, cuco::erased_key<Key>{-2}};

  // Keys 1..num_keys (avoid the -1 / -2 sentinels).
  auto keys_begin  = cuda::counting_iterator<Key>(1);
  auto pairs_begin = cuda::make_transform_iterator(
    keys_begin, cuda::proclaim_return_type<cuco::pair<Key, Value>>([] __device__(Key k) {
      return cuco::pair<Key, Value>{k, static_cast<Value>(k)};
    }));

  map.insert(pairs_begin, pairs_begin + num_keys);
  REQUIRE(map.size() == num_keys);

  constexpr size_type num_erased = num_keys / 2;
  map.erase(keys_begin, keys_begin + num_erased);
  REQUIRE(map.size() == num_keys - num_erased);

  thrust::device_vector<bool> d_contained(num_keys);
  map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
  REQUIRE(cuco::test::none_of(
    d_contained.begin(), d_contained.begin() + num_erased, cuda::std::identity{}));
  REQUIRE(
    cuco::test::all_of(d_contained.begin() + num_erased, d_contained.end(), cuda::std::identity{}));
}

// Robin Hood erase reuse + structural invariant at high load: insert, erase half (-> tombstones),
// re-insert (-> consume tombstones), checking the per-bucket Robin Hood layout invariant after each
// step (tombstones counted as residents, age read from their payload). A wrongly consumed tombstone
// (the age inversion) corrupts the layout and trips the invariant; a lost/duplicate key trips
// size/contains.
TEMPLATE_TEST_CASE_SIG("static_map robin_hood erase reuse + invariant",
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
  constexpr size_type num_keys = 10'000;

  using probe = cuco::robin_hood_probing<cuco::linear_probing<CGSize, cuco::murmurhash3_32<Key>>>;
  using map_type = cuco::static_map<Key,
                                    Value,
                                    cuco::extent<size_type>,
                                    cuda::thread_scope_device,
                                    cuda::std::equal_to<Key>,
                                    probe,
                                    cuco::cuda_allocator<cuda::std::byte>,
                                    cuco::storage<2>>;

  constexpr size_type capacity = static_cast<size_type>(num_keys / 0.85);
  auto map                     = map_type{
    capacity, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}, cuco::erased_key<Key>{-2}};

  auto keys_begin  = cuda::counting_iterator<Key>(1);
  auto pairs_begin = cuda::make_transform_iterator(
    keys_begin, cuda::proclaim_return_type<cuco::pair<Key, Value>>([] __device__(Key k) {
      return cuco::pair<Key, Value>{k, static_cast<Value>(k)};
    }));

  map.insert(pairs_begin, pairs_begin + num_keys);
  REQUIRE(map.size() == num_keys);
  cuco::test::check_robin_hood_invariant(map);

  constexpr size_type num_erased = num_keys / 2;
  map.erase(keys_begin, keys_begin + num_erased);
  REQUIRE(map.size() == num_keys - num_erased);
  cuco::test::check_robin_hood_invariant(map);  // tombstones-as-residents layout still valid

  map.insert(pairs_begin, pairs_begin + num_erased);  // consume tombstones / fill empties
  REQUIRE(map.size() == num_keys);
  cuco::test::check_robin_hood_invariant(map);  // layout valid after reuse

  thrust::device_vector<bool> d_contained(num_keys);
  map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
  REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
}

namespace {
enum class reinsert_via { insert_or_assign, insert_or_apply, insert_and_find };

// Robin Hood erase + reuse through a specific re-insert API: insert, check invariant, erase the
// first half (-> tombstones), check invariant, then re-insert that half via `how` (-> consume
// tombstones) and check invariant + that every key is present. Exercises the tombstone path of the
// chosen insert variant.
template <typename Key, typename Value, int CGSize>
void test_rh_erase_reuse(size_type num_keys, reinsert_via how)
{
  using probe = cuco::robin_hood_probing<cuco::linear_probing<CGSize, cuco::murmurhash3_32<Key>>>;
  using map_type = cuco::static_map<Key,
                                    Value,
                                    cuco::extent<size_type>,
                                    cuda::thread_scope_device,
                                    cuda::std::equal_to<Key>,
                                    probe,
                                    cuco::cuda_allocator<cuda::std::byte>,
                                    cuco::storage<2>>;

  auto const capacity = static_cast<size_type>(num_keys / 0.85);
  auto map            = map_type{
    capacity, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}, cuco::erased_key<Key>{-2}};

  auto keys_begin  = cuda::counting_iterator<Key>(1);
  auto pairs_begin = cuda::make_transform_iterator(
    keys_begin, cuda::proclaim_return_type<cuco::pair<Key, Value>>([] __device__(Key k) {
      return cuco::pair<Key, Value>{k, static_cast<Value>(k)};
    }));

  map.insert(pairs_begin, pairs_begin + num_keys);
  REQUIRE(map.size() == num_keys);
  cuco::test::check_robin_hood_invariant(map);

  auto const num_erased = num_keys / 2;
  map.erase(keys_begin, keys_begin + num_erased);
  REQUIRE(map.size() == num_keys - num_erased);
  cuco::test::check_robin_hood_invariant(map);

  // re-insert the erased half through the variant under test (consuming tombstones)
  switch (how) {
    case reinsert_via::insert_or_assign:
      map.insert_or_assign(pairs_begin, pairs_begin + num_erased);
      break;
    case reinsert_via::insert_or_apply:
      map.insert_or_apply(pairs_begin, pairs_begin + num_erased, cuco::reduce::plus{});
      break;
    case reinsert_via::insert_and_find: {
      thrust::device_vector<size_type> found(num_erased);
      thrust::device_vector<bool> inserted(num_erased);
      map.insert_and_find(pairs_begin, pairs_begin + num_erased, found.begin(), inserted.begin());
      break;
    }
  }

  REQUIRE(map.size() == num_keys);
  cuco::test::check_robin_hood_invariant(map);

  thrust::device_vector<bool> d_contained(num_keys);
  map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
  REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
}
}  // namespace

TEMPLATE_TEST_CASE_SIG("static_map robin_hood erase + insert_or_assign reuse",
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
  test_rh_erase_reuse<Key, Value, CGSize>(10'000, reinsert_via::insert_or_assign);
}

TEMPLATE_TEST_CASE_SIG("static_map robin_hood erase + insert_or_apply reuse",
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
  test_rh_erase_reuse<Key, Value, CGSize>(10'000, reinsert_via::insert_or_apply);
}

TEMPLATE_TEST_CASE_SIG("static_map robin_hood erase + insert_and_find reuse",
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
  test_rh_erase_reuse<Key, Value, CGSize>(10'000, reinsert_via::insert_and_find);
}
