/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, Jonas Hahnfeld, CERN.
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/__config>
#include <cuco/static_map.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <catch2/catch_template_test_macros.hpp>

// Disabled under the experimental hard-wired Robin Hood static_map.
// insert_and_find is excluded under Robin Hood: its returned iterator can dangle once a later
// insert displaces the key (pointer instability). Re-enable (and trim slot types / probe scheme as
// needed) once Robin Hood support is generalized. See robin_hood_refactor_plan.md.
//
// using size_type = std::size_t;
//
// TEMPLATE_TEST_CASE_SIG(
//   "static_map insert_and_find tests",
//   "",
//   ((typename Key, typename Value, cuco::test::probe_sequence Probe, int CGSize),
//    Key,
//    Value,
//    Probe,
//    CGSize),
//   (int32_t, int32_t, cuco::test::probe_sequence::double_hashing, 1),
//   (int32_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),
//   (int32_t, int32_t, cuco::test::probe_sequence::double_hashing, 2),
//   (int32_t, int64_t, cuco::test::probe_sequence::double_hashing, 2),
//   (int64_t, int32_t, cuco::test::probe_sequence::double_hashing, 1),
//   (int64_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),
//   (int64_t, int32_t, cuco::test::probe_sequence::double_hashing, 2),
//   (int64_t, int64_t, cuco::test::probe_sequence::double_hashing, 2),
//   (int32_t, int32_t, cuco::test::probe_sequence::linear_probing, 1),
//   (int32_t, int64_t, cuco::test::probe_sequence::linear_probing, 1),
//   (int32_t, int32_t, cuco::test::probe_sequence::linear_probing, 2),
//   (int32_t, int64_t, cuco::test::probe_sequence::linear_probing, 2),
//   (int64_t, int32_t, cuco::test::probe_sequence::linear_probing, 1),
//   (int64_t, int64_t, cuco::test::probe_sequence::linear_probing, 1),
//   (int64_t, int32_t, cuco::test::probe_sequence::linear_probing, 2),
//   (int64_t, int64_t, cuco::test::probe_sequence::linear_probing, 2)
// #if defined(CUCO_HAS_128BIT_ATOMICS)
//     ,
//   (__int128_t, __int128_t, cuco::test::probe_sequence::double_hashing, 2),
//   (__int128_t, int64_t, cuco::test::probe_sequence::double_hashing, 1),
//   (int32_t, __int128_t, cuco::test::probe_sequence::linear_probing, 2)
// #endif
// )
// {
// #if !defined(CUCO_HAS_INDEPENDENT_THREADS)
//   if constexpr (cuco::detail::is_packable<cuco::pair<Key, Value>>())
// #endif
//   {
//     using probe = std::conditional_t<
//       Probe == cuco::test::probe_sequence::linear_probing,
//       cuco::linear_probing<CGSize, cuco::murmurhash3_32<Key>>,
//       cuco::double_hashing<CGSize, cuco::murmurhash3_32<Key>, cuco::murmurhash3_32<Key>>>;
//
//     constexpr size_type num_keys{400};
//
//     auto map = cuco::static_map<Key,
//                                 Value,
//                                 cuco::extent<size_type>,
//                                 cuda::thread_scope_device,
//                                 cuda::std::equal_to<Key>,
//                                 probe,
//                                 cuco::cuda_allocator<cuda::std::byte>,
//                                 cuco::storage<2>>{
//       num_keys, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};
//
//     auto pairs_begin = cuda::make_transform_iterator(
//       cuda::counting_iterator<size_type>(0),
//       cuda::proclaim_return_type<cuco::pair<Key, Value>>(
//         [] __device__(auto i) { return cuco::pair<Key, Value>{i, 1}; }));
//
//     thrust::device_vector<size_type> found1(num_keys);
//     thrust::device_vector<size_type> found2(num_keys);
//
//     thrust::device_vector<bool> inserted(num_keys);
//
//     // insert first time, fills inserted with true
//     map.insert_and_find(pairs_begin, pairs_begin + num_keys, found1.begin(), inserted.begin());
//     REQUIRE(cuco::test::all_of(inserted.begin(), inserted.end(), cuda::std::identity{}));
//
//     // insert second time, fills inserted with false as keys already in map
//     map.insert_and_find(pairs_begin, pairs_begin + num_keys, found2.begin(), inserted.begin());
//     REQUIRE(cuco::test::none_of(inserted.begin(), inserted.end(), cuda::std::identity{}));
//
//     // both found1 and found2 should be same, as keys will be referring to same slot
//     REQUIRE(
//       cuco::test::equal(found1.begin(), found1.end(), found2.begin(),
//       cuda::std::equal_to<Key>{}));
//   }
// }
