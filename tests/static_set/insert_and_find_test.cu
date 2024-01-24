/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuco/static_set.cuh>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <catch2/catch_template_test_macros.hpp>

#include <cuda/functional>

template <typename Set>
__inline__ void test_insert_and_find(Set& set, std::size_t num_keys)
{
  using Key                     = typename Set::key_type;
  static auto constexpr cg_size = Set::cg_size;

  auto const keys_begin = [&]() {
    if constexpr (cg_size == 1) {
      return thrust::counting_iterator<Key>(0);
    } else {
      return thrust::make_transform_iterator(
        thrust::counting_iterator<Key>(0),
        cuda::proclaim_return_type<Key>([] __device__(auto i) { return i / cg_size; }));
    }
  }();
  auto const keys_end = [&]() {
    if constexpr (cg_size == 1) {
      return keys_begin + num_keys;
    } else {
      return keys_begin + num_keys * cg_size;
    }
  }();

  auto ref = set.ref(cuco::op::insert_and_find);

  REQUIRE(cuco::test::all_of(keys_begin, keys_end, [ref] __device__(Key key) mutable {
    auto [iter, inserted] = [&]() {
      if constexpr (cg_size == 1) {
        return ref.insert_and_find(key);
      } else {
        auto const tile =
          cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());
        return ref.insert_and_find(tile, key);
      }
    }();
    return inserted == true;
  }));

  SECTION("Inserting elements for the second time will always fail.")
  {
    REQUIRE(cuco::test::all_of(keys_begin, keys_end, [ref] __device__(Key key) mutable {
      auto [iter, inserted] = [&]() {
        if constexpr (cg_size == 1) {
          return ref.insert_and_find(key);
        } else {
          auto const tile =
            cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());
          return ref.insert_and_find(tile, key);
        }
      }();
      return inserted == false and key == *iter;
    }));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Insert and find",
  "",
  ((typename Key, cuco::test::probe_sequence Probe, int CGSize), Key, Probe, CGSize),
  (int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, cuco::test::probe_sequence::linear_probing, 2))
{
  constexpr std::size_t num_keys{400};

  using probe = std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                                   cuco::linear_probing<CGSize, cuco::default_hash_function<Key>>,
                                   cuco::double_hashing<CGSize, cuco::default_hash_function<Key>>>;

  auto set =
    cuco::static_set{num_keys, cuco::empty_key<Key>{-1}, {}, probe{}, {}, cuco::storage<2>{}};

  test_insert_and_find(set, num_keys);
}
