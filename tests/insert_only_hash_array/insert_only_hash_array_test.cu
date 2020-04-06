/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "catch.hpp"

#include <cu_collections/utilities/error.hpp>
#include <cuco/insert_only_hash_array.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <algorithm>

namespace {
// Thrust logical algorithms (any_of/all_of/none_of) don't work with device
// lambdas: See https://github.com/thrust/thrust/issues/1062
template <typename Iterator, typename Predicate>
bool all_of(Iterator begin, Iterator end, Predicate p) {
  auto size = thrust::distance(begin, end);
  return size == thrust::count_if(begin, end, p);
}

template <typename Iterator, typename Predicate>
bool any_of(Iterator begin, Iterator end, Predicate p) {
  return thrust::count_if(begin, end, p) > 0;
}

template <typename Iterator, typename Predicate>
bool none_of(Iterator begin, Iterator end, Predicate p) {
  return not all_of(begin, end, p);
}
}  // namespace

TEMPLATE_TEST_CASE("Unique sequence of keys", "", int32_t, int64_t) {
  using Key = TestType;
  using Value = TestType;

  constexpr std::size_t num_keys{50'000'000};
  cuco::insert_only_hash_array<Key, Value> a{100'000'000, -1, -1};


/*
  static constexpr std::size_t max_cas_size_bytes{8};
  if ((sizeof(Key) + sizeof(Value)) <= max_cas_size_bytes) {
    REQUIRE(true == a.is_lock_free());
  } else {
    REQUIRE(false == a.is_lock_free());
  }
  */

  auto view = a.get_device_view();

  auto key_iterator = thrust::make_counting_iterator<Key>(0);
  auto value_iterator = thrust::make_counting_iterator<Value>(0);
  auto kv_pairs = thrust::make_zip_iterator(
      thrust::make_tuple(key_iterator, value_iterator));

  SECTION("Inserting keys should return valid iterator and insert success.") {
    REQUIRE(all_of(
        kv_pairs, kv_pairs + num_keys,
        [view] __device__(cuco::pair<Key, Value> const& pair) mutable {
          auto const result = view.insert(pair);
          return (result.first != view.end()) and (result.second == true);
        }));
  }

  SECTION("Dereferenced iterator should equal inserted pair.") {
    REQUIRE(
        all_of(kv_pairs, kv_pairs + num_keys,
               [view] __device__(cuco::pair<Key, Value> const& pair) mutable {
                 auto const result = view.insert(pair);
                 auto const iter = result.first;
                 return iter->first.load() == pair.first and
                        iter->second.load() == pair.second;
               }));
  }

  SECTION("Key is found immediately after insertion.") {
    REQUIRE(all_of(
        kv_pairs, kv_pairs + num_keys,
        [view] __device__(cuco::pair<Key, Value> const& pair) mutable {
          auto const insert_result = view.insert(pair);
          auto const find_result = view.find(pair.first);
          bool const same_iterator = (insert_result.first == find_result);
          return same_iterator;
        }));
  }

  SECTION("Inserting same key twice.") {
    REQUIRE(
        all_of(kv_pairs, kv_pairs + num_keys,
               [view] __device__(cuco::pair<Key, Value> const& pair) mutable {
                 auto const first_insert = view.insert(pair);
                 auto const second_insert = view.insert(pair);
                 auto const find_result = view.find(pair.first);
                 bool const same_iterator =
                     (first_insert.first == second_insert.first);

                 // Inserting the same key twice should return the same
                 // iterator and false for the insert result
                 return same_iterator and (second_insert.second == false);
               }));
  }

  SECTION("Pair isn't changed after inserting twice.") {
    REQUIRE(
        all_of(kv_pairs, kv_pairs + num_keys,
               [view] __device__(cuco::pair<Key, Value> const& pair) mutable {
                 auto const first_insert = view.insert(pair);
                 auto const second_insert = view.insert(pair);
                 auto const iter = second_insert.first;
                 return iter->first.load() == pair.first and
                        iter->second.load() == pair.second;
               }));
  }

  SECTION("Cannot find any key in an empty hash map") {
    REQUIRE(
        all_of(kv_pairs, kv_pairs + num_keys,
               [view] __device__(cuco::pair<Key, Value> const& pair) mutable {
                 return view.find(pair.first) == view.end();
               }));
  }

  SECTION("Keys are all found after inserting many keys.") {
    // Bulk insert keys
    REQUIRE(
        all_of(kv_pairs, kv_pairs + num_keys,
               [view] __device__(cuco::pair<Key, Value> const& pair) mutable {
                 return view.insert(pair).second;
               }));

    // All keys should be found
    REQUIRE(
        all_of(kv_pairs, kv_pairs + num_keys,
               [view] __device__(cuco::pair<Key, Value> const& pair) mutable {
                 auto const found = view.find(pair.first);
                 return (found != view.end()) and
                        (found->first.load() == pair.first and
                         found->second.load() == pair.second);
               }));
  }
}
