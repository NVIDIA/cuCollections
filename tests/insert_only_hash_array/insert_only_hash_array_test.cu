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

#include <cu_collections/utilities/error.hpp>
#include <cuco/insert_only_hash_array.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <catch.hpp>

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

TEST_CASE("The first test") {
  insert_only_hash_array<int32_t, int32_t> a{1000, -1, -1};

  auto view = a.get_device_view();

  std::vector<thrust::pair<int32_t, int32_t>> pairs(100);
  std::generate(pairs.begin(), pairs.end(), []() {
    static int32_t counter{};
    ++counter;
    return thrust::make_pair(counter, counter);
  });

  thrust::device_vector<thrust::pair<int32_t, int32_t>> d_pairs(pairs);

  SECTION("Inserting keys should return valid iterator and insert success.") {
    REQUIRE(all_of(
        d_pairs.begin(), d_pairs.end(),
        [view] __device__(thrust::pair<int32_t, int32_t> const& pair) mutable {
          auto result = view.insert(pair);
          return (result.first != view.end()) and (result.second == true);
        }));
  }

  SECTION("Key is found immediately after insertion.") {
    REQUIRE(all_of(
        d_pairs.begin(), d_pairs.end(),
        [view] __device__(thrust::pair<int32_t, int32_t> const& pair) mutable {
          auto insert_result = view.insert(pair);
          auto find_result = view.find(pair.first);
          bool same_iterator = (insert_result.first == find_result);
          return same_iterator;
        }));
  }

  SECTION("Inserting same key twice.") {
    REQUIRE(all_of(
        d_pairs.begin(), d_pairs.end(),
        [view] __device__(thrust::pair<int32_t, int32_t> const& pair) mutable {
          auto const first_insert = view.insert(pair);
          auto const second_insert = view.insert(pair);
          bool const same_iterator =
              (first_insert.first == second_insert.first);

          // Inserting the same key twice should return the same
          // iterator and false for the insert result
          return same_iterator and (second_insert.second == false);
        }));
  }

  SECTION("Cannot find any key in an empty hash map") {
    REQUIRE(all_of(
        d_pairs.begin(), d_pairs.end(),
        [view] __device__(thrust::pair<int32_t, int32_t> const& pair) mutable {
          return view.find(pair.first) == view.end();
        }));
  }

  SECTION("Keys are all found after inserting many keys.") {
    // Bulk insert keys
    thrust::for_each(
        d_pairs.begin(), d_pairs.end(),
        [view] __device__(auto const& pair) mutable { view.insert(pair); });

    // All keys should be found
    REQUIRE(all_of(
        d_pairs.begin(), d_pairs.end(),
        [view] __device__(thrust::pair<int32_t, int32_t> const& pair) mutable {
          return view.find(pair.first) != view.end();
        }));
  }
}
