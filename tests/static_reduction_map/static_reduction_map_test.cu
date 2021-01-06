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

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cuco/static_reduction_map.cuh>
#include <limits>

namespace {
namespace cg = cooperative_groups;

// Thrust logical algorithms (any_of/all_of/none_of) don't work with device
// lambdas: See https://github.com/thrust/thrust/issues/1062
template <typename Iterator, typename Predicate>
bool all_of(Iterator begin, Iterator end, Predicate p)
{
  auto size = thrust::distance(begin, end);
  return size == thrust::count_if(begin, end, p);
}

template <typename Iterator, typename Predicate>
bool any_of(Iterator begin, Iterator end, Predicate p)
{
  return thrust::count_if(begin, end, p) > 0;
}

template <typename Iterator, typename Predicate>
bool none_of(Iterator begin, Iterator end, Predicate p)
{
  return not all_of(begin, end, p);
}
}  // namespace

enum class dist_type { UNIQUE, UNIFORM, GAUSSIAN };

template <dist_type Dist, typename Key, typename OutputIt>
static void generate_keys(OutputIt output_begin, OutputIt output_end)
{
  auto num_keys = std::distance(output_begin, output_end);

  std::random_device rd;
  std::mt19937 gen{rd()};

  switch (Dist) {
    case dist_type::UNIQUE:
      for (auto i = 0; i < num_keys; ++i) {
        output_begin[i] = i;
      }
      break;
    case dist_type::UNIFORM:
      for (auto i = 0; i < num_keys; ++i) {
        output_begin[i] = std::abs(static_cast<Key>(gen()));
      }
      break;
    case dist_type::GAUSSIAN:
      std::normal_distribution<> dg{1e9, 1e7};
      for (auto i = 0; i < num_keys; ++i) {
        output_begin[i] = std::abs(static_cast<Key>(dg(gen)));
      }
      break;
  }
}

TEMPLATE_TEST_CASE_SIG("Unique sequence of keys",
                       "",
                       ((typename T, dist_type Dist), T, Dist),
                       (int32_t, dist_type::UNIQUE))
{
  using Key   = T;
  using Value = T;

  constexpr std::size_t num_slots{200};
  cuco::static_reduction_map<cuco::reduce_add<Value>, Key, Value> map{num_slots, -1};

  SECTION("Inserting identical keys")
  {
    thrust::device_vector<Key> keys(100, 42);
    thrust::device_vector<Value> values(keys.size(), 1);
    auto zip     = thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), values.begin()));
    auto zip_end = zip + keys.size();
    map.insert(zip, zip_end);

    SECTION("There should only be one key in the map") { REQUIRE(map.get_size() == 1); }

    SECTION("Map should contain the inserted key")
    {
      thrust::device_vector<bool> contained(keys.size());
      map.contains(keys.begin(), keys.end(), contained.begin());
      REQUIRE(all_of(contained.begin(), contained.end(), [] __device__(bool c) { return c; }));
    }

    SECTION("Found value should equal aggregate of inserted values")
    {
      thrust::device_vector<Value> found(keys.size());
      map.find(keys.begin(), keys.end(), found.begin());
      auto const expected_aggregate = keys.size();  // All keys inserted "1", so the
                                                    // sum aggregate should be
                                                    // equal to the number of keys inserted
      REQUIRE(all_of(found.begin(), found.end(), [expected_aggregate] __device__(Value v) {
        return v == expected_aggregate;
      }));
    }
  }
}
