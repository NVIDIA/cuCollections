/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cuco/static_multimap.cuh>
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

enum class dist_type { UNIQUE, DUAL, UNIFORM, GAUSSIAN };

template <dist_type Dist, typename Key, typename OutputIt>
static void generate_keys(OutputIt output_begin, OutputIt output_end)
{
  auto num_keys = std::distance(output_begin, output_end);

  std::random_device rd;
  std::mt19937 gen{rd()};

  switch (Dist) {
    case dist_type::UNIQUE: {
      for (auto i = 0; i < num_keys; ++i) {
        output_begin[i] = i;
      }
      break;
    }
    case dist_type::DUAL: {
      for (auto i = 0; i < num_keys; ++i) {
        output_begin[i] = i % (num_keys / 2);
      }
      break;
    }
    case dist_type::UNIFORM: {
      std::uniform_int_distribution<Key> distribution{0, std::numeric_limits<Key>::max()};
      for (auto i = 0; i < num_keys; ++i) {
        output_begin[i] = distribution(gen);
      }
      break;
    }
    case dist_type::GAUSSIAN: {
      std::normal_distribution<> dg{1e9, 1e7};
      for (auto i = 0; i < num_keys; ++i) {
        output_begin[i] = std::abs(static_cast<Key>(dg(gen)));
      }
      break;
    }
  }
}

TEMPLATE_TEST_CASE_SIG("Unique sequence of keys",
                       "",
                       ((typename T, dist_type Dist), T, Dist),
                       (int32_t, dist_type::UNIQUE),
                       (int64_t, dist_type::UNIQUE),
                       (int32_t, dist_type::UNIFORM),
                       (int64_t, dist_type::UNIFORM),
                       (int32_t, dist_type::GAUSSIAN),
                       (int64_t, dist_type::GAUSSIAN))
{
  using Key   = T;
  using Value = T;

  constexpr std::size_t num_keys{50'000'000};
  cuco::static_multimap<Key, Value> map{100'000'000, -1, -1};

  auto m_view = map.get_device_mutable_view();
  auto view   = map.get_device_view();

  std::vector<Key> h_keys(num_keys);
  std::vector<Value> h_values(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
    h_values[i]       = val;
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<Value> d_values(h_values);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);
  thrust::device_vector<Value> d_results(num_keys);
  thrust::device_vector<bool> d_contained(num_keys);

  SECTION("Keys are all found.")
  {
    // Bulk insert keys
    thrust::for_each(thrust::device,
                     d_pairs.begin(),
                     d_pairs.end(),
                     [m_view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
                       m_view.insert(pair);
                     });

    SECTION("non-const view")
    {
      REQUIRE(all_of(d_pairs.begin(),
                     d_pairs.end(),
                     [view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
                       return view.find_all(pair.first) != view.end();
                     }));
    }
    SECTION("const view")
    {
      REQUIRE(all_of(
        d_pairs.begin(), d_pairs.end(), [view] __device__(cuco::pair_type<Key, Value> const& pair) {
          return view.find_all(pair.first) != view.end();
        }));
    }
  }
}

TEMPLATE_TEST_CASE_SIG("Each key appears twice",
                       "",
                       ((typename T, dist_type Dist), T, Dist),
                       (int32_t, dist_type::DUAL),
                       (int64_t, dist_type::DUAL))
{
  using Key   = T;
  using Value = T;

  constexpr std::size_t num_keys{50'000'000};
  cuco::static_multimap<Key, Value> map{100'000'000, -1, -1};

  auto m_view = map.get_device_mutable_view();
  auto view   = map.get_device_view();

  std::vector<Key> h_keys(num_keys);
  std::vector<Value> h_values(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = i;
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
    h_values[i]       = val;
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<Value> d_values(h_values);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);
  thrust::device_vector<Value> d_results(num_keys);
  thrust::device_vector<bool> d_contained(num_keys);

  SECTION(
    "Counting the number of key/value pairs corresponding to each key should always return two.")
  {
    // Bulk insert keys
    map.insert(d_pairs.begin(), d_pairs.end());

    SECTION("non-const view")
    {
      REQUIRE(all_of(d_pairs.begin(),
                     d_pairs.end(),
                     [view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
                       return view.count(pair.first) == 2;
                     }));
    }
    SECTION("const view")
    {
      REQUIRE(all_of(
        d_pairs.begin(), d_pairs.end(), [view] __device__(cuco::pair_type<Key, Value> const& pair) {
          return view.count(pair.first) == 2;
        }));
    }
  }
}
