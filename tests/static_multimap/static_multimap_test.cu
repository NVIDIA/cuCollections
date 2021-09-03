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
#include <thrust/iterator/discard_iterator.h>
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

template <typename Key, typename Value>
struct pair_equal {
  __host__ __device__ bool operator()(const cuco::pair_type<Key, Value>& lhs,
                                      const cuco::pair_type<Key, Value>& rhs) const
  {
    return lhs.first == rhs.first;
  }
};

}  // namespace

enum class dist_type { UNIQUE, DUAL, UNIFORM, GAUSSIAN };

template <dist_type Dist, typename Key, typename OutputIt>
static void generate_keys(OutputIt output_begin, OutputIt output_end)
{
  auto num_items = std::distance(output_begin, output_end);

  std::random_device rd;
  std::mt19937 gen{rd()};

  switch (Dist) {
    case dist_type::UNIQUE: {
      for (auto i = 0; i < num_items; ++i) {
        output_begin[i] = i;
      }
      break;
    }
    case dist_type::DUAL: {
      for (auto i = 0; i < num_items; ++i) {
        output_begin[i] = i % (num_items / 2);
      }
      break;
    }
  }
}

TEMPLATE_TEST_CASE_SIG("Each key appears twice",
                       "",
                       ((typename Key, typename Value, dist_type Dist), Key, Value, Dist),
                       (int32_t, int32_t, dist_type::DUAL),
                       (int32_t, int64_t, dist_type::DUAL),
                       (int64_t, int64_t, dist_type::DUAL))
{
  constexpr std::size_t num_items{400};
  cuco::static_multimap<Key, Value> map{500, -1, -1};

  std::vector<Key> h_keys(num_items);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_items);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_items; ++i) {
    Key key           = h_keys[i];
    Value val         = i;
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_results(num_items);

  // Get unique keys
  std::set<Key> key_set(h_keys.begin(), h_keys.end());
  std::vector<Key> h_unique_keys(key_set.begin(), key_set.end());
  thrust::device_vector<Key> d_unique_keys(h_unique_keys);

  thrust::device_vector<bool> d_contained(num_items / 2);

  SECTION("Non-inserted key/value pairs should not be contained.")
  {
    map.contains(d_unique_keys.begin(), d_unique_keys.end(), d_contained.begin());

    REQUIRE(
      none_of(d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b; }));
  }

  map.insert(d_pairs.begin(), d_pairs.end());

  SECTION("All inserted key/value pairs should be contained.")
  {
    map.contains(d_unique_keys.begin(), d_unique_keys.end(), d_contained.begin());

    REQUIRE(
      all_of(d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b; }));
  }

  SECTION("Total count should be equal to the number of inserted pairs.")
  {
    // Count matching keys
    auto num = map.count(d_unique_keys.begin(), d_unique_keys.end());

    REQUIRE(num == num_items);

    auto output_begin = d_results.data().get();
    auto output_end   = map.retrieve(d_unique_keys.begin(), d_unique_keys.end(), output_begin);
    auto size         = thrust::distance(output_begin, output_end);

    REQUIRE(size == num_items);
  }

  SECTION("count and count_outer should return the same value.")
  {
    auto num       = map.count(d_unique_keys.begin(), d_unique_keys.end());
    auto num_outer = map.count_outer(d_unique_keys.begin(), d_unique_keys.end());

    REQUIRE(num == num_outer);
  }

  SECTION("Output size of retrieve and retrieve_outer should be the same.")
  {
    auto output_begin = d_results.data().get();
    auto output_end   = map.retrieve(d_unique_keys.begin(), d_unique_keys.end(), output_begin);
    auto size         = thrust::distance(output_begin, output_end);

    output_end      = map.retrieve_outer(d_unique_keys.begin(), d_unique_keys.end(), output_begin);
    auto size_outer = thrust::distance(output_begin, output_end);

    REQUIRE(size == size_outer);
  }
}

TEMPLATE_TEST_CASE_SIG("Handling of non-matches",
                       "",
                       ((typename Key, typename Value, dist_type Dist), Key, Value, Dist),
                       (int32_t, int32_t, dist_type::UNIQUE),
                       (int32_t, int64_t, dist_type::UNIQUE),
                       (int64_t, int64_t, dist_type::UNIQUE))
{
  constexpr std::size_t num_keys{1'000'000};
  cuco::static_multimap<Key, Value> map{2'000'000, -1, -1};

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_keys; ++i) {
    h_pairs[i].first  = h_keys[i] / 2;
    h_pairs[i].second = h_keys[i];
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  map.insert(d_pairs.begin(), d_pairs.end());

  SECTION("Output of count and retrieve should be coherent.")
  {
    auto num = map.count(d_keys.begin(), d_keys.end());
    thrust::device_vector<cuco::pair_type<Key, Value>> d_results(num);

    REQUIRE(num == num_keys);

    auto output_begin = d_results.data().get();
    auto output_end   = map.retrieve(d_keys.begin(), d_keys.end(), output_begin);
    auto size         = thrust::distance(output_begin, output_end);

    REQUIRE(num == size);
  }

  SECTION("Output of count_outer and retrieve_outer should be coherent.")
  {
    auto num = map.count_outer(d_keys.begin(), d_keys.end());
    thrust::device_vector<cuco::pair_type<Key, Value>> d_results(num);

    REQUIRE(num == (num_keys + num_keys / 2));

    auto output_begin = d_results.data().get();
    auto output_end   = map.retrieve_outer(d_keys.begin(), d_keys.end(), output_begin);
    auto size         = thrust::distance(output_begin, output_end);

    REQUIRE(num == size);
  }

  SECTION("count_outer handles non-matches wile count doesn't.")
  {
    auto num_outer = map.count_outer(d_keys.begin(), d_keys.end());
    auto num       = map.count(d_keys.begin(), d_keys.end());

    REQUIRE(num_outer == (num + num_keys / 2));
  }

  SECTION("retrieve_outer handles non-matches wile retrieve doesn't.")
  {
    auto num_outer = map.count_outer(d_keys.begin(), d_keys.end());
    thrust::device_vector<cuco::pair_type<Key, Value>> d_results(num_outer);

    auto output_begin = d_results.data().get();
    auto output_end   = map.retrieve(d_keys.begin(), d_keys.end(), output_begin);
    auto size         = thrust::distance(output_begin, output_end);

    output_end      = map.retrieve_outer(d_keys.begin(), d_keys.end(), output_begin);
    auto size_outer = thrust::distance(output_begin, output_end);

    REQUIRE(size_outer == (size + num_keys / 2));
  }
}

TEMPLATE_TEST_CASE_SIG("Tests of insert_if",
                       "",
                       ((typename Key, typename Value, dist_type Dist), Key, Value, Dist),
                       (int32_t, int32_t, dist_type::UNIQUE),
                       (int32_t, int64_t, dist_type::UNIQUE),
                       (int64_t, int64_t, dist_type::UNIQUE))
{
  constexpr std::size_t num_keys{1'000'000};
  cuco::static_multimap<Key, Value> map{2'000'000, -1, -1};

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_keys; ++i) {
    h_pairs[i].first  = h_keys[i];
    h_pairs[i].second = h_keys[i];
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  auto pred_lambda = [] __device__(Key k) { return k % 2 == 0; };
  map.insert_if(d_pairs.begin(), d_pairs.begin() + d_pairs.size(), d_keys.begin(), pred_lambda);

  auto num = map.count(d_keys.begin(), d_keys.end());

  REQUIRE(num * 2 == num_keys);
}

TEMPLATE_TEST_CASE_SIG("Evaluation of pair functions",
                       "",
                       ((typename Key, typename Value, dist_type Dist), Key, Value, Dist),
                       (int32_t, int32_t, dist_type::UNIQUE),
                       (int32_t, int64_t, dist_type::UNIQUE),
                       (int64_t, int64_t, dist_type::UNIQUE))
{
  constexpr std::size_t num_pairs{5'000'000};
  cuco::static_multimap<Key, Value> map{10'000'000, -1, -1};

  std::vector<Key> h_keys(num_pairs);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_pairs);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_pairs; ++i) {
    h_pairs[i].first  = h_keys[i] / 2;
    h_pairs[i].second = h_keys[i];
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  map.insert(d_pairs.begin(), d_pairs.end());

  for (auto i = 0; i < num_pairs; ++i) {
    h_pairs[i].first = h_keys[i];
  }
  d_pairs = h_pairs;

  SECTION("pair_count_outer handles non-matches wile pair_count doesn't.")
  {
    auto num_outer = map.pair_count_outer(d_pairs.begin(), d_pairs.end(), pair_equal<Key, Value>{});
    auto num       = map.pair_count(d_pairs.begin(), d_pairs.end(), pair_equal<Key, Value>{});

    REQUIRE(num_outer == (num + num_pairs / 2));
  }

  SECTION("Output of pair_count and pair_retrieve should be coherent.")
  {
    auto num = map.pair_count(d_pairs.begin(), d_pairs.end(), pair_equal<Key, Value>{});

    auto out1_zip = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
    auto out2_zip = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

    REQUIRE(num == num_pairs);

    auto size = map.pair_retrieve(
      d_pairs.begin(), d_pairs.end(), out1_zip, out2_zip, pair_equal<Key, Value>{});

    REQUIRE(num == size);
  }

  SECTION("Output of pair_count_outer and pair_retrieve_outer should be coherent.")
  {
    auto num = map.pair_count_outer(d_pairs.begin(), d_pairs.end(), pair_equal<Key, Value>{});

    auto out1_zip = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
    auto out2_zip = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

    REQUIRE(num == num_pairs * 1.5);

    auto size = map.pair_retrieve_outer(
      d_pairs.begin(), d_pairs.end(), out1_zip, out2_zip, pair_equal<Key, Value>{});

    REQUIRE(num == size);
  }
}

TEMPLATE_TEST_CASE_SIG("Evaluation of small test cases",
                       "",
                       ((typename Key, typename Value, dist_type Dist), Key, Value, Dist),
                       (int32_t, int32_t, dist_type::UNIQUE),
                       (int32_t, int64_t, dist_type::UNIQUE),
                       (int64_t, int64_t, dist_type::UNIQUE))
{
  constexpr std::size_t num_items{3};
  cuco::static_multimap<Key, Value> map{2 * num_items, -1, -1};

  std::vector<Key> h_keys(num_items);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_items);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_items; ++i) {
    h_pairs[i].first  = h_keys[i] / 2;
    h_pairs[i].second = h_keys[i];
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  map.insert(d_pairs.begin(), d_pairs.end());

  for (auto i = 0; i < num_items; ++i) {
    h_pairs[i].first = h_keys[i];
  }
  d_pairs = h_pairs;

  SECTION("Output of count and retrieve should be coherent.")
  {
    auto num = map.count(d_keys.begin(), d_keys.end());
    thrust::device_vector<cuco::pair_type<Key, Value>> d_results(num);

    REQUIRE(num == num_items);

    auto output_begin = d_results.data().get();
    auto output_end   = map.retrieve(d_keys.begin(), d_keys.end(), output_begin);
    auto size         = thrust::distance(output_begin, output_end);

    REQUIRE(num == size);
  }

  SECTION("Output of count_outer and retrieve_outer should be coherent.")
  {
    auto num = map.count_outer(d_keys.begin(), d_keys.end());
    thrust::device_vector<cuco::pair_type<Key, Value>> d_results(num);

    REQUIRE(num == num_items + 1);

    auto output_begin = d_results.data().get();
    auto output_end   = map.retrieve_outer(d_keys.begin(), d_keys.end(), output_begin);
    auto size         = thrust::distance(output_begin, output_end);

    REQUIRE(num == size);
  }

  SECTION("Output of pair_count and pair_retrieve should be coherent.")
  {
    auto num = map.pair_count(d_pairs.begin(), d_pairs.end(), pair_equal<Key, Value>{});

    auto out1_zip = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
    auto out2_zip = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

    REQUIRE(num == num_items);

    auto size = map.pair_retrieve(
      d_pairs.begin(), d_pairs.end(), out1_zip, out2_zip, pair_equal<Key, Value>{});

    REQUIRE(num == size);
  }

  SECTION("Output of pair_count_outer and pair_retrieve_outer should be coherent.")
  {
    auto num = map.pair_count_outer(d_pairs.begin(), d_pairs.end(), pair_equal<Key, Value>{});

    auto out1_zip = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
    auto out2_zip = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

    REQUIRE(num == num_items + 1);

    auto size = map.pair_retrieve_outer(
      d_pairs.begin(), d_pairs.end(), out1_zip, out2_zip, pair_equal<Key, Value>{});

    REQUIRE(num == size);
  }
}
