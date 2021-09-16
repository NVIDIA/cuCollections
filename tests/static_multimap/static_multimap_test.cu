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

#include <algorithm>
#include <limits>

#include <catch2/catch.hpp>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/discard_iterator.h>

#include <cuco/static_multimap.cuh>

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
enum class probe_sequence { linear_probing, double_hashing };

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

template <typename Map, typename PairIt, typename KeyIt, typename ResultIt>
__inline__ void test_multiplicity_two(
  Map& map, PairIt pair_begin, KeyIt key_begin, ResultIt result_begin, std::size_t num_items)
{
  auto num_keys = num_items / 2;
  thrust::device_vector<bool> d_contained(num_keys);

  SECTION("Non-inserted key/value pairs should not be contained.")
  {
    map.contains(key_begin, key_begin + num_keys, d_contained.begin());

    REQUIRE(
      none_of(d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b; }));
  }

  map.insert(pair_begin, pair_begin + num_items);

  SECTION("All inserted key/value pairs should be contained.")
  {
    map.contains(key_begin, key_begin + num_keys, d_contained.begin());

    REQUIRE(
      all_of(d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b; }));
  }

  SECTION("Total count should be equal to the number of inserted pairs.")
  {
    // Count matching keys
    auto num = map.count(key_begin, key_begin + num_keys);

    REQUIRE(num == num_items);

    auto output_begin = result_begin;
    auto output_end   = map.retrieve(key_begin, key_begin + num_keys, output_begin);
    auto size         = thrust::distance(output_begin, output_end);

    REQUIRE(size == num_items);
  }

  SECTION("count and count_outer should return the same value.")
  {
    auto num       = map.count(key_begin, key_begin + num_keys);
    auto num_outer = map.count_outer(key_begin, key_begin + num_keys);

    REQUIRE(num == num_outer);
  }

  SECTION("Output of retrieve and retrieve_outer should be the same.")
  {
    auto output_begin = result_begin;
    auto output_end   = map.retrieve(key_begin, key_begin + num_keys, output_begin);
    auto size         = thrust::distance(output_begin, output_end);

    output_end      = map.retrieve_outer(key_begin, key_begin + num_keys, output_begin);
    auto size_outer = thrust::distance(output_begin, output_end);

    REQUIRE(size == size_outer);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Multiplicity equals two",
  "",
  ((typename Key, typename Value, dist_type Dist, probe_sequence Probe), Key, Value, Dist, Probe),
  (int32_t, int32_t, dist_type::DUAL, probe_sequence::linear_probing),
  (int32_t, int64_t, dist_type::DUAL, probe_sequence::linear_probing),
  (int64_t, int64_t, dist_type::DUAL, probe_sequence::linear_probing),
  (int32_t, int32_t, dist_type::DUAL, probe_sequence::double_hashing),
  (int32_t, int64_t, dist_type::DUAL, probe_sequence::double_hashing),
  (int64_t, int64_t, dist_type::DUAL, probe_sequence::double_hashing))
{
  constexpr std::size_t num_items{4};

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

  // Get unique keys
  std::set<Key> key_set(h_keys.begin(), h_keys.end());
  std::vector<Key> h_unique_keys(key_set.begin(), key_set.end());
  thrust::device_vector<Key> d_unique_keys(h_unique_keys);

  thrust::device_vector<cuco::pair_type<Key, Value>> d_results(num_items);

  if constexpr (Probe == probe_sequence::linear_probing) {
    cuco::static_multimap<Key, Value, cuco::detail::linear_probing<Key, Value, 1>> map{5, -1, -1};
    test_multiplicity_two(
      map, d_pairs.begin(), d_unique_keys.begin(), d_results.begin(), num_items);
  }
  if constexpr (Probe == probe_sequence::double_hashing) {
    cuco::static_multimap<Key, Value> map{5, -1, -1};
    test_multiplicity_two(
      map, d_pairs.begin(), d_unique_keys.begin(), d_results.begin(), num_items);
  }
}

template <typename Key, typename Value, typename Map, typename PairIt, typename KeyIt>
__inline__ void test_non_matches(Map& map, PairIt pair_begin, KeyIt key_begin, std::size_t num_keys)
{
  map.insert(pair_begin, pair_begin + num_keys);

  SECTION("Output of count and retrieve should be coherent.")
  {
    auto num = map.count(key_begin, key_begin + num_keys);
    thrust::device_vector<cuco::pair_type<Key, Value>> d_results(num);

    REQUIRE(num == num_keys);

    auto output_begin = d_results.data().get();
    auto output_end   = map.retrieve(key_begin, key_begin + num_keys, output_begin);
    auto size         = thrust::distance(output_begin, output_end);

    REQUIRE(size == num_keys);
  }

  SECTION("Output of count_outer and retrieve_outer should be coherent.")
  {
    auto num = map.count_outer(key_begin, key_begin + num_keys);
    thrust::device_vector<cuco::pair_type<Key, Value>> d_results(num);

    REQUIRE(num == (num_keys + num_keys / 2));

    auto output_begin = d_results.data().get();
    auto output_end   = map.retrieve_outer(key_begin, key_begin + num_keys, output_begin);
    auto size         = thrust::distance(output_begin, output_end);

    REQUIRE(size == (num_keys + num_keys / 2));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Tests of non-matches",
  "",
  ((typename Key, typename Value, dist_type Dist, probe_sequence Probe), Key, Value, Dist, Probe),
  (int32_t, int32_t, dist_type::UNIQUE, probe_sequence::linear_probing),
  (int32_t, int64_t, dist_type::UNIQUE, probe_sequence::linear_probing),
  (int64_t, int64_t, dist_type::UNIQUE, probe_sequence::linear_probing),
  (int32_t, int32_t, dist_type::UNIQUE, probe_sequence::double_hashing),
  (int32_t, int64_t, dist_type::UNIQUE, probe_sequence::double_hashing),
  (int64_t, int64_t, dist_type::UNIQUE, probe_sequence::double_hashing))
{
  constexpr std::size_t num_keys{1'000'000};

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_keys; ++i) {
    h_pairs[i].first  = h_keys[i] / 2;
    h_pairs[i].second = h_keys[i];
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  if constexpr (Probe == probe_sequence::linear_probing) {
    cuco::static_multimap<Key, Value, cuco::detail::linear_probing<Key, Value, 1>> map{
      num_keys * 2, -1, -1};
    test_non_matches<Key, Value>(map, d_pairs.begin(), d_keys.begin(), num_keys);
  }
  if constexpr (Probe == probe_sequence::double_hashing) {
    cuco::static_multimap<Key, Value> map{num_keys * 2, -1, -1};
    test_non_matches<Key, Value>(map, d_pairs.begin(), d_keys.begin(), num_keys);
  }
}

template <typename Key, typename Map, typename PairIt, typename KeyIt>
__inline__ void test_insert_if(Map& map, PairIt pair_begin, KeyIt key_begin, std::size_t size)
{
  // 50% insertion
  auto pred_lambda = [] __device__(Key k) { return k % 2 == 0; };

  map.insert_if(pair_begin, pair_begin + size, key_begin, pred_lambda);

  auto num = map.count(key_begin, key_begin + size);

  REQUIRE(num * 2 == size);
}

TEMPLATE_TEST_CASE_SIG(
  "Tests of insert_if",
  "",
  ((typename Key, typename Value, dist_type Dist, probe_sequence Probe), Key, Value, Dist, Probe),
  (int32_t, int32_t, dist_type::UNIQUE, probe_sequence::linear_probing),
  (int32_t, int64_t, dist_type::UNIQUE, probe_sequence::linear_probing),
  (int64_t, int64_t, dist_type::UNIQUE, probe_sequence::linear_probing),
  (int32_t, int32_t, dist_type::UNIQUE, probe_sequence::double_hashing),
  (int32_t, int64_t, dist_type::UNIQUE, probe_sequence::double_hashing),
  (int64_t, int64_t, dist_type::UNIQUE, probe_sequence::double_hashing))
{
  constexpr std::size_t num_keys{1'000};

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_keys; ++i) {
    h_pairs[i].first  = h_keys[i];
    h_pairs[i].second = h_keys[i];
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  if constexpr (Probe == probe_sequence::linear_probing) {
    cuco::static_multimap<Key, Value, cuco::detail::linear_probing<Key, Value, 1>> map{
      num_keys * 2, -1, -1};
    test_insert_if<Key>(map, d_pairs.begin(), d_keys.begin(), num_keys);
  }
  if constexpr (Probe == probe_sequence::double_hashing) {
    cuco::static_multimap<Key, Value> map{num_keys * 2, -1, -1};
    test_insert_if<Key>(map, d_pairs.begin(), d_keys.begin(), num_keys);
  }
}

template <typename Map, typename Key, typename Value>
__inline__ void test_pair_functions(Map& map,
                                    std::vector<cuco::pair_type<Key, Value>>& h_pairs,
                                    thrust::device_vector<cuco::pair_type<Key, Value>>& d_pairs,
                                    std::size_t num_pairs)
{
  map.insert(d_pairs.begin(), d_pairs.end());

  for (auto i = 0; i < num_pairs; ++i) {
    h_pairs[i].first = i;
  }
  d_pairs = h_pairs;

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

    REQUIRE(size == num_pairs);
  }

  SECTION("Output of pair_count_outer and pair_retrieve_outer should be coherent.")
  {
    auto num = map.pair_count_outer(d_pairs.begin(), d_pairs.end(), pair_equal<Key, Value>{});

    auto out1_zip = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
    auto out2_zip = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

    REQUIRE(num == (num_pairs + num_pairs / 2));

    auto size = map.pair_retrieve_outer(
      d_pairs.begin(), d_pairs.end(), out1_zip, out2_zip, pair_equal<Key, Value>{});

    REQUIRE(size == (num_pairs + num_pairs / 2));
  }
}

TEMPLATE_TEST_CASE_SIG(
  "Tests of pair functions",
  "",
  ((typename Key, typename Value, dist_type Dist, probe_sequence Probe), Key, Value, Dist, Probe),
  (int32_t, int32_t, dist_type::UNIQUE, probe_sequence::linear_probing),
  (int32_t, int64_t, dist_type::UNIQUE, probe_sequence::linear_probing),
  (int64_t, int64_t, dist_type::UNIQUE, probe_sequence::linear_probing),
  (int32_t, int32_t, dist_type::UNIQUE, probe_sequence::double_hashing),
  (int32_t, int64_t, dist_type::UNIQUE, probe_sequence::double_hashing),
  (int64_t, int64_t, dist_type::UNIQUE, probe_sequence::double_hashing))
{
  constexpr std::size_t num_pairs{4};

  std::vector<Key> h_keys(num_pairs);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_pairs);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_pairs; ++i) {
    h_pairs[i].first  = h_keys[i] / 2;
    h_pairs[i].second = h_keys[i];
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  if constexpr (Probe == probe_sequence::linear_probing) {
    cuco::static_multimap<Key, Value, cuco::detail::linear_probing<Key, Value, 1>> map{
      num_pairs * 2, -1, -1};
    test_pair_functions(map, h_pairs, d_pairs, num_pairs);
  }
  if constexpr (Probe == probe_sequence::double_hashing) {
    cuco::static_multimap<Key, Value> map{num_pairs * 2, -1, -1};
    test_pair_functions(map, h_pairs, d_pairs, num_pairs);
  }
}
