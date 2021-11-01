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

}  // namespace

enum class probe_sequence { linear_probing, double_hashing };

// User-defined key type
struct key_pair {
  int32_t a;
  int32_t b;
  __device__ bool operator!=(key_pair const& other) const { return a != other.a and b != other.b; }
};

struct hash_key_pair {
  __device__ uint32_t operator()(key_pair k) { return k.a; };
};

struct key_pair_equals {
  __device__ bool operator()(const key_pair& lhs, const key_pair& rhs)
  {
    return std::tie(lhs.a, lhs.b) == std::tie(rhs.a, rhs.b);
  }
};

struct value_pair {
  int32_t f;
  int32_t s;
};

template <typename Key, typename Value, typename Map, typename PairIt, typename KeyIt>
__inline__ void test_custom_key_value_type(Map& map,
                                           PairIt pair_begin,
                                           KeyIt key_begin,
                                           size_t num_pairs)
{
  constexpr cudaStream_t stream = 0;

  SECTION("All inserted keys-value pairs should be correctly recovered during find")
  {
    map.insert(pair_begin, pair_begin + num_pairs);

    auto res = map.get_size();
    REQUIRE(res == num_pairs);

    auto count = map.count(key_begin, key_begin + num_pairs, stream, key_pair_equals{});
    REQUIRE(count == num_pairs);

    thrust::device_vector<cuco::pair_type<Key, Value>> found_pairs(num_pairs);
    auto output_end = map.retrieve(
      key_begin, key_begin + num_pairs, found_pairs.begin(), stream, key_pair_equals{});
    auto size = output_end - found_pairs.begin();

    REQUIRE(size == num_pairs);

    // sort before compare
    thrust::sort(
      thrust::device,
      found_pairs.begin(),
      found_pairs.end(),
      [] __device__(const cuco::pair_type<Key, Value>& lhs,
                    const cuco::pair_type<Key, Value>& rhs) { return lhs.first.a < rhs.first.a; });

    REQUIRE(thrust::equal(
      thrust::device,
      pair_begin,
      pair_begin + num_pairs,
      found_pairs.begin(),
      [] __device__(cuco::pair_type<Key, Value> lhs, cuco::pair_type<Key, Value> rhs) {
        return lhs.first.a == rhs.first.a;
      }));
  }

  SECTION("Non-matches are not included in the output")
  {
    map.insert(pair_begin, pair_begin + num_pairs);

    auto const num = num_pairs * 2;
    thrust::device_vector<Key> query_keys(num);
    auto query_key_begin = query_keys.begin();
    thrust::transform(thrust::device,
                      thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(num),
                      query_key_begin,
                      [] __device__(auto i) {
                        return Key{i, i};
                      });

    auto count = map.count(query_key_begin, query_key_begin + num, stream, key_pair_equals{});
    REQUIRE(count == num_pairs);

    thrust::device_vector<cuco::pair_type<Key, Value>> found_pairs(num_pairs);
    auto output_end = map.retrieve(
      query_key_begin, query_key_begin + num, found_pairs.begin(), stream, key_pair_equals{});
    auto size = output_end - found_pairs.begin();

    REQUIRE(size == num_pairs);

    // sort before compare
    thrust::sort(
      thrust::device,
      found_pairs.begin(),
      found_pairs.end(),
      [] __device__(const cuco::pair_type<Key, Value>& lhs,
                    const cuco::pair_type<Key, Value>& rhs) { return lhs.first.a < rhs.first.a; });

    REQUIRE(thrust::equal(
      thrust::device,
      pair_begin,
      pair_begin + num_pairs,
      found_pairs.begin(),
      [] __device__(cuco::pair_type<Key, Value> lhs, cuco::pair_type<Key, Value> rhs) {
        return lhs.first.a == rhs.first.a;
      }));
  }

  SECTION("Outer functions include non-matches in the output")
  {
    map.insert(pair_begin, pair_begin + num_pairs);

    auto const num = num_pairs * 2;
    thrust::device_vector<Key> query_keys(num);
    auto query_key_begin = query_keys.begin();
    thrust::transform(thrust::device,
                      thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(num),
                      query_key_begin,
                      [] __device__(auto i) {
                        return Key{i, i};
                      });

    auto count_outer =
      map.count_outer(query_key_begin, query_key_begin + num, stream, key_pair_equals{});
    REQUIRE(count_outer == num);

    thrust::device_vector<cuco::pair_type<Key, Value>> found_pairs(num);
    auto output_end = map.retrieve_outer(
      query_key_begin, query_key_begin + num, found_pairs.begin(), stream, key_pair_equals{});
    auto size_outer = output_end - found_pairs.begin();

    REQUIRE(size_outer == num);
  }

  SECTION("All inserted keys-value pairs should be contained")
  {
    map.insert(pair_begin, pair_begin + num_pairs);

    auto size = map.get_size();
    REQUIRE(size == num_pairs);

    thrust::device_vector<bool> contained(num_pairs);
    map.contains(key_begin, key_begin + num_pairs, contained.begin(), stream, key_pair_equals{});
    REQUIRE(all_of(contained.begin(), contained.end(), [] __device__(bool const& b) { return b; }));
  }

  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    auto size = map.get_size();
    REQUIRE(size == 0);

    thrust::device_vector<bool> contained(num_pairs);
    map.contains(key_begin, key_begin + num_pairs, contained.begin(), stream, key_pair_equals{});
    REQUIRE(
      none_of(contained.begin(), contained.end(), [] __device__(bool const& b) { return b; }));
  }
}

TEMPLATE_TEST_CASE_SIG("User defined key and value type",
                       "",
                       ((probe_sequence Probe), Probe),
                       (probe_sequence::linear_probing),
                       (probe_sequence::double_hashing))
{
  using Key   = key_pair;
  using Value = value_pair;

  auto constexpr sentinel_key   = Key{-1, -1};
  auto constexpr sentinel_value = Value{-1, -1};

  constexpr std::size_t num_pairs = 100;
  constexpr std::size_t capacity  = num_pairs * 2;

  thrust::device_vector<Key> insert_keys(num_pairs);
  thrust::device_vector<Value> insert_values(num_pairs);

  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    insert_keys.begin(),
                    [] __device__(auto i) {
                      return Key{i, i};
                    });

  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    insert_values.begin(),
                    [] __device__(auto i) {
                      return Value{i, i};
                    });
  auto insert_pairs =
    thrust::make_zip_iterator(thrust::make_tuple(insert_keys.begin(), insert_values.begin()));

  if constexpr (Probe == probe_sequence::linear_probing) {
    cuco::static_multimap<key_pair,
                          value_pair,
                          cuda::thread_scope_device,
                          cuco::cuda_allocator<char>,
                          cuco::linear_probing<1, hash_key_pair>>
      map{capacity, sentinel_key, sentinel_value};
    test_custom_key_value_type<Key, Value>(map, insert_pairs, insert_keys.begin(), num_pairs);
  }
  if constexpr (Probe == probe_sequence::double_hashing) {
    cuco::static_multimap<Key, Value> map{capacity, sentinel_key, sentinel_value};
    test_custom_key_value_type<Key, Value>(map, insert_pairs, insert_keys.begin(), num_pairs);
  }
}

template <typename Key,
          typename Value,
          typename Map,
          typename PairIt,
          typename KeyIt,
          typename ResultIt>
__inline__ void test_multiplicity_two(
  Map& map, PairIt pair_begin, KeyIt key_begin, ResultIt result_begin, std::size_t num_items)
{
  auto num_keys = num_items / 2;
  thrust::device_vector<bool> d_contained(num_keys);

  SECTION("Non-inserted key/value pairs should not be contained.")
  {
    auto size = map.get_size();
    REQUIRE(size == 0);

    map.contains(key_begin, key_begin + num_keys, d_contained.begin());

    REQUIRE(
      none_of(d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b; }));
  }

  map.insert(pair_begin, pair_begin + num_items);

  SECTION("All inserted key/value pairs should be contained.")
  {
    auto size = map.get_size();
    REQUIRE(size == num_items);

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

    // sort before compare
    thrust::sort(thrust::device,
                 output_begin,
                 output_end,
                 [] __device__(const cuco::pair_type<Key, Value>& lhs,
                               const cuco::pair_type<Key, Value>& rhs) {
                   if (lhs.first != rhs.first) { return lhs.first < rhs.first; }
                   return lhs.second < rhs.second;
                 });

    REQUIRE(thrust::equal(
      thrust::device,
      pair_begin,
      pair_begin + num_items,
      output_begin,
      [] __device__(cuco::pair_type<Key, Value> lhs, cuco::pair_type<Key, Value> rhs) {
        return lhs.first == rhs.first and lhs.second == rhs.second;
      }));
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

    // sort before compare
    thrust::sort(thrust::device,
                 output_begin,
                 output_end,
                 [] __device__(const cuco::pair_type<Key, Value>& lhs,
                               const cuco::pair_type<Key, Value>& rhs) {
                   if (lhs.first != rhs.first) { return lhs.first < rhs.first; }
                   return lhs.second < rhs.second;
                 });

    REQUIRE(thrust::equal(
      thrust::device,
      pair_begin,
      pair_begin + num_items,
      output_begin,
      [] __device__(cuco::pair_type<Key, Value> lhs, cuco::pair_type<Key, Value> rhs) {
        return lhs.first == rhs.first and lhs.second == rhs.second;
      }));
  }
}

TEMPLATE_TEST_CASE_SIG("Multiplicity equals two",
                       "",
                       ((typename Key, typename Value, probe_sequence Probe), Key, Value, Probe),
                       (int32_t, int32_t, probe_sequence::linear_probing),
                       (int32_t, int64_t, probe_sequence::linear_probing),
                       (int64_t, int64_t, probe_sequence::linear_probing),
                       (int32_t, int32_t, probe_sequence::double_hashing),
                       (int32_t, int64_t, probe_sequence::double_hashing),
                       (int64_t, int64_t, probe_sequence::double_hashing))
{
  constexpr std::size_t num_items{4};

  thrust::device_vector<Key> d_keys(num_items / 2);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(num_items);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  // multiplicity = 2
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_items),
                    d_pairs.begin(),
                    [] __device__(auto i) {
                      return cuco::pair_type<Key, Value>{i / 2, i};
                    });

  thrust::device_vector<cuco::pair_type<Key, Value>> d_results(num_items);

  if constexpr (Probe == probe_sequence::linear_probing) {
    cuco::static_multimap<Key,
                          Value,
                          cuda::thread_scope_device,
                          cuco::cuda_allocator<char>,
                          cuco::linear_probing<1, cuco::detail::MurmurHash3_32<Key>>>
      map{5, -1, -1};
    test_multiplicity_two<Key, Value>(
      map, d_pairs.begin(), d_keys.begin(), d_results.begin(), num_items);
  }
  if constexpr (Probe == probe_sequence::double_hashing) {
    cuco::static_multimap<Key, Value> map{5, -1, -1};
    test_multiplicity_two<Key, Value>(
      map, d_pairs.begin(), d_keys.begin(), d_results.begin(), num_items);
  }
}

template <typename Key, typename Value, typename Map, typename PairIt, typename KeyIt>
__inline__ void test_non_matches(Map& map, PairIt pair_begin, KeyIt key_begin, std::size_t num_keys)
{
  map.insert(pair_begin, pair_begin + num_keys);

  auto res = map.get_size();
  REQUIRE(res == num_keys);

  SECTION("Output of count and retrieve should be coherent.")
  {
    auto num = map.count(key_begin, key_begin + num_keys);
    thrust::device_vector<cuco::pair_type<Key, Value>> d_results(num);

    REQUIRE(num == num_keys);

    auto output_begin = d_results.data().get();
    auto output_end   = map.retrieve(key_begin, key_begin + num_keys, output_begin);
    auto size         = thrust::distance(output_begin, output_end);

    REQUIRE(size == num_keys);

    // sort before compare
    thrust::sort(thrust::device,
                 output_begin,
                 output_end,
                 [] __device__(const cuco::pair_type<Key, Value>& lhs,
                               const cuco::pair_type<Key, Value>& rhs) {
                   if (lhs.first != rhs.first) { return lhs.first < rhs.first; }
                   return lhs.second < rhs.second;
                 });

    REQUIRE(thrust::equal(
      thrust::device,
      pair_begin,
      pair_begin + num_keys,
      output_begin,
      [] __device__(cuco::pair_type<Key, Value> lhs, cuco::pair_type<Key, Value> rhs) {
        return lhs.first == rhs.first and lhs.second == rhs.second;
      }));
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

    // sort before compare
    thrust::sort(thrust::device,
                 output_begin,
                 output_end,
                 [] __device__(const cuco::pair_type<Key, Value>& lhs,
                               const cuco::pair_type<Key, Value>& rhs) {
                   if (lhs.first != rhs.first) { return lhs.first < rhs.first; }
                   return lhs.second < rhs.second;
                 });

    // create gold reference
    thrust::device_vector<cuco::pair_type<Key, Value>> gold(size);
    auto gold_begin = gold.begin();
    thrust::transform(thrust::device,
                      thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(size),
                      gold_begin,
                      [num_keys] __device__(auto i) {
                        if (i < num_keys) { return cuco::pair_type<Key, Value>{i / 2, i}; }
                        return cuco::pair_type<Key, Value>{i - num_keys / 2, -1};
                      });

    REQUIRE(thrust::equal(
      thrust::device,
      gold_begin,
      gold_begin + size,
      output_begin,
      [] __device__(cuco::pair_type<Key, Value> lhs, cuco::pair_type<Key, Value> rhs) {
        return lhs.first == rhs.first and lhs.second == rhs.second;
      }));
  }
}

TEMPLATE_TEST_CASE_SIG("Tests of non-matches",
                       "",
                       ((typename Key, typename Value, probe_sequence Probe), Key, Value, Probe),
                       (int32_t, int32_t, probe_sequence::linear_probing),
                       (int32_t, int64_t, probe_sequence::linear_probing),
                       (int64_t, int64_t, probe_sequence::linear_probing),
                       (int32_t, int32_t, probe_sequence::double_hashing),
                       (int32_t, int64_t, probe_sequence::double_hashing),
                       (int64_t, int64_t, probe_sequence::double_hashing))
{
  constexpr std::size_t num_keys{1'000'000};

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  // multiplicity = 2
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_keys),
                    d_pairs.begin(),
                    [] __device__(auto i) {
                      return cuco::pair_type<Key, Value>{i / 2, i};
                    });

  if constexpr (Probe == probe_sequence::linear_probing) {
    cuco::static_multimap<Key,
                          Value,
                          cuda::thread_scope_device,
                          cuco::cuda_allocator<char>,
                          cuco::linear_probing<1, cuco::detail::MurmurHash3_32<Key>>>
      map{num_keys * 2, -1, -1};
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

  auto res = map.get_size();
  REQUIRE(res * 2 == size);

  auto num = map.count(key_begin, key_begin + size);
  REQUIRE(num * 2 == size);
}

TEMPLATE_TEST_CASE_SIG("Tests of insert_if",
                       "",
                       ((typename Key, typename Value, probe_sequence Probe), Key, Value, Probe),
                       (int32_t, int32_t, probe_sequence::linear_probing),
                       (int32_t, int64_t, probe_sequence::linear_probing),
                       (int64_t, int64_t, probe_sequence::linear_probing),
                       (int32_t, int32_t, probe_sequence::double_hashing),
                       (int32_t, int64_t, probe_sequence::double_hashing),
                       (int64_t, int64_t, probe_sequence::double_hashing))
{
  constexpr std::size_t num_keys{1'000};

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  // multiplicity = 1
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_keys),
                    d_pairs.begin(),
                    [] __device__(auto i) {
                      return cuco::pair_type<Key, Value>{i, i};
                    });

  if constexpr (Probe == probe_sequence::linear_probing) {
    cuco::static_multimap<Key,
                          Value,
                          cuda::thread_scope_device,
                          cuco::cuda_allocator<char>,
                          cuco::linear_probing<1, cuco::detail::MurmurHash3_32<Key>>>
      map{num_keys * 2, -1, -1};
    test_insert_if<Key>(map, d_pairs.begin(), d_keys.begin(), num_keys);
  }
  if constexpr (Probe == probe_sequence::double_hashing) {
    cuco::static_multimap<Key, Value> map{num_keys * 2, -1, -1};
    test_insert_if<Key>(map, d_pairs.begin(), d_keys.begin(), num_keys);
  }
}

// Custom pair equal
template <typename Key, typename Value>
struct pair_equal {
  __device__ bool operator()(const cuco::pair_type<Key, Value>& lhs,
                             const cuco::pair_type<Key, Value>& rhs) const
  {
    return lhs.first == rhs.first;
  }
};

template <typename Key, typename Value, typename Map, typename PairIt>
__inline__ void test_pair_functions(Map& map, PairIt pair_begin, std::size_t num_pairs)
{
  map.insert(pair_begin, pair_begin + num_pairs);
  cudaStreamSynchronize(0);

  auto res = map.get_size();
  REQUIRE(res == num_pairs);

  // query pair matching rate = 50%
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    pair_begin,
                    [] __device__(auto i) {
                      return cuco::pair_type<Key, Value>{i, i};
                    });

  SECTION("Output of pair_count and pair_retrieve should be coherent.")
  {
    auto num = map.pair_count(pair_begin, pair_begin + num_pairs, pair_equal<Key, Value>{});

    auto out1_begin = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
    auto out2_begin = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

    REQUIRE(num == num_pairs);

    auto [out1_end, out2_end] = map.pair_retrieve(
      pair_begin, pair_begin + num_pairs, out1_begin, out2_begin, pair_equal<Key, Value>{});

    REQUIRE((out1_end - out1_begin) == num_pairs);
  }

  SECTION("Output of pair_count_outer and pair_retrieve_outer should be coherent.")
  {
    auto num = map.pair_count_outer(pair_begin, pair_begin + num_pairs, pair_equal<Key, Value>{});

    auto out1_begin = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
    auto out2_begin = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

    REQUIRE(num == (num_pairs + num_pairs / 2));

    auto [out1_end, out2_end] = map.pair_retrieve_outer(
      pair_begin, pair_begin + num_pairs, out1_begin, out2_begin, pair_equal<Key, Value>{});

    REQUIRE((out1_end - out1_begin) == (num_pairs + num_pairs / 2));
  }
}

TEMPLATE_TEST_CASE_SIG("Tests of pair functions",
                       "",
                       ((typename Key, typename Value, probe_sequence Probe), Key, Value, Probe),
                       (int32_t, int32_t, probe_sequence::linear_probing),
                       (int32_t, int64_t, probe_sequence::linear_probing),
                       (int64_t, int64_t, probe_sequence::linear_probing),
                       (int32_t, int32_t, probe_sequence::double_hashing),
                       (int32_t, int64_t, probe_sequence::double_hashing),
                       (int64_t, int64_t, probe_sequence::double_hashing))
{
  constexpr std::size_t num_pairs{4};
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(num_pairs);

  // pair multiplicity = 2
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    d_pairs.begin(),
                    [] __device__(auto i) {
                      return cuco::pair_type<Key, Value>{i / 2, i};
                    });

  if constexpr (Probe == probe_sequence::linear_probing) {
    cuco::static_multimap<Key,
                          Value,
                          cuda::thread_scope_device,
                          cuco::cuda_allocator<char>,
                          cuco::linear_probing<1, cuco::detail::MurmurHash3_32<Key>>>
      map{num_pairs * 2, -1, -1};
    test_pair_functions<Key, Value>(map, d_pairs.begin(), num_pairs);
  }
  if constexpr (Probe == probe_sequence::double_hashing) {
    cuco::static_multimap<Key, Value> map{num_pairs * 2, -1, -1};
    test_pair_functions<Key, Value>(map, d_pairs.begin(), num_pairs);
  }
}
