/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cuco/static_multimap.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <catch2/catch_template_test_macros.hpp>

#include <tuple>

// User-defined key type
struct key_pair {
  int32_t a;
  int32_t b;
  __device__ bool operator!=(key_pair const& other) const { return a != other.a and b != other.b; }
};

struct hash_key_pair {
  __host__ __device__ hash_key_pair() : hash_key_pair{0} {}
  __host__ __device__ hash_key_pair(uint32_t offset) : offset_(offset) {}
  __device__ uint32_t operator()(key_pair k) const { return k.a + offset_; };
  uint32_t offset_;
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

template <typename Map>
__inline__ void test_custom_key_value_type(Map& map, std::size_t num_pairs)
{
  using Key   = key_pair;
  using Value = value_pair;

  constexpr cudaStream_t stream = 0;

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

  auto pair_begin =
    thrust::make_zip_iterator(thrust::make_tuple(insert_keys.begin(), insert_values.begin()));
  auto key_begin = insert_keys.begin();

  SECTION("All inserted keys-value pairs should be correctly recovered during find")
  {
    map.insert(pair_begin, pair_begin + num_pairs);

    auto res = map.get_size();
    REQUIRE(res == num_pairs);

    auto count = map.count(key_begin, key_begin + num_pairs, stream, key_pair_equals{});
    REQUIRE(count == num_pairs);

    thrust::device_vector<cuco::pair<Key, Value>> found_pairs(num_pairs);
    auto output_end = map.retrieve(
      key_begin, key_begin + num_pairs, found_pairs.begin(), stream, key_pair_equals{});
    std::size_t const size = std::distance(found_pairs.begin(), output_end);

    REQUIRE(size == num_pairs);

    // sort before compare
    thrust::sort(
      thrust::device,
      found_pairs.begin(),
      found_pairs.end(),
      [] __device__(const cuco::pair<Key, Value>& lhs, const cuco::pair<Key, Value>& rhs) {
        return lhs.first.a < rhs.first.a;
      });

    REQUIRE(
      cuco::test::equal(pair_begin,
                        pair_begin + num_pairs,
                        found_pairs.begin(),
                        [] __device__(cuco::pair<Key, Value> lhs, cuco::pair<Key, Value> rhs) {
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

    thrust::device_vector<cuco::pair<Key, Value>> found_pairs(num_pairs);
    auto output_end = map.retrieve(
      query_key_begin, query_key_begin + num, found_pairs.begin(), stream, key_pair_equals{});
    std::size_t const size = std::distance(found_pairs.begin(), output_end);

    REQUIRE(size == num_pairs);

    // sort before compare
    thrust::sort(
      thrust::device,
      found_pairs.begin(),
      found_pairs.end(),
      [] __device__(const cuco::pair<Key, Value>& lhs, const cuco::pair<Key, Value>& rhs) {
        return lhs.first.a < rhs.first.a;
      });
    REQUIRE(
      cuco::test::equal(pair_begin,
                        pair_begin + num_pairs,
                        found_pairs.begin(),
                        [] __device__(cuco::pair<Key, Value> lhs, cuco::pair<Key, Value> rhs) {
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

    thrust::device_vector<cuco::pair<Key, Value>> found_pairs(num);
    auto output_end = map.retrieve_outer(
      query_key_begin, query_key_begin + num, found_pairs.begin(), stream, key_pair_equals{});
    std::size_t const size_outer = std::distance(found_pairs.begin(), output_end);

    REQUIRE(size_outer == num);
  }

  SECTION("All inserted keys-value pairs should be contained")
  {
    map.insert(pair_begin, pair_begin + num_pairs);

    auto size = map.get_size();
    REQUIRE(size == num_pairs);

    thrust::device_vector<bool> contained(num_pairs);
    map.contains(key_begin, key_begin + num_pairs, contained.begin(), key_pair_equals{}, stream);

    REQUIRE(cuco::test::all_of(contained.begin(), contained.end(), thrust::identity{}));
  }

  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    auto size = map.get_size();
    REQUIRE(size == 0);

    thrust::device_vector<bool> contained(num_pairs);
    map.contains(key_begin, key_begin + num_pairs, contained.begin(), key_pair_equals{}, stream);

    REQUIRE(cuco::test::none_of(contained.begin(), contained.end(), thrust::identity{}));
  }
}

TEMPLATE_TEST_CASE_SIG("User defined key and value type",
                       "",
                       ((cuco::test::probe_sequence Probe), Probe),
                       (cuco::test::probe_sequence::linear_probing),
                       (cuco::test::probe_sequence::double_hashing))
{
  using Key   = key_pair;
  using Value = value_pair;

  auto constexpr sentinel_key   = Key{-1, -1};
  auto constexpr sentinel_value = Value{-1, -1};

  constexpr std::size_t num_pairs = 100;
  constexpr std::size_t capacity  = num_pairs * 2;

  using probe = std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                                   cuco::legacy::linear_probing<1, hash_key_pair>,
                                   cuco::legacy::double_hashing<8, hash_key_pair, hash_key_pair>>;

  cuco::static_multimap<Key, Value, cuda::thread_scope_device, cuco::cuda_allocator<char>, probe>
    map{capacity, cuco::empty_key{sentinel_key}, cuco::empty_value{sentinel_value}};
  test_custom_key_value_type(map, num_pairs);
}
