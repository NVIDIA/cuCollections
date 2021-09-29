/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>

#include <catch2/catch.hpp>
#include <cuco/static_map.cuh>

#include <util.hpp>

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

// User-defined key type
// Need to specify alignment to WAR libcu++ bug where cuda::atomic fails for underaligned types:
// https://github.com/NVIDIA/libcudacxx/issues/160
struct alignas(8) key_pair {
  int32_t a;
  int32_t b;
};

struct hash_key_pair {
  __device__ uint32_t operator()(key_pair k) { return k.a; };
};

struct key_pair_equals {
  __device__ bool operator()(key_pair lhs, key_pair rhs)
  {
    return std::tie(lhs.a, lhs.b) == std::tie(rhs.a, rhs.b);
  }
};

struct alignas(8) value_pair {
  int32_t f;
  int32_t s;
};

#define SIZE 10
__device__ int A[SIZE];

template <typename T>
struct custom_equals {
  __device__ bool operator()(T lhs, T rhs) { return A[lhs] == A[rhs]; }
};

TEST_CASE("User defined key and value type", "")
{
  using Key   = key_pair;
  using Value = value_pair;

  auto constexpr sentinel_key   = Key{-1, -1};
  auto constexpr sentinel_value = Value{-1, -1};

  constexpr std::size_t num_pairs = 100;
  constexpr std::size_t capacity  = num_pairs * 2;
  cuco::static_map<key_pair, value_pair> map{capacity, sentinel_key, sentinel_value};

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

  SECTION("All inserted keys-value pairs should be correctly recovered during find")
  {
    thrust::device_vector<Value> found_values(num_pairs);
    map.insert(insert_pairs, insert_pairs + num_pairs, hash_key_pair{}, key_pair_equals{});

    REQUIRE(num_pairs == map.get_size());

    map.find(insert_keys.begin(),
             insert_keys.end(),
             found_values.begin(),
             hash_key_pair{},
             key_pair_equals{});

    REQUIRE(thrust::equal(thrust::device,
                          insert_values.begin(),
                          insert_values.end(),
                          found_values.begin(),
                          [] __device__(value_pair lhs, value_pair rhs) {
                            return std::tie(lhs.f, lhs.s) == std::tie(rhs.f, rhs.s);
                          }));
  }

  SECTION("All inserted keys-value pairs should be contained")
  {
    thrust::device_vector<bool> contained(num_pairs);
    map.insert(insert_pairs, insert_pairs + num_pairs, hash_key_pair{}, key_pair_equals{});
    map.contains(insert_keys.begin(),
                 insert_keys.end(),
                 contained.begin(),
                 hash_key_pair{},
                 key_pair_equals{});
    REQUIRE(all_of(contained.begin(), contained.end(), [] __device__(bool const& b) { return b; }));
  }

  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    thrust::device_vector<bool> contained(num_pairs);
    map.contains(insert_keys.begin(),
                 insert_keys.end(),
                 contained.begin(),
                 hash_key_pair{},
                 key_pair_equals{});
    REQUIRE(
      none_of(contained.begin(), contained.end(), [] __device__(bool const& b) { return b; }));
  }

  SECTION("Inserting unique keys should return insert success.")
  {
    auto m_view = map.get_device_mutable_view();
    REQUIRE(all_of(insert_pairs,
                   insert_pairs + num_pairs,
                   [m_view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
                     return m_view.insert(pair, hash_key_pair{}, key_pair_equals{});
                   }));
  }

  SECTION("Cannot find any key in an empty hash map")
  {
    SECTION("non-const view")
    {
      auto view = map.get_device_view();
      REQUIRE(all_of(insert_pairs,
                     insert_pairs + num_pairs,
                     [view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
                       return view.find(pair.first, hash_key_pair{}, key_pair_equals{}) ==
                              view.end();
                     }));
    }

    SECTION("const view")
    {
      auto const view = map.get_device_view();
      REQUIRE(all_of(insert_pairs,
                     insert_pairs + num_pairs,
                     [view] __device__(cuco::pair_type<Key, Value> const& pair) {
                       return view.find(pair.first, hash_key_pair{}, key_pair_equals{}) ==
                              view.end();
                     }));
    }
  }
}

TEMPLATE_TEST_CASE_SIG("Key comparison against sentinel",
                       "",
                       ((typename T, dist_type Dist), T, Dist),
                       (int32_t, dist_type::UNIQUE),
                       (int64_t, dist_type::UNIQUE))
{
  using Key   = T;
  using Value = T;

  constexpr std::size_t num_keys{SIZE};
  cuco::static_map<Key, Value> map{SIZE * 2, -1, -1};

  auto m_view = map.get_device_mutable_view();
  auto view   = map.get_device_view();

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  int h_A[SIZE];
  for (int i = 0; i < SIZE; i++) {
    h_A[i] = i;
  }
  cudaMemcpyToSymbol(A, h_A, SIZE * sizeof(int));

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  SECTION(
    "Tests of non-CG insert: The custom `key_equal` can never be used to compare against sentinel")
  {
    REQUIRE(all_of(d_pairs.begin(),
                   d_pairs.end(),
                   [m_view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
                     return m_view.insert(
                       pair, cuco::detail::MurmurHash3_32<Key>{}, custom_equals<Key>{});
                   }));
  }

  SECTION(
    "Tests of CG insert: The custom `key_equal` can never be used to compare against sentinel")
  {
    map.insert(
      d_pairs.begin(), d_pairs.end(), cuco::detail::MurmurHash3_32<Key>{}, custom_equals<Key>{});
    // All keys inserted via custom `key_equal` should be found
    REQUIRE(all_of(
      d_pairs.begin(), d_pairs.end(), [view] __device__(cuco::pair_type<Key, Value> const& pair) {
        auto const found = view.find(pair.first);
        return (found != view.end()) and
               (found->first.load() == pair.first and found->second.load() == pair.second);
      }));
  }
}
