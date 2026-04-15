/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <test_utils.hpp>

#include <cuco/detail/__config>
#include <cuco/static_map.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/device_vector.h>

#include <catch2/catch_template_test_macros.hpp>

struct alignas(16) uint128_t {
  cuda::std::uint64_t lo;
  cuda::std::uint64_t hi;

  __host__ __device__ constexpr bool operator==(uint128_t const& o) const
  {
    return lo == o.lo and hi == o.hi;
  }
  __host__ __device__ constexpr bool operator!=(uint128_t const& o) const { return !(*this == o); }
};

CUCO_DECLARE_BITWISE_COMPARABLE(uint128_t)

using size_type = int32_t;

TEST_CASE("static_map 128-bit packed CAS", "")
{
  using Key   = int64_t;
  using Value = int64_t;
  using probe = cuco::linear_probing<1, cuco::default_hash_function<Key>>;

  constexpr size_type num_keys{400};

  auto map = cuco::static_map<Key,
                              Value,
                              cuco::extent<size_type>,
                              cuda::thread_scope_device,
                              cuda::std::equal_to<Key>,
                              probe,
                              cuco::cuda_allocator<cuda::std::byte>,
                              cuco::storage<2>>{
    num_keys, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};

  auto keys_begin  = cuda::counting_iterator<Key>(1);
  auto pairs_begin = cuda::make_transform_iterator(
    keys_begin, cuda::proclaim_return_type<cuco::pair<Key, Value>>([] __device__(Key const& x) {
      return cuco::pair<Key, Value>(x, static_cast<Value>(x));
    }));

  thrust::device_vector<bool> d_contained(num_keys);

  SECTION("insert + contains")
  {
    auto const inserted = map.insert(pairs_begin, pairs_begin + num_keys);
    REQUIRE(inserted == num_keys);
    REQUIRE(map.size() == num_keys);

    map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  }

  SECTION("insert + find")
  {
    map.insert(pairs_begin, pairs_begin + num_keys);

    thrust::device_vector<Value> d_results(num_keys);
    map.find(keys_begin, keys_begin + num_keys, d_results.begin());

    auto zip_equal = cuda::proclaim_return_type<bool>(
      [] __device__(auto const& p) { return cuda::std::get<0>(p) == cuda::std::get<1>(p); });
    auto zip = thrust::make_zip_iterator(
      cuda::std::tuple{d_results.begin(),
                       cuda::make_transform_iterator(
                         keys_begin, cuda::proclaim_return_type<Value>([] __device__(Key const& x) {
                           return static_cast<Value>(x);
                         }))});
    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));
  }

  SECTION("insert + erase + re-insert")
  {
    auto erase_map = cuco::static_map<Key,
                                      Value,
                                      cuco::extent<size_type>,
                                      cuda::thread_scope_device,
                                      cuda::std::equal_to<Key>,
                                      probe,
                                      cuco::cuda_allocator<cuda::std::byte>,
                                      cuco::storage<2>>{num_keys * 2,
                                                        cuco::empty_key<Key>{-1},
                                                        cuco::empty_value<Value>{-1},
                                                        cuco::erased_key<Key>{-2}};

    erase_map.insert(pairs_begin, pairs_begin + num_keys);
    REQUIRE(erase_map.size() == num_keys);

    erase_map.erase(keys_begin, keys_begin + num_keys);
    REQUIRE(erase_map.size() == 0);

    erase_map.insert(pairs_begin, pairs_begin + num_keys);
    REQUIRE(erase_map.size() == num_keys);

    erase_map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  }

  SECTION("insert_and_find")
  {
    thrust::device_vector<size_type> found1(num_keys);
    thrust::device_vector<bool> inserted(num_keys);

    map.insert_and_find(pairs_begin, pairs_begin + num_keys, found1.begin(), inserted.begin());
    REQUIRE(cuco::test::all_of(inserted.begin(), inserted.end(), cuda::std::identity{}));

    thrust::device_vector<size_type> found2(num_keys);
    map.insert_and_find(pairs_begin, pairs_begin + num_keys, found2.begin(), inserted.begin());
    REQUIRE(cuco::test::none_of(inserted.begin(), inserted.end(), cuda::std::identity{}));

    REQUIRE(
      cuco::test::equal(found1.begin(), found1.end(), found2.begin(), cuda::std::equal_to<Key>{}));
  }
}

#if defined(CUCO_HAS_128BIT_ATOMICS)

TEST_CASE("static_map 128-bit key b2b CAS", "")
{
  using Key   = uint128_t;
  using Value = int64_t;
  using probe = cuco::linear_probing<1, cuco::default_hash_function<Key>>;

  constexpr size_type num_keys{400};

  Key const empty_key{~0ULL, ~0ULL};
  Key const erased_key{~0ULL - 1, ~0ULL};

  auto map = cuco::static_map<Key,
                              Value,
                              cuco::extent<size_type>,
                              cuda::thread_scope_device,
                              cuda::std::equal_to<Key>,
                              probe,
                              cuco::cuda_allocator<cuda::std::byte>,
                              cuco::storage<2>>{num_keys * 2,
                                                cuco::empty_key<Key>{empty_key},
                                                cuco::empty_value<Value>{-1},
                                                cuco::erased_key<Key>{erased_key}};

  auto keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>(1),
    cuda::proclaim_return_type<Key>(
      [] __device__(size_type i) -> Key { return Key{static_cast<cuda::std::uint64_t>(i), 0}; }));

  auto pairs_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>(1),
    cuda::proclaim_return_type<cuco::pair<Key, Value>>([] __device__(size_type i) {
      return cuco::pair<Key, Value>{Key{static_cast<cuda::std::uint64_t>(i), 0},
                                    static_cast<Value>(i)};
    }));

  thrust::device_vector<bool> d_contained(num_keys);

  SECTION("insert + contains")
  {
    auto const inserted = map.insert(pairs_begin, pairs_begin + num_keys);
    REQUIRE(inserted == num_keys);

    map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  }

  SECTION("insert + find")
  {
    map.insert(pairs_begin, pairs_begin + num_keys);

    thrust::device_vector<Value> d_results(num_keys);
    map.find(keys_begin, keys_begin + num_keys, d_results.begin());

    auto zip_equal = cuda::proclaim_return_type<bool>(
      [] __device__(auto const& p) { return cuda::std::get<0>(p) == cuda::std::get<1>(p); });
    auto gold =
      cuda::make_transform_iterator(cuda::counting_iterator<size_type>(1),
                                    cuda::proclaim_return_type<Value>([] __device__(size_type i) {
                                      return static_cast<Value>(i);
                                    }));
    auto zip = thrust::make_zip_iterator(cuda::std::tuple{d_results.begin(), gold});
    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));
  }

  SECTION("insert + erase + re-insert")
  {
    map.insert(pairs_begin, pairs_begin + num_keys);
    REQUIRE(map.size() == num_keys);

    map.erase(keys_begin, keys_begin + num_keys);
    REQUIRE(map.size() == 0);

    map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::none_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));

    map.insert(pairs_begin, pairs_begin + num_keys);
    REQUIRE(map.size() == num_keys);

    map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  }
}

TEST_CASE("static_map 128-bit key and value", "")
{
  using Key   = uint128_t;
  using Value = uint128_t;
  using probe = cuco::linear_probing<1, cuco::default_hash_function<Key>>;

  constexpr size_type num_keys{400};

  Key const empty_key{~0ULL, ~0ULL};
  Value const empty_value{~0ULL, ~0ULL};

  auto map = cuco::static_map<Key,
                              Value,
                              cuco::extent<size_type>,
                              cuda::thread_scope_device,
                              cuda::std::equal_to<Key>,
                              probe,
                              cuco::cuda_allocator<cuda::std::byte>,
                              cuco::storage<2>>{
    num_keys, cuco::empty_key<Key>{empty_key}, cuco::empty_value<Value>{empty_value}};

  auto keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>(1),
    cuda::proclaim_return_type<Key>(
      [] __device__(size_type i) -> Key { return Key{static_cast<cuda::std::uint64_t>(i), 0}; }));

  auto pairs_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<size_type>(1),
    cuda::proclaim_return_type<cuco::pair<Key, Value>>([] __device__(size_type i) {
      return cuco::pair<Key, Value>{Key{static_cast<cuda::std::uint64_t>(i), 0},
                                    Value{static_cast<cuda::std::uint64_t>(i * 10), 0}};
    }));

  thrust::device_vector<bool> d_contained(num_keys);

  SECTION("insert + contains")
  {
    auto const inserted = map.insert(pairs_begin, pairs_begin + num_keys);
    REQUIRE(inserted == num_keys);
    REQUIRE(map.size() == num_keys);

    map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  }

  SECTION("insert + find")
  {
    map.insert(pairs_begin, pairs_begin + num_keys);

    thrust::device_vector<Value> d_results(num_keys);
    map.find(keys_begin, keys_begin + num_keys, d_results.begin());

    auto zip_equal = cuda::proclaim_return_type<bool>([] __device__(auto const& p) {
      return static_cast<Value>(cuda::std::get<0>(p)) == static_cast<Value>(cuda::std::get<1>(p));
    });
    auto gold      = cuda::make_transform_iterator(
      cuda::counting_iterator<size_type>(1),
      cuda::proclaim_return_type<Value>([] __device__(size_type i) -> Value {
        return Value{static_cast<cuda::std::uint64_t>(i * 10), 0};
      }));
    auto zip = thrust::make_zip_iterator(cuda::std::tuple{d_results.begin(), gold});
    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));
  }
}

#endif  // CUCO_HAS_128BIT_ATOMICS
