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

#include <cuco/detail/__config>

#if defined(CUCO_HAS_128BIT_ATOMICS)

#include <test_utils.hpp>

#include <cuco/static_set.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>
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

TEST_CASE("static_set 128-bit key unique sequence", "")
{
  using Key = uint128_t;

  constexpr std::size_t num_keys{400};

  Key const empty_sentinel{~0ULL, ~0ULL};

  using probe = cuco::linear_probing<1, cuco::default_hash_function<Key>>;

  auto set = cuco::static_set{
    num_keys, cuco::empty_key<Key>{empty_sentinel}, {}, probe{}, {}, cuco::storage<2>{}};

  auto keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<std::size_t>(0),
    cuda::proclaim_return_type<Key>(
      [] __device__(std::size_t i) -> Key { return Key{static_cast<cuda::std::uint64_t>(i), 0}; }));

  thrust::device_vector<bool> d_contained(num_keys);

  SECTION("Non-inserted keys should not be contained.")
  {
    REQUIRE(set.size() == 0);

    set.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::none_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  }

  SECTION("All inserted keys should be contained.")
  {
    auto const inserted = set.insert(keys_begin, keys_begin + num_keys);
    REQUIRE(inserted == num_keys);
    REQUIRE(set.size() == num_keys);

    set.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
    REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));
  }

  SECTION("All inserted keys should be correctly recovered during find")
  {
    set.insert(keys_begin, keys_begin + num_keys);

    thrust::device_vector<Key> d_results(num_keys);
    set.find(keys_begin, keys_begin + num_keys, d_results.begin());

    auto zip_equal = cuda::proclaim_return_type<bool>([] __device__(auto const& p) {
      return static_cast<Key>(cuda::std::get<0>(p)) == static_cast<Key>(cuda::std::get<1>(p));
    });
    auto zip       = thrust::make_zip_iterator(cuda::std::tuple{d_results.begin(), keys_begin});

    REQUIRE(cuco::test::all_of(zip, zip + num_keys, zip_equal));
  }
}

TEST_CASE("static_set 128-bit key insert_and_find", "")
{
  using Key = uint128_t;

  constexpr std::size_t num_keys{400};

  Key const empty_sentinel{~0ULL, ~0ULL};

  using probe = cuco::linear_probing<1, cuco::default_hash_function<Key>>;

  auto set = cuco::static_set{
    num_keys, cuco::empty_key<Key>{empty_sentinel}, {}, probe{}, {}, cuco::storage<2>{}};

  auto keys_begin = cuda::make_transform_iterator(
    cuda::counting_iterator<std::size_t>(0),
    cuda::proclaim_return_type<Key>(
      [] __device__(std::size_t i) -> Key { return Key{static_cast<cuda::std::uint64_t>(i), 0}; }));

  thrust::device_vector<Key> iters1(num_keys);
  thrust::device_vector<bool> inserted(num_keys);

  // insert first time, fills inserted with true
  set.insert_and_find(keys_begin, keys_begin + num_keys, iters1.begin(), inserted.begin());
  REQUIRE(cuco::test::all_of(inserted.begin(), inserted.end(), cuda::std::identity{}));

  // insert second time, fills inserted with false as keys already in set
  thrust::device_vector<Key> iters2(num_keys);
  set.insert_and_find(keys_begin, keys_begin + num_keys, iters2.begin(), inserted.begin());
  REQUIRE(cuco::test::none_of(inserted.begin(), inserted.end(), cuda::std::identity{}));

  // both iters1 and iters2 should be same, as keys will be referring to same slot
  auto equal_fn = cuda::proclaim_return_type<bool>(
    [] __device__(auto const& a, auto const& b) { return a == b; });
  REQUIRE(cuco::test::equal(iters1.begin(), iters1.end(), iters2.begin(), equal_fn));
}

#endif  // CUCO_HAS_128BIT_ATOMICS
