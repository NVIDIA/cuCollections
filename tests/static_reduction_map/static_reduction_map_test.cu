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

#include <algorithm>
#include <catch2/catch.hpp>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>

#include <cuco/static_reduction_map.cuh>

#include <utils.hpp>

// cuco::custom op functor that should give the same result as cuco::reduce_add
template <typename T>
using custom_reduce_add = cuco::custom_op<T, 0, thrust::plus<T>, 0>;

TEMPLATE_TEST_CASE_SIG("Insert all identical keys",
                       "",
                       ((typename Key, typename Value, typename Op), Key, Value, Op),
                       (int32_t, int32_t, cuco::reduce_add<int32_t>),
                       (int32_t, int32_t, custom_reduce_add<int32_t>),
                       (int32_t, float, cuco::reduce_add<float>),
                       (int64_t, double, cuco::reduce_add<double>))
{
  thrust::device_vector<Key> keys(100, 42);
  thrust::device_vector<Value> values(keys.size(), 1);

  auto const num_slots{keys.size() * 2};
  cuco::static_reduction_map<Op, Key, Value> map{num_slots, -1};

  auto zip     = thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), values.begin()));
  auto zip_end = zip + keys.size();
  map.insert(zip, zip_end);

  SECTION("There should only be one key in the map") { REQUIRE(map.get_size() == 1); }

  SECTION("Map should contain the inserted key")
  {
    thrust::device_vector<bool> contained(keys.size());
    map.contains(keys.begin(), keys.end(), contained.begin());
    REQUIRE(
      cuco::test::all_of(contained.begin(), contained.end(), [] __device__(bool c) { return c; }));
  }

  SECTION("Found value should equal aggregate of inserted values")
  {
    thrust::device_vector<Value> found(keys.size());
    map.find(keys.begin(), keys.end(), found.begin());
    auto const expected_aggregate = keys.size();  // All keys inserted "1", so the
                                                  // sum aggregate should be
                                                  // equal to the number of keys inserted
    REQUIRE(
      cuco::test::all_of(found.begin(), found.end(), [expected_aggregate] __device__(Value v) {
        return v == expected_aggregate;
      }));
  }
}

TEMPLATE_TEST_CASE_SIG("Insert all unique keys",
                       "",
                       ((typename Key, typename Value, typename Op), Key, Value, Op),
                       (int32_t, int32_t, cuco::reduce_add<int32_t>),
                       (int32_t, int32_t, custom_reduce_add<int32_t>))
{
  constexpr std::size_t num_keys = 10000;
  constexpr std::size_t num_slots{num_keys * 2};
  cuco::static_reduction_map<Op, Key, Value> map{num_slots, -1};

  auto keys_begin   = thrust::make_counting_iterator<Key>(0);
  auto values_begin = thrust::make_counting_iterator<Value>(0);
  auto zip          = thrust::make_zip_iterator(thrust::make_tuple(keys_begin, values_begin));
  auto zip_end      = zip + num_keys;
  map.insert(zip, zip_end);

  SECTION("Size of map should equal number of inserted keys")
  {
    REQUIRE(map.get_size() == num_keys);
  }

  SECTION("Map should contain the inserted keys")
  {
    thrust::device_vector<bool> contained(num_keys);
    map.contains(keys_begin, keys_begin + num_keys, contained.begin());
    REQUIRE(
      cuco::test::all_of(contained.begin(), contained.end(), [] __device__(bool c) { return c; }));
  }

  SECTION("Found value should equal inserted value")
  {
    thrust::device_vector<Value> found(num_keys);
    map.find(keys_begin, keys_begin + num_keys, found.begin());
    REQUIRE(thrust::equal(thrust::device, values_begin, values_begin + num_keys, found.begin()));
  }
}

template <typename MapType, std::size_t N>
__global__ void static_reduction_map_shared_memory_kernel(bool* key_found)
{
  using Key   = typename MapType::key_type;
  using Value = typename MapType::mapped_type;

  namespace cg            = cooperative_groups;
  using mutable_view_type = typename MapType::device_mutable_view<N>;
  using view_type         = typename MapType::device_view<N>;
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ typename mutable_view_type::slot_type slots[N];
  auto map = mutable_view_type::make_from_uninitialized_slots(cg::this_thread_block(), slots, -1);

  auto g            = cg::this_thread_block();
  std::size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  int rank          = g.thread_rank();

  // insert {thread_rank, thread_rank} for each thread in thread-block
  map.insert(cuco::pair<Key, Value>(rank, rank));
  g.sync();

  auto find_map       = view_type(map);
  auto retrieved_pair = find_map.find(rank);
  if (retrieved_pair != find_map.end() && retrieved_pair->second == rank) {
    key_found[index] = true;
  }
}

TEMPLATE_TEST_CASE_SIG("Reduction map in shared memory",
                       "",
                       ((typename Key, typename Value, typename Op), Key, Value, Op),
                       (int32_t, int32_t, cuco::reduce_add<int32_t>),
                       (int32_t, int32_t, custom_reduce_add<int32_t>),
                       (int32_t, float, cuco::reduce_add<float>),
                       (int64_t, double, cuco::reduce_add<double>))
{
  constexpr std::size_t N = 256;
  thrust::device_vector<bool> key_found(N, false);

  static_reduction_map_shared_memory_kernel<
    cuco::static_reduction_map<Op, Key, Value, cuda::thread_scope_block>,
    N><<<8, 32>>>(key_found.data().get());

  REQUIRE(cuco::test::all_of(key_found.begin(), key_found.end(), thrust::identity<bool>{}));
}
