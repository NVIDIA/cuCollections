/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>

#include <catch2/catch.hpp>

#include <limits>

template <typename MapType, int CAPACITY>
__global__ void shared_memory_test_kernel(
  typename MapType::device_view const* const device_views,
  typename MapType::device_view::key_type const* const insterted_keys,
  typename MapType::device_view::mapped_type const* const inserted_values,
  const size_t number_of_elements,
  bool* const keys_exist,
  bool* const keys_and_values_correct)
{
  // Each block processes one map
  const size_t map_id = blockIdx.x;
  const size_t offset = map_id * number_of_elements;

  __shared__ typename MapType::pair_atomic_type sm_buffer[CAPACITY];

  auto g = cuco::test::cg::this_thread_block();
  typename MapType::device_view sm_device_view =
    MapType::device_view::make_copy(g, sm_buffer, device_views[map_id]);

  for (int i = g.thread_rank(); i < number_of_elements; i += g.size()) {
    auto found_pair_it = sm_device_view.find(insterted_keys[offset + i]);

    if (found_pair_it != sm_device_view.end()) {
      keys_exist[offset + i] = true;
      if (found_pair_it->first == insterted_keys[offset + i] and
          found_pair_it->second == inserted_values[offset + i]) {
        keys_and_values_correct[offset + i] = true;
      } else {
        keys_and_values_correct[offset + i] = false;
      }
    } else {
      keys_exist[offset + i]              = false;
      keys_and_values_correct[offset + i] = true;
    }
  }
}

TEMPLATE_TEST_CASE_SIG("Shared memory static map",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t),
                       (int32_t, int64_t),
                       (int64_t, int32_t),
                       (int64_t, int64_t))
{
  using MapType                = cuco::static_map<Key, Value>;
  using DeviceViewType         = typename MapType::device_view;
  using DeviceViewIteratorType = typename DeviceViewType::iterator;

  constexpr std::size_t number_of_maps  = 1000;
  constexpr std::size_t elements_in_map = 500;
  constexpr std::size_t map_capacity    = 2 * elements_in_map;

  // one array for all maps, first elements_in_map element belong to map 0, second to map 1 and so
  // on
  thrust::device_vector<Key> d_keys(number_of_maps * elements_in_map);
  thrust::device_vector<Value> d_values(number_of_maps * elements_in_map);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  thrust::sequence(thrust::device, d_values.begin(), d_values.end(), 1);

  // using std::unique_ptr because static_map does not have copy/move constructor/assignment
  // operator yet
  std::vector<std::unique_ptr<MapType>> maps;
  for (std::size_t map_id = 0; map_id < number_of_maps; ++map_id) {
    maps.push_back(std::make_unique<MapType>(
      map_capacity, cuco::sentinel::empty_key<Key>{-1}, cuco::sentinel::empty_value<Value>{-1}));
  }

  thrust::device_vector<bool> d_keys_exist(number_of_maps * elements_in_map);
  thrust::device_vector<bool> d_keys_and_values_correct(number_of_maps * elements_in_map);

  SECTION("Keys are all found after insertion.")
  {
    auto pairs_begin =
      thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_values.begin()));
    std::vector<DeviceViewType> h_device_views;
    for (std::size_t map_id = 0; map_id < number_of_maps; ++map_id) {
      const std::size_t offset = map_id * elements_in_map;

      MapType* map = maps[map_id].get();
      map->insert(pairs_begin + offset, pairs_begin + offset + elements_in_map);
      h_device_views.push_back(map->get_device_view());
    }
    thrust::device_vector<DeviceViewType> d_device_views(h_device_views);

    shared_memory_test_kernel<MapType, map_capacity>
      <<<number_of_maps, 64>>>(d_device_views.data().get(),
                               d_keys.data().get(),
                               d_values.data().get(),
                               elements_in_map,
                               d_keys_exist.data().get(),
                               d_keys_and_values_correct.data().get());

    REQUIRE(d_keys_exist.size() == d_keys_and_values_correct.size());
    auto zip = thrust::make_zip_iterator(
      thrust::make_tuple(d_keys_exist.begin(), d_keys_and_values_correct.begin()));

    REQUIRE(cuco::test::all_of(zip, zip + d_keys_exist.size(), [] __device__(auto const& z) {
      return thrust::get<0>(z) and thrust::get<1>(z);
    }));
  }

  SECTION("No key is found before insertion.")
  {
    std::vector<DeviceViewType> h_device_views;
    for (std::size_t map_id = 0; map_id < number_of_maps; ++map_id) {
      h_device_views.push_back(maps[map_id].get()->get_device_view());
    }
    thrust::device_vector<DeviceViewType> d_device_views(h_device_views);

    shared_memory_test_kernel<MapType, map_capacity>
      <<<number_of_maps, 64>>>(d_device_views.data().get(),
                               d_keys.data().get(),
                               d_values.data().get(),
                               elements_in_map,
                               d_keys_exist.data().get(),
                               d_keys_and_values_correct.data().get());

    REQUIRE(cuco::test::none_of(d_keys_exist.begin(),
                                d_keys_exist.end(),
                                [] __device__(const bool key_found) { return key_found; }));
  }
}

template <typename K, typename V, std::size_t N>
__global__ void shared_memory_hash_table_kernel(bool* key_found)
{
  namespace cg   = cooperative_groups;
  using map_type = typename cuco::static_map<K, V, cuda::thread_scope_block>::device_mutable_view;
  using find_map_type = typename cuco::static_map<K, V, cuda::thread_scope_block>::device_view;
  __shared__ typename map_type::slot_type slots[N];
  auto map = map_type::make_from_uninitialized_slots(cg::this_thread_block(),
                                                     &slots[0],
                                                     N,
                                                     cuco::sentinel::empty_key<K>{-1},
                                                     cuco::sentinel::empty_value<V>{-1});

  auto g            = cg::this_thread_block();
  std::size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  int rank          = g.thread_rank();

  // insert {thread_rank, thread_rank} for each thread in thread-block
  map.insert(cuco::pair<int, int>(rank, rank));
  g.sync();

  auto find_map       = find_map_type(map);
  auto retrieved_pair = find_map.find(rank);
  if (retrieved_pair != find_map.end() && retrieved_pair->second == rank) {
    key_found[index] = true;
  }
}

TEMPLATE_TEST_CASE("Shared memory slots.", "", int32_t)
{
  constexpr std::size_t N = 256;
  thrust::device_vector<bool> key_found(N, false);
  shared_memory_hash_table_kernel<TestType, TestType, N><<<8, 32>>>(key_found.data().get());

  REQUIRE(cuco::test::all_of(key_found.begin(), key_found.end(), thrust::identity<bool>{}));
}
