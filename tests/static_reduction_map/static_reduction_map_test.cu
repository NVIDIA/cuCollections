/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cuco/static_reduction_map.cuh>
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
  cuco::static_reduction_map<Key, Value> map{100'000'000, -1, -1};

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

  // bulk function test cases
  SECTION("All inserted keys-value pairs should be correctly recovered during find")
  {
    map.insert(d_pairs.begin(), d_pairs.end());
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(d_results.begin(), d_values.begin()));

    REQUIRE(all_of(zip, zip + num_keys, [] __device__(auto const& p) {
      return thrust::get<0>(p) == thrust::get<1>(p);
    }));
  }

  SECTION("All inserted keys-value pairs should be contained")
  {
    map.insert(d_pairs.begin(), d_pairs.end());
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());

    REQUIRE(
      all_of(d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b; }));
  }

  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());

    REQUIRE(
      none_of(d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b; }));
  }

  SECTION("Inserting unique keys should return insert success.")
  {
    if (Dist == dist_type::UNIQUE) {
      REQUIRE(all_of(d_pairs.begin(),
                     d_pairs.end(),
                     [m_view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
                       return m_view.insert(pair);
                     }));
    }
  }

  SECTION("Cannot find any key in an empty hash map with non-const view")
  {
    SECTION("non-const view")
    {
      REQUIRE(all_of(d_pairs.begin(),
                     d_pairs.end(),
                     [view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
                       return view.find(pair.first) == view.end();
                     }));
    }
    SECTION("const view")
    {
      REQUIRE(all_of(
        d_pairs.begin(), d_pairs.end(), [view] __device__(cuco::pair_type<Key, Value> const& pair) {
          return view.find(pair.first) == view.end();
        }));
    }
  }

  SECTION("Keys are all found after inserting many keys.")
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
      // All keys should be found
      REQUIRE(all_of(d_pairs.begin(),
                     d_pairs.end(),
                     [view] __device__(cuco::pair_type<Key, Value> const& pair) mutable {
                       auto const found = view.find(pair.first);
                       return (found != view.end()) and (found->first.load() == pair.first and
                                                         found->second.load() == pair.second);
                     }));
    }
    SECTION("const view")
    {
      // All keys should be found
      REQUIRE(all_of(
        d_pairs.begin(), d_pairs.end(), [view] __device__(cuco::pair_type<Key, Value> const& pair) {
          auto const found = view.find(pair.first);
          return (found != view.end()) and
                 (found->first.load() == pair.first and found->second.load() == pair.second);
        }));
    }
  }
}

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

  auto g = cg::this_thread_block();
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
                       ((typename T, dist_type Dist), T, Dist),
                       (int32_t, dist_type::UNIQUE),
                       (int64_t, dist_type::UNIQUE),
                       (int32_t, dist_type::UNIFORM),
                       (int64_t, dist_type::UNIFORM),
                       (int32_t, dist_type::GAUSSIAN),
                       (int64_t, dist_type::GAUSSIAN))
{
  using KeyType                = T;
  using ValueType              = T;
  using MapType                = cuco::static_reduction_map<KeyType, ValueType>;
  using DeviceViewType         = typename MapType::device_view;
  using DeviceViewIteratorType = typename DeviceViewType::iterator;

  constexpr std::size_t number_of_maps  = 1000;
  constexpr std::size_t elements_in_map = 500;
  constexpr std::size_t map_capacity    = 2 * elements_in_map;

  // one array for all maps, first elements_in_map element belong to map 0, second to map 1 and so
  // on
  std::vector<KeyType> h_keys(number_of_maps * elements_in_map);
  std::vector<ValueType> h_values(number_of_maps * elements_in_map);
  std::vector<cuco::pair_type<KeyType, ValueType>> h_pairs(number_of_maps * elements_in_map);

  // using std::unique_ptr because static_reduction_map does not have copy/move
  // constructor/assignment operator yet
  std::vector<std::unique_ptr<MapType>> maps;

  for (std::size_t map_id = 0; map_id < number_of_maps; ++map_id) {
    const std::size_t offset = map_id * elements_in_map;

    generate_keys<Dist, KeyType>(h_keys.begin() + offset,
                                 h_keys.begin() + offset + elements_in_map);

    for (std::size_t i = 0; i < elements_in_map; ++i) {
      KeyType key                = h_keys[offset + i];
      ValueType val              = key < std::numeric_limits<KeyType>::max() ? key + 1 : 0;
      h_values[offset + i]       = val;
      h_pairs[offset + i].first  = key;
      h_pairs[offset + i].second = val;
    }

    maps.push_back(std::make_unique<MapType>(map_capacity, -1, -1));
  }

  thrust::device_vector<KeyType> d_keys(h_keys);
  thrust::device_vector<ValueType> d_values(h_values);
  thrust::device_vector<cuco::pair_type<KeyType, ValueType>> d_pairs(h_pairs);

  SECTION("Keys are all found after insertion.")
  {
    std::vector<DeviceViewType> h_device_views;
    for (std::size_t map_id = 0; map_id < number_of_maps; ++map_id) {
      const std::size_t offset = map_id * elements_in_map;

      MapType* map = maps[map_id].get();
      map->insert(d_pairs.begin() + offset, d_pairs.begin() + offset + elements_in_map);
      h_device_views.push_back(map->get_device_view());
    }
    thrust::device_vector<DeviceViewType> d_device_views(h_device_views);

    thrust::device_vector<bool> d_keys_exist(number_of_maps * elements_in_map);
    thrust::device_vector<bool> d_keys_and_values_correct(number_of_maps * elements_in_map);

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

    REQUIRE(all_of(zip, zip + d_keys_exist.size(), [] __device__(auto const& z) {
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

    thrust::device_vector<bool> d_keys_exist(number_of_maps * elements_in_map);
    thrust::device_vector<bool> d_keys_and_values_correct(number_of_maps * elements_in_map);

    shared_memory_test_kernel<MapType, map_capacity>
      <<<number_of_maps, 64>>>(d_device_views.data().get(),
                               d_keys.data().get(),
                               d_values.data().get(),
                               elements_in_map,
                               d_keys_exist.data().get(),
                               d_keys_and_values_correct.data().get());

    REQUIRE(none_of(d_keys_exist.begin(), d_keys_exist.end(), [] __device__(const bool key_found) {
      return key_found;
    }));
  }
}