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

#include <random>

#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <nvbench/nvbench.cuh>

#include <cuco/static_multimap.cuh>
#include <key_generator.hpp>

namespace {
// Custom pair equal
template <typename Key, typename Value>
struct pair_equal {
  __device__ bool operator()(const cuco::pair_type<Key, Value>& lhs,
                             const cuco::pair_type<Key, Value>& rhs) const
  {
    return lhs.first == rhs.first;
  }
};
}  // anonymous namespace

/**
 * @brief A benchmark evaluating multi-value `insert` performance:
 * - Total number of insertions: 100'000'000
 * - CG size: 8
 */
template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_static_multimap_insert(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  auto const num_keys  = state.get_int64("NumInputs");
  auto const occupancy = state.get_float64("Occupancy");

  std::size_t const size = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Dist, Multiplicity, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  state.add_element_count(num_keys, "NumKeys");

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               cuco::static_multimap<Key, Value> map{size, -1, -1};

               // Use timers to explicitly mark the target region
               timer.start();
               map.insert(d_pairs.begin(), d_pairs.end(), launch.get_stream());
               timer.stop();
             });
}

template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_static_multimap_insert(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  state.skip("Key should be the same type as Value.");
}

/**
 * @brief A benchmark evaluating multi-value `count` performance:
 * - Total number of insertions: 100'000'000
 * - CG size: 8
 */
template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_static_multimap_count(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  auto const num_keys      = state.get_int64("NumInputs");
  auto const occupancy     = state.get_float64("Occupancy");
  auto const matching_rate = state.get_float64("MatchingRate");

  std::size_t const size = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Dist, Multiplicity, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  generate_probe_keys<Key>(matching_rate, h_keys.begin(), h_keys.end());

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  state.add_element_count(num_keys, "NumKeys");

  cuco::static_multimap<Key, Value> map{size, -1, -1};
  map.insert(d_pairs.begin(), d_pairs.end());

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto count = map.count(d_keys.begin(), d_keys.end(), launch.get_stream());
  });
}

template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_static_multimap_count(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  state.skip("Key should be the same type as Value.");
}

/**
 * @brief A benchmark evaluating multi-value `retrieve` performance:
 * - Total number of insertions: 100'000'000
 * - CG size: 8
 */
template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_static_multimap_retrieve(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  auto const num_keys      = state.get_int64("NumInputs");
  auto const occupancy     = state.get_float64("Occupancy");
  auto const matching_rate = state.get_float64("MatchingRate");

  std::size_t const size = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Dist, Multiplicity, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  generate_probe_keys<Key>(matching_rate, h_keys.begin(), h_keys.end());

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  state.add_element_count(num_keys, "NumKeys");

  cuco::static_multimap<Key, Value> map{size, -1, -1};
  map.insert(d_pairs.begin(), d_pairs.end());

  auto const output_size = map.count_outer(d_keys.begin(), d_keys.end());
  thrust::device_vector<cuco::pair_type<Key, Value>> d_results(output_size);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    map.retrieve_outer(d_keys.begin(), d_keys.end(), d_results.data().get(), launch.get_stream());
  });
}

template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_static_multimap_retrieve(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  state.skip("Key should be the same type as Value.");
}

/**
 * @brief A benchmark evaluating multi-value query (`count` + `retrieve`) performance:
 * - Total number of insertions: 100'000'000
 * - CG size: 8
 */
template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_static_multimap_query(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  auto const num_keys      = state.get_int64("NumInputs");
  auto const occupancy     = state.get_float64("Occupancy");
  auto const matching_rate = state.get_float64("MatchingRate");

  std::size_t const size = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Dist, Multiplicity, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  generate_probe_keys<Key>(matching_rate, h_keys.begin(), h_keys.end());

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  state.add_element_count(num_keys, "NumKeys");

  cuco::static_multimap<Key, Value> map{size, -1, -1};
  map.insert(d_pairs.begin(), d_pairs.end());

  auto const output_size = map.count_outer(d_keys.begin(), d_keys.end());
  thrust::device_vector<cuco::pair_type<Key, Value>> d_results(output_size);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto count = map.count_outer(d_keys.begin(), d_keys.end(), launch.get_stream());
    map.retrieve_outer(d_keys.begin(), d_keys.end(), d_results.data().get(), launch.get_stream());
  });
}

template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_static_multimap_query(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  state.skip("Key should be the same type as Value.");
}

/**
 * @brief A benchmark evaluating `pair_retrieve` performance:
 * - CG size: 8
 */
template <typename Key, typename Value, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_static_multimap_pair_retrieve(
  nvbench::state& state, nvbench::type_list<Key, Value, nvbench::enum_type<Multiplicity>>)
{
  auto constexpr matching_rate = 0.5;
  auto constexpr occupancy     = 0.5;
  auto constexpr dist          = dist_type::UNIFORM;

  auto const num_input = state.get_int64("NumInputs");

  std::size_t const size = num_input / occupancy;

  std::vector<Key> h_keys(num_input);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_input);

  generate_keys<dist, Multiplicity, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_input; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);
  auto const pair_begin = d_pairs.begin();

  cuco::static_multimap<Key, Value> map{size, -1, -1};
  map.insert(pair_begin, pair_begin + num_input);

  generate_probe_keys<Key>(matching_rate, h_keys.begin(), h_keys.end());
  thrust::device_vector<Key> d_keys(h_keys);

  thrust::transform(
    thrust::device, d_keys.begin(), d_keys.begin() + num_input, pair_begin, [] __device__(Key i) {
      return cuco::pair_type<Key, Value>{i, i};
    });

  state.add_element_count(num_input, "NumInputs");

  auto const output_size =
    map.pair_count(pair_begin, pair_begin + num_input, pair_equal<Key, Value>{});
  thrust::device_vector<cuco::pair_type<Key, Value>> d_results(output_size);

  auto out1_begin = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
  auto out2_begin = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto [out1_end, out2_end] = map.pair_retrieve(
      pair_begin, pair_begin + num_input, out1_begin, out2_begin, pair_equal<Key, Value>{});
  });
}

template <typename Key, typename Value, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_static_multimap_pair_retrieve(
  nvbench::state& state, nvbench::type_list<Key, Value, nvbench::enum_type<Multiplicity>>)
{
  state.skip("Key should be the same type as Value.");
}

using key_type   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using value_type = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using d_type =
  nvbench::enum_type_list<dist_type::GAUSSIAN, dist_type::GEOMETRIC, dist_type::UNIFORM>;

using multiplicity = nvbench::enum_type_list<1, 2, 4, 8, 16, 32, 64, 128, 256>;

NVBENCH_BENCH_TYPES(nvbench_static_multimap_insert,
                    NVBENCH_TYPE_AXES(key_type,
                                      value_type,
                                      nvbench::enum_type_list<dist_type::UNIFORM>,
                                      multiplicity))
  .set_name("staic_multimap_insert_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_insert,
                    NVBENCH_TYPE_AXES(key_type, value_type, d_type, nvbench::enum_type_list<8>))
  .set_name("staic_multimap_insert_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1));

NVBENCH_BENCH_TYPES(nvbench_static_multimap_count,
                    NVBENCH_TYPE_AXES(key_type,
                                      value_type,
                                      nvbench::enum_type_list<dist_type::UNIFORM>,
                                      multiplicity))
  .set_name("staic_multimap_count_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("MatchingRate", {0.5});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_count,
                    NVBENCH_TYPE_AXES(key_type, value_type, d_type, nvbench::enum_type_list<8>))
  .set_name("staic_multimap_count_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1))
  .add_float64_axis("MatchingRate", {0.5});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_count,
                    NVBENCH_TYPE_AXES(key_type, value_type, d_type, nvbench::enum_type_list<8>))
  .set_name("staic_multimap_count_matching_rate")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("MatchingRate", {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_retrieve,
                    NVBENCH_TYPE_AXES(key_type,
                                      value_type,
                                      nvbench::enum_type_list<dist_type::UNIFORM>,
                                      multiplicity))
  .set_name("staic_multimap_retrieve_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("MatchingRate", {0.5});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_retrieve,
                    NVBENCH_TYPE_AXES(key_type, value_type, d_type, nvbench::enum_type_list<8>))
  .set_name("staic_multimap_retrieve_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1))
  .add_float64_axis("MatchingRate", {0.5});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_retrieve,
                    NVBENCH_TYPE_AXES(key_type, value_type, d_type, nvbench::enum_type_list<8>))
  .set_name("staic_multimap_retrieve_matching_rate")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("MatchingRate", {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_query,
                    NVBENCH_TYPE_AXES(key_type,
                                      value_type,
                                      nvbench::enum_type_list<dist_type::UNIFORM>,
                                      multiplicity))
  .set_name("staic_multimap_query_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("MatchingRate", {0.5});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_query,
                    NVBENCH_TYPE_AXES(key_type, value_type, d_type, nvbench::enum_type_list<8>))
  .set_name("staic_multimap_query_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1))
  .add_float64_axis("MatchingRate", {0.5});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_query,
                    NVBENCH_TYPE_AXES(key_type, value_type, d_type, nvbench::enum_type_list<8>))
  .set_name("staic_multimap_query_matching_rate")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("MatchingRate", {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_pair_retrieve,
                    NVBENCH_TYPE_AXES(key_type, value_type, multiplicity))
  .set_name("staic_multimap_pair_retrieve_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value", "Multiplicity"})
  .set_timeout(100)  // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)  // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs",
                  {1'000,
                   100'000,
                   1'000'000,
                   10'000'000,
                   100'000'000});  // Total number of key/value pairs: 100'000'000
