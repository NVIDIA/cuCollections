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

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <random>

#include "cuco/static_multimap.cuh"

/**
 * @brief Generates input keys with a specific distribution.
 *
 */
template <typename Key, typename OutputIt>
static void generate_keys(OutputIt output_begin,
                          OutputIt output_end,
                          size_t const num_reps,
                          const std::string& dist)
{
  auto num_keys = std::distance(output_begin, output_end);

  if (dist == "RANDOM") {
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<Key> distribution{1, static_cast<Key>(num_keys / num_reps)};
    for (auto i = 0; i < num_keys; ++i) {
      output_begin[i] = distribution(gen);
    }
  } else if (dist == "CYCLE") {
    for (auto i = 0; i < num_keys; ++i) {
      output_begin[i] = (i % (num_keys / num_reps)) + 1;
    }
  }
}

/**
 * @brief Helper function to launch insertion benchmark.
 *
 */
template <typename Key, typename Value, size_t cg_size>
void launch_nvbench_insert(nvbench::state& state,
                           std::vector<Key> const& h_keys,
                           std::size_t const num_keys,
                           std::size_t const size)
{
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = i;
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  state.add_element_count(num_keys, "NumKeys");
  state.add_global_memory_writes<Key>(num_keys * 2);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               cuco::static_multimap<Key, Value, cg_size> map{size, -1, -1};
               auto m_view = map.get_device_mutable_view();

               timer.start();
               map.insert(d_pairs.begin(), d_pairs.end(), launch.get_stream());
               timer.stop();
             });
}

/**
 * @brief A benchmark evaluating multi-value insertion performance by varing number of repetitions
 * per key:
 * - Total number of insertions: 100'000'000
 * - Map occupancy: 0.8
 * - CG size: 8
 * - Number of repetitions per key: 1, ... , 128, 256
 *
 */
template <typename Key, typename Value>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_static_multimap_insert(
  nvbench::state& state, nvbench::type_list<Key, Value>)
{
  std::size_t const num_keys = state.get_int64("NumInputs");
  auto const occupancy       = state.get_float64("Occupancy");
  std::size_t const num_reps = state.get_int64("NumReps");
  auto const dist            = state.get_string("Distribution");

  std::size_t const size    = num_keys / occupancy;
  std::size_t const cg_size = 8;

  std::vector<Key> h_keys(num_keys);

  generate_keys<Key>(h_keys.begin(), h_keys.end(), num_reps, dist);

  launch_nvbench_insert<Key, Value, cg_size>(state, h_keys, num_keys, size);
}

template <typename Key, typename Value>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_static_multimap_insert(
  nvbench::state& state, nvbench::type_list<Key, Value>)
{
  state.skip("Key should be the same type as Value.");
}

/**
 * @brief A benchmark evaluating multi-value count performance by varing number of repetitions
 * per key:
 * - 100'000'000 keys are inserted
 * - Map occupancy: 0.8
 * - CG size: 8
 * - Number of repetitions per key: 1, ... , 128, 256
 *
 */
template <typename Key, typename Value>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_static_multimap_count(
  nvbench::state& state, nvbench::type_list<Key, Value>)
{
  std::size_t const num_keys = state.get_int64("NumInputs");
  auto const occupancy       = state.get_float64("Occupancy");
  std::size_t const num_reps = state.get_int64("NumReps");
  auto const dist            = state.get_string("Distribution");

  std::size_t const size = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);
  generate_keys<Key>(h_keys.begin(), h_keys.end(), num_reps, dist);

  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);
  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;

    if (dist == "RANDOM") { h_keys[i] = i; }
  }

  // Get an array of unique keys
  std::set<Key> key_set(h_keys.begin(), h_keys.end());
  std::vector<Key> h_unique_keys(key_set.begin(), key_set.end());
  thrust::device_vector<Key> d_unique_keys(h_unique_keys);

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  state.add_element_count(num_keys, "NumKeys");
  state.add_global_memory_writes<Key>(num_keys * 2);

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      auto const cg_size = 8;
      cuco::static_multimap<Key, Value, cg_size> map{size, -1, -1};
      map.insert(d_pairs.begin(), d_pairs.end());

      // Use timers to explicitly mark the target region
      timer.start();
      auto count = map.count(d_unique_keys.begin(), d_unique_keys.end(), launch.get_stream());
      timer.stop();
    });
}

template <typename Key, typename Value>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_static_multimap_count(
  nvbench::state& state, nvbench::type_list<Key, Value>)
{
  state.skip("Key should be the same type as Value.");
}

template <typename Key, typename Value>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_static_multimap_count_analysis(
  nvbench::state& state, nvbench::type_list<Key, Value>)
{
  std::size_t const num_keys = state.get_int64("NumInputs");
  auto const occupancy       = state.get_float64("Occupancy");
  std::size_t const num_reps = state.get_int64("NumReps");
  auto const matching_rate   = state.get_float64("MatchingRate");

  std::size_t const size = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);
  generate_keys<Key>(h_keys.begin(), h_keys.end(), num_reps, "RANDOM");

  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  std::random_device rd;
  std::mt19937 gen{rd()};
  auto tmp_max = static_cast<double>(num_keys / num_reps) / matching_rate;
  std::uniform_int_distribution<Key> distribution{1, static_cast<Key>(tmp_max)};

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = i;

    h_keys[i] = distribution(gen);
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  state.add_element_count(num_keys, "NumKeys");
  state.add_global_memory_writes<Key>(num_keys * 2);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               auto const cg_size = 8;
               cuco::static_multimap<Key, Value, cg_size> map{size, -1, -1};
               map.insert(d_pairs.begin(), d_pairs.end());

               // Use timers to explicitly mark the target region
               timer.start();
               auto count = map.count(d_keys.begin(), d_keys.end(), launch.get_stream());
               timer.stop();
             });
}

template <typename Key, typename Value>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_static_multimap_count_analysis(
  nvbench::state& state, nvbench::type_list<Key, Value>)
{
  state.skip("Key should be the same type as Value.");
}

/**
 * @brief A benchmark evaluating multi-value retrieval performance by varing number of repetitions
 * per key:
 * - 100'000'000 keys are inserted
 * - Map occupancy: 0.8
 * - CG size: 8
 * - Number of repetitions per key: 1, ... , 128, 256
 *
 */
template <typename Key, typename Value>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_static_multimap_find_all(
  nvbench::state& state, nvbench::type_list<Key, Value>)
{
  std::size_t const num_keys = state.get_int64("NumInputs");
  auto const occupancy       = state.get_float64("Occupancy");
  std::size_t const num_reps = state.get_int64("NumReps");
  auto const dist            = state.get_string("Distribution");

  std::size_t const size = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);
  generate_keys<Key>(h_keys.begin(), h_keys.end(), num_reps, dist);

  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);
  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;

    if (dist == "RANDOM") { h_keys[i] = i; }
  }

  // Get an array of unique keys
  std::set<Key> key_set(h_keys.begin(), h_keys.end());
  std::vector<Key> h_unique_keys(key_set.begin(), key_set.end());
  thrust::device_vector<Key> d_unique_keys(h_unique_keys);

  thrust::device_vector<cuco::pair_type<Key, Value>> d_results(2 * num_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  state.add_element_count(num_keys, "NumKeys");
  state.add_global_memory_writes<Key>(num_keys * 2);

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      auto const cg_size = 8;
      cuco::static_multimap<Key, Value, cg_size> map{size, -1, -1};
      map.insert(d_pairs.begin(), d_pairs.end());

      // Use timers to explicitly mark the target region
      timer.start();
      map.find_all(
        d_unique_keys.begin(), d_unique_keys.end(), d_results.data().get(), launch.get_stream());
      timer.stop();
    });
}

template <typename Key, typename Value>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_static_multimap_find_all(
  nvbench::state& state, nvbench::type_list<Key, Value>)
{
  state.skip("Key should be the same type as Value.");
}

/**
 * @brief A benchmark evaluating multi-value retrieval performance (`count` + `find_all`) by varing
 * number of repetitions per key:
 * - 100'000'000 keys are inserted
 * - Map occupancy: 0.8
 * - CG size: 8
 * - Number of repetitions per key: 1, ... , 128, 256
 *
 */
template <typename Key, typename Value>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_static_multimap_retrieve(
  nvbench::state& state, nvbench::type_list<Key, Value>)
{
  std::size_t const num_keys = state.get_int64("NumInputs");
  auto const occupancy       = state.get_float64("Occupancy");
  std::size_t const num_reps = state.get_int64("NumReps");
  auto const dist            = state.get_string("Distribution");

  std::size_t const size = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);
  generate_keys<Key>(h_keys.begin(), h_keys.end(), num_reps, dist);

  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);
  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;

    if (dist == "RANDOM") { h_keys[i] = i; }
  }

  // Get an array of unique keys
  std::set<Key> key_set(h_keys.begin(), h_keys.end());
  std::vector<Key> h_unique_keys(key_set.begin(), key_set.end());
  thrust::device_vector<Key> d_unique_keys(h_unique_keys);

  thrust::device_vector<cuco::pair_type<Key, Value>> d_results(2 * num_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  state.add_element_count(num_keys, "NumKeys");
  state.add_global_memory_writes<Key>(num_keys * 2);

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      auto const cg_size = 8;
      cuco::static_multimap<Key, Value, cg_size> map{size, -1, -1};
      map.insert(d_pairs.begin(), d_pairs.end());

      // Use timers to explicitly mark the target region
      timer.start();
      auto count = map.count(d_unique_keys.begin(), d_unique_keys.end(), launch.get_stream());
      map.find_all(
        d_unique_keys.begin(), d_unique_keys.end(), d_results.data().get(), launch.get_stream());
      timer.stop();
    });
}

template <typename Key, typename Value>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_static_multimap_retrieve(
  nvbench::state& state, nvbench::type_list<Key, Value>)
{
  state.skip("Key should be the same type as Value.");
}

using key_type   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using value_type = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;

NVBENCH_BENCH_TYPES(nvbench_static_multimap_insert, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_type_axes_names({"Key", "Value"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_string_axis("Distribution", {"CYCLE", "RANDOM"})
  .add_int64_power_of_two_axis("NumReps", nvbench::range(0, 8, 1));

NVBENCH_BENCH_TYPES(nvbench_static_multimap_count, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_type_axes_names({"Key", "Value"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_string_axis("Distribution", {"CYCLE", "RANDOM"})
  .add_int64_power_of_two_axis("NumReps", nvbench::range(0, 8, 1));

NVBENCH_BENCH_TYPES(nvbench_static_multimap_find_all, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_type_axes_names({"Key", "Value"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_string_axis("Distribution", {"CYCLE", "RANDOM"})
  .add_int64_power_of_two_axis("NumReps", nvbench::range(0, 8, 1));

NVBENCH_BENCH_TYPES(nvbench_static_multimap_retrieve, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_type_axes_names({"Key", "Value"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_string_axis("Distribution", {"CYCLE", "RANDOM"})
  .add_int64_power_of_two_axis("NumReps", nvbench::range(0, 8, 1));

NVBENCH_BENCH_TYPES(nvbench_static_multimap_count_analysis, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_type_axes_names({"Key", "Value"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("MatchingRate", {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1})
  .add_int64_power_of_two_axis("NumReps", {3});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_count_analysis, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_type_axes_names({"Key", "Value"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("MatchingRate", {0.5})
  .add_int64_power_of_two_axis("NumReps", nvbench::range(0, 8, 1));
