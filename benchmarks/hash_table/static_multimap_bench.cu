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

template <typename Key, typename OutputIt>
static void generate_keys(OutputIt output_begin, OutputIt output_end, const std::string& dist)
{
  auto num_keys = std::distance(output_begin, output_end);

  std::random_device rd;
  std::mt19937 gen{rd()};

  if (dist == "UNIQUE") {
    for (auto i = 0; i < num_keys; ++i) {
      output_begin[i] = i;
    }
  } else if (dist == "UNIFORM") {
    std::uniform_int_distribution<Key> distribution{0, std::numeric_limits<Key>::max()};
    for (auto i = 0; i < num_keys; ++i) {
      output_begin[i] = distribution(gen);
    }
  } else if (dist == "GAUSSIAN") {
    std::normal_distribution<> dg{1e9, 1e7};
    for (auto i = 0; i < num_keys; ++i) {
      output_begin[i] = std::abs(static_cast<Key>(dg(gen)));
    }
  }
}

template <typename Key, typename OutputIt>
static void generate_multikeys(OutputIt output_begin, OutputIt output_end, size_t const num_reps)
{
  auto num_keys = std::distance(output_begin, output_end);

  std::random_device rd;
  std::mt19937 gen{rd()};

  std::uniform_int_distribution<Key> distribution{1, static_cast<Key>(num_keys / num_reps)};
  for (auto i = 0; i < num_keys; ++i) {
    output_begin[i] = distribution(gen);
  }
}

template <typename Key, typename Value>
void launch_nvbench_insert(nvbench::state& state,
                           std::vector<Key> const& h_keys,
                           const std::size_t num_keys,
                           const std::size_t size)
{
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  state.add_element_count(num_keys, "NumKeys");
  state.add_global_memory_writes<Key>(num_keys * 2);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               cuco::static_multimap<Key, Value> map{size, -1, -1};
               auto m_view = map.get_device_mutable_view();

               auto const block_size = 128;
               auto const stride     = 1;
               auto const tile_size  = 4;
               auto const grid_size =
                 (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

               using Hash     = cuco::detail::MurmurHash3_32<Key>;
               using KeyEqual = thrust::equal_to<Key>;

               Hash hash;
               KeyEqual key_equal;

               timer.start();
               cuco::detail::insert<block_size, tile_size>
                 <<<grid_size, block_size, 0, launch.get_stream()>>>(
                   d_pairs.begin(), d_pairs.end(), m_view, hash, key_equal);
               CUCO_CUDA_TRY(cudaDeviceSynchronize());
               timer.stop();
             });
}

template <typename Key, typename Value>
void nvbench_static_multimap_single_insert(nvbench::state& state, nvbench::type_list<Key, Value>)
{
  if (not std::is_same<Key, Value>::value) {
    state.skip("Key should be the same type as Value.");
    return;
  }

  std::size_t const num_keys = state.get_int64("NumInputs");
  auto const occupancy       = state.get_float64("Occupancy");
  std::size_t const size     = num_keys / occupancy;
  auto const dist            = state.get_string("Distribution");

  std::vector<Key> h_keys(num_keys);

  generate_keys<Key>(h_keys.begin(), h_keys.end(), dist);

  launch_nvbench_insert<Key, Value>(state, h_keys, num_keys, size);
}

template <typename Key, typename Value>
void nvbench_static_multimap_multi_insert(nvbench::state& state, nvbench::type_list<Key, Value>)
{
  if (not std::is_same<Key, Value>::value) {
    state.skip("Key should be the same type as Value.");
    return;
  }

  std::size_t const num_keys = state.get_int64("NumInputs");
  auto const occupancy       = state.get_float64("Occupancy");
  std::size_t const size     = num_keys / occupancy;
  std::size_t const num_reps = state.get_int64("NumReps");

  std::vector<Key> h_keys(num_keys);

  generate_multikeys<Key>(h_keys.begin(), h_keys.end(), num_reps);

  launch_nvbench_insert<Key, Value>(state, h_keys, num_keys, size);
}

template <typename Key, typename Value>
void nvbench_static_multimap_find(nvbench::state& state, nvbench::type_list<Key, Value>)
{
  if (not std::is_same<Key, Value>::value) {
    state.skip("Key should be the same type as Value.");
    return;
  }

  std::size_t const num_keys = state.get_int64("NumInputs");
  auto const occupancy       = state.get_float64("Occupancy");
  std::size_t const size     = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Key>(h_keys.begin(), h_keys.end(), "UNIFORM");

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<Value> d_results(num_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  state.add_element_count(num_keys, "NumKeys");
  state.add_global_memory_writes<Key>(num_keys * 2);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               cuco::static_multimap<Key, Value> map{size, -1, -1};
               map.insert(d_pairs.begin(), d_pairs.end());

               auto view = map.get_device_view();

               auto const block_size = 128;
               auto const stride     = 1;
               auto const tile_size  = 4;
               auto const grid_size =
                 (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

               using Hash     = cuco::detail::MurmurHash3_32<Key>;
               using KeyEqual = thrust::equal_to<Key>;

               Hash hash;
               KeyEqual key_equal;

               // Use timers to explicitly mark the target region
               timer.start();
               cuco::detail::find<block_size, tile_size, Value>
                 <<<grid_size, block_size, 0, launch.get_stream()>>>(
                   d_keys.begin(), d_keys.end(), d_results.begin(), view, hash, key_equal);
               CUCO_CUDA_TRY(cudaDeviceSynchronize());
               timer.stop();
             });
}

template <typename Key, typename Value>
void nvbench_static_multimap_find_all(nvbench::state& state, nvbench::type_list<Key, Value>)
{
  if (not std::is_same<Key, Value>::value) {
    state.skip("Key should be the same type as Value.");
    return;
  }

  std::size_t const num_keys = state.get_int64("NumInputs");
  auto const occupancy       = state.get_float64("Occupancy");
  std::size_t const size     = num_keys / occupancy;
  std::size_t const num_reps = state.get_int64("NumReps");

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_multikeys<Key>(h_keys.begin(), h_keys.end(), num_reps);

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;

    h_keys[i] = i;
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_results(2 * num_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  state.add_element_count(num_keys, "NumKeys");
  state.add_global_memory_writes<Key>(num_keys * 2);

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      cuco::static_multimap<Key, Value> map{size, -1, -1};
      map.insert(d_pairs.begin(), d_pairs.end());

      auto view = map.get_device_view();

      auto const block_size = 128;
      auto const stride     = 1;
      auto const tile_size  = 4;
      auto const grid_size =
        (tile_size * num_keys + stride * block_size - 1) / (stride * block_size);

      using Hash     = cuco::detail::MurmurHash3_32<Key>;
      using KeyEqual = thrust::equal_to<Key>;

      Hash hash;
      KeyEqual key_equal;

      using atomic_ctr_type = typename cuco::static_multimap<Key, Value>::atomic_ctr_type;
      atomic_ctr_type* num_items;
      CUCO_CUDA_TRY(cudaMallocManaged(&num_items, sizeof(atomic_ctr_type)));
      *num_items = 0;
      int device_id;
      CUCO_CUDA_TRY(cudaGetDevice(&device_id));
      CUCO_CUDA_TRY(cudaMemPrefetchAsync(num_items, sizeof(atomic_ctr_type), device_id));

      // Use timers to explicitly mark the target region
      timer.start();
      cuco::detail::find_all<block_size, tile_size, Key, Value>
        <<<grid_size, block_size, 0, launch.get_stream()>>>(
          d_keys.begin(), d_keys.end(), d_results.begin(), num_items, view, hash, key_equal);
      CUCO_CUDA_TRY(cudaDeviceSynchronize());
      timer.stop();
    });
}

using key_type   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using value_type = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;

NVBENCH_BENCH_TYPES(nvbench_static_multimap_single_insert, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_type_axes_names({"Key", "Value"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1))
  .add_string_axis("Distribution", {"UNIQUE", "UNIFORM", "GAUSSIAN"});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_multi_insert, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_type_axes_names({"Key", "Value"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_int64_power_of_two_axis("NumReps", nvbench::range(0, 8, 1));

NVBENCH_BENCH_TYPES(nvbench_static_multimap_find, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_type_axes_names({"Key", "Value"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1));

NVBENCH_BENCH_TYPES(nvbench_static_multimap_find_all, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_type_axes_names({"Key", "Value"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.4})
  .add_int64_power_of_two_axis("NumReps", nvbench::range(0, 8, 1));
