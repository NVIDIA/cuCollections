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

#include <cuda_benchmark.h>
#include <cuco/static_reduction_map.cuh>
#include <iostream>
#include <string>
#include <util.hpp>
#include "fmt/core.h"
#include "fmt/format.h"

template <typename T>
std::string get_type_str();

template <>
std::string get_type_str<std::uint32_t>()
{
  return "U32";
}
template <>
std::string get_type_str<std::uint64_t>()
{
  return "U64";
}

/**
 * @brief Device-side benchmark for shared memory reduction hash table insert.
 *
 * @tparam Key The hash table's key type
 * @tparam Value The hash table's value/reduction type
 * @param controller Benchmark controller/state handler
 * @param bench_name Benchmark identifier
 * @param num_elems_log2 Total number of key/value pairs to be inserted (log2)
 * @param multiplicity_log2 Number of times each key occures in the input (log2)
 * @param occupancy Target occupancy of the hash table after inserting all elements
 */
template <typename Key, typename Value>
void static_reduction_map_smem_insert_bench(cuda_benchmark::controller &controller,
                                            std::string const &bench_name,
                                            std::uint32_t num_elems_log2,
                                            std::uint32_t multiplicity_log2,
                                            float occupancy)
{
  using map_type = cuco::
    static_reduction_map<typename cuco::reduce_add<Value>, Key, Value, cuda::thread_scope_block>;
  using pair_type = typename map_type::value_type;

  auto const num_elems    = 1UL << num_elems_log2;
  auto const multiplicity = 1UL << multiplicity_log2;

  std::string full_bench_name = "INSERT " + bench_name + " key_type=" + get_type_str<Key>() +
                                " value_type=" + get_type_str<Value>() +
                                " num_elems=" + std::to_string(num_elems) +
                                " occupancy=" + fmt::format("{:.2f}", occupancy) +
                                " multiplicity=" + std::to_string(multiplicity);

  static constexpr std::size_t max_smem_bytes = 49152;  // 48 KB
  static constexpr std::size_t max_capacity   = max_smem_bytes / sizeof(pair_type);

  auto const elems_per_thread = num_elems / controller.get_block_size();
  auto const num_unique_keys  = num_elems / multiplicity;
  auto const capacity         = std::ceil(num_unique_keys / occupancy);

  if (capacity > max_capacity) {
    std::cerr << "[ERROR] (" + full_bench_name + ") Not enough shared memory available. ("
              << capacity * sizeof(pair_type) << ">" << max_capacity * sizeof(pair_type)
              << " bytes)\n";
    return;
  }

  controller.benchmark(std::string{full_bench_name}, [=] __device__(cuda_benchmark::state & state) {
    using map_type = typename cuco::static_reduction_map<typename cuco::reduce_add<Value>, Key, Value, cuda::thread_scope_block>;
    using map_view_type = typename map_type::device_mutable_view<>;

    __shared__ char sm_buffer[max_smem_bytes];

    auto g   = cooperative_groups::this_thread_block();
    auto map = map_view_type::make_from_uninitialized_slots(g, reinterpret_cast<typename map_type::slot_type *>(&sm_buffer[0]), capacity, ~Key(0));

    g.sync();

    for (auto _ : state) {
      for (Key i = g.thread_rank(); i < num_elems; i += g.size()) {
        map.insert(cuco::pair<Key, Value>((i & (multiplicity - 1)), g.thread_rank()));
      }
      g.sync();
    }
    state.set_operations_processed(state.max_iterations() * elems_per_thread);
  });
}

int main()
{
  int device_id{};
  cudaGetDevice(&device_id);

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device_id);

  int peak_clk{};
  cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, device_id);

  // can be used to calculate throughput (ops/second)
  std::cout << "GPU Clock Rate: " << std::to_string(peak_clk) << " KHz\n";

  // start one CUDA block with 1024 threads
  cuda_benchmark::controller controller(1024, 1);

  // unique keys; total number of keys fix; varying table occupancy
  for (float occupancy = 0.5; occupancy < 1.0; occupancy += 0.1) {
    static_reduction_map_smem_insert_bench<std::uint32_t, std::uint32_t>(
      controller, "OCCUPANCY", 10, 0, occupancy);
  }

  // unique keys; total number of keys fix; varying table occupancy
  for (float occupancy = 0.5; occupancy < 1.0; occupancy += 0.1) {
    static_reduction_map_smem_insert_bench<std::uint64_t, std::uint64_t>(
      controller, "OCCUPANCY", 10, 0, occupancy);
  }

  // total number of keys fix; occuoancy fix; varying key multiplicity
  for (float multiplicity_log2 = 1; multiplicity_log2 < 7; ++multiplicity_log2) {
    static_reduction_map_smem_insert_bench<std::uint32_t, std::uint32_t>(
      controller, "MULTIPLICITY", 12, multiplicity_log2, 0.8);
  }

  // total number of keys fix; occuoancy fix; varying key multiplicity
  for (float multiplicity_log2 = 1; multiplicity_log2 < 7; ++multiplicity_log2) {
    static_reduction_map_smem_insert_bench<std::uint64_t, std::uint64_t>(
      controller, "MULTIPLICITY", 12, multiplicity_log2, 0.8);
  }

  // occupancy fix; capacity fix; varying number of keys; varying key multiplicity
  for (float i = 0; i < 11; ++i) {
    static_reduction_map_smem_insert_bench<std::uint32_t, std::uint32_t>(
      controller, "EQUAL CAPACITY", 10 + i, 0 + i, 0.8);
  }

  // occupancy fix; capacity fix; varying number of keys; varying key multiplicity
  for (float i = 0; i < 11; ++i) {
    static_reduction_map_smem_insert_bench<std::uint64_t, std::uint64_t>(
      controller, "EQUAL CAPACITY", 10 + i, 0 + i, 0.8);
  }
}