/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark_utils.hpp>

#include <cuco/roaring_bitmap.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/std/cstdint>
#include <thrust/device_vector.h>
#include <thrust/reverse.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>

using namespace cuco::benchmark;
using namespace cuco::utility;

enum class build_mode { indices, sorted_indices, sorted_unique_indices };

template <build_mode Mode, class Dist>
void roaring_bitmap_build(nvbench::state& state, nvbench::type_list<Dist>)
{
  using index_type  = cuda::std::uint32_t;
  using bitmap_type = cuco::experimental::roaring_bitmap<index_type>;

  auto const num_inputs = state.get_int64("NumInputs");
  thrust::device_vector<index_type> indices(num_inputs);

  [[maybe_unused]] key_generator generator{};
  if constexpr (Mode == build_mode::sorted_unique_indices) {
    thrust::sequence(indices.begin(), indices.end());
  } else {
    generator.generate(dist_from_state<Dist>(state), indices.begin(), indices.end());
    if constexpr (Mode == build_mode::sorted_indices) {
      thrust::sort(indices.begin(), indices.end());
    }
  }

  state.add_element_count(num_inputs);
  state.add_global_memory_reads<index_type>(num_inputs, "InputSize");

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               if constexpr (Mode == build_mode::indices) {
                 [[maybe_unused]] auto bitmap = bitmap_type::from_indices(
                   indices.begin(), indices.end(), {}, cuda::stream_ref{launch.get_stream()});
                 timer.stop();
               } else if constexpr (Mode == build_mode::sorted_indices) {
                 [[maybe_unused]] auto bitmap = bitmap_type::from_sorted_indices(
                   indices.begin(), indices.end(), {}, cuda::stream_ref{launch.get_stream()});
                 timer.stop();
               } else {
                 [[maybe_unused]] auto bitmap = bitmap_type::from_sorted_unique_indices(
                   indices.begin(), indices.end(), {}, cuda::stream_ref{launch.get_stream()});
                 timer.stop();
               }
             });
}

template <class Dist>
void roaring_bitmap_from_indices(nvbench::state& state, nvbench::type_list<Dist> types)
{
  roaring_bitmap_build<build_mode::indices>(state, types);
}

template <class Dist>
void roaring_bitmap_from_sorted_indices(nvbench::state& state, nvbench::type_list<Dist> types)
{
  roaring_bitmap_build<build_mode::sorted_indices>(state, types);
}

template <class Dist>
void roaring_bitmap_from_sorted_unique_indices(nvbench::state& state,
                                               nvbench::type_list<Dist> types)
{
  roaring_bitmap_build<build_mode::sorted_unique_indices>(state, types);
}

void roaring_bitmap_from_indices_array_containers(nvbench::state& state)
{
  using index_type  = cuda::std::uint32_t;
  using bitmap_type = cuco::experimental::roaring_bitmap<index_type>;

  constexpr cuda::std::int64_t num_containers = 1 << 16;
  auto const cardinality                      = state.get_int64("ContainerCardinality");
  auto const num_inputs                       = num_containers * cardinality;
  thrust::device_vector<index_type> indices(num_inputs);

  thrust::tabulate(
    indices.begin(), indices.end(), [cardinality] __device__(cuda::std::int64_t index) {
      auto const container = static_cast<index_type>(index / cardinality);
      auto const lower     = static_cast<index_type>(index % cardinality);
      return (container << 16) | lower;
    });
  thrust::reverse(indices.begin(), indices.end());

  state.add_element_count(num_inputs);
  state.add_global_memory_reads<index_type>(num_inputs, "InputSize");

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               [[maybe_unused]] auto bitmap = bitmap_type::from_indices(
                 indices.begin(), indices.end(), {}, cuda::stream_ref{launch.get_stream()});
               timer.stop();
             });
}

NVBENCH_BENCH_TYPES(roaring_bitmap_from_indices,
                    NVBENCH_TYPE_AXES(nvbench::type_list<distribution::unique>))
  .set_name("roaring_bitmap_from_indices_unique")
  .set_type_axes_names({"Distribution"})
  .add_int64_power_of_two_axis("NumInputs", {20, 24, 28})
  .add_int64_axis("Multiplicity", {1});

NVBENCH_BENCH(roaring_bitmap_from_indices_array_containers)
  .set_name("roaring_bitmap_from_indices_array_containers")
  .add_int64_axis("ContainerCardinality", {1, 8, 64, 512, 4096});

NVBENCH_BENCH_TYPES(roaring_bitmap_from_indices,
                    NVBENCH_TYPE_AXES(nvbench::type_list<distribution::uniform>))
  .set_name("roaring_bitmap_from_indices_uniform")
  .set_type_axes_names({"Distribution"})
  .add_int64_power_of_two_axis("NumInputs", {20, 24, 28})
  .add_int64_axis("Multiplicity", {2, 8, 32});

NVBENCH_BENCH_TYPES(roaring_bitmap_from_sorted_indices,
                    NVBENCH_TYPE_AXES(nvbench::type_list<distribution::uniform>))
  .set_name("roaring_bitmap_from_sorted_indices")
  .set_type_axes_names({"Distribution"})
  .add_int64_power_of_two_axis("NumInputs", {20, 24, 28})
  .add_int64_axis("Multiplicity", {2, 8, 32});

NVBENCH_BENCH_TYPES(roaring_bitmap_from_sorted_unique_indices,
                    NVBENCH_TYPE_AXES(nvbench::type_list<distribution::unique>))
  .set_name("roaring_bitmap_from_sorted_unique_indices")
  .set_type_axes_names({"Distribution"})
  .add_int64_power_of_two_axis("NumInputs", {20, 24, 28})
  .add_int64_axis("Multiplicity", {1});
