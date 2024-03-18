/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <defaults.hpp>
#include <utils.hpp>

#include <cuco/distinct_count_estimator.cuh>
#include <cuco/static_set.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda/functional>

#include <cstddef>

using namespace cuco::benchmark;
using namespace cuco::utility;

template <typename InputIt>
[[nodiscard]] std::size_t exact_distinct_count(InputIt first, std::size_t n)
{
  // TODO static_set currently only supports types up-to 8-bytes in size.
  // Casting is valid since the keys generated are representable in int64_t.
  using T = std::int64_t;

  auto cast_iter = thrust::make_transform_iterator(
    first, cuda::proclaim_return_type<T>([] __device__(auto i) { return static_cast<T>(i); }));

  auto set = cuco::static_set{n, 0.8, cuco::empty_key<T>{-1}};
  set.insert(cast_iter, cast_iter + n);
  return set.size();
}

/**
 * @brief A benchmark evaluating `cuco::distinct_count_estimator` end-to-end performance
 */
template <typename Estimator, typename Dist>
void distinct_count_estimator_e2e(nvbench::state& state, nvbench::type_list<Estimator, Dist>)
{
  using T = typename Estimator::value_type;

  auto const num_items      = state.get_int64("NumInputs");
  auto const sketch_size_kb = state.get_int64("SketchSizeKB");

  thrust::device_vector<T> items(num_items);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), items.begin(), items.end());

  state.add_element_count(num_items);
  state.add_global_memory_reads<T>(num_items, "InputSize");

  Estimator estimator(sketch_size_kb);
  estimator.add(items.begin(), items.end());

  double estimated_cardinality  = estimator.estimate();
  double const true_cardinality = exact_distinct_count(items.begin(), num_items);
  auto const relative_error     = abs(true_cardinality - estimated_cardinality) / true_cardinality;

  auto& summ = state.add_summary("RelativeError");
  summ.set_string("hint", "RelErr");
  summ.set_string("short_name", "RelativeError");
  summ.set_string("description", "Relatve approximation error.");
  summ.set_float64("value", relative_error);

  estimator.clear();
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               estimator.clear_async({launch.get_stream()});

               timer.start();
               estimator.add_async(items.begin(), items.end(), {launch.get_stream()});
               estimated_cardinality = estimator.estimate({launch.get_stream()});
               timer.stop();
             });
}

/**
 * @brief A benchmark evaluating `cuco::distinct_count_estimator::add` performance
 */
template <typename Estimator, typename Dist>
void distinct_count_estimator_add(nvbench::state& state, nvbench::type_list<Estimator, Dist>)
{
  using T = typename Estimator::value_type;

  auto const num_items      = state.get_int64("NumInputs");
  auto const sketch_size_kb = state.get_int64("SketchSizeKB");

  thrust::device_vector<T> items(num_items);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), items.begin(), items.end());

  state.add_element_count(num_items);
  state.add_global_memory_reads<T>(num_items, "InputSize");

  Estimator estimator(sketch_size_kb);
  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    estimator.clear_async({launch.get_stream()});

    timer.start();
    estimator.add_async(items.begin(), items.end(), {launch.get_stream()});
    timer.stop();
  });
}

using ESTIMATOR_RANGE = nvbench::type_list<cuco::distinct_count_estimator<nvbench::int32_t>,
                                           cuco::distinct_count_estimator<nvbench::int64_t>,
                                           cuco::distinct_count_estimator<__int128_t>>;

NVBENCH_BENCH_TYPES(distinct_count_estimator_e2e,
                    NVBENCH_TYPE_AXES(ESTIMATOR_RANGE, nvbench::type_list<distribution::unique>))
  .set_name("distinct_count_estimator_e2e")
  .set_type_axes_names({"Estimator", "Distribution"})
  .add_int64_power_of_two_axis("NumInputs", {28, 29, 30})
  .add_int64_axis("SketchSizeKB", {8, 16, 32})
  .set_max_noise(defaults::MAX_NOISE);

NVBENCH_BENCH_TYPES(distinct_count_estimator_add,
                    NVBENCH_TYPE_AXES(ESTIMATOR_RANGE, nvbench::type_list<distribution::unique>))
  .set_name("distinct_count_estimator::add_async")
  .set_type_axes_names({"Estimator", "Distribution"})
  .add_int64_power_of_two_axis("NumInputs", {28, 29, 30})
  .add_int64_axis("SketchSizeKB", {8, 16, 32})
  .set_max_noise(defaults::MAX_NOISE);