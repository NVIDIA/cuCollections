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

#include <cstddef>

using namespace cuco::benchmark;
using namespace cuco::utility;

template <typename T, typename InputIt>
[[nodiscard]] std::size_t exact_distinct_count(InputIt first, InputIt last)
{
  // TODO don't use detail ns in user land
  auto const num_items = cuco::detail::distance(first, last);
  if (num_items == 0) { return 0; }

  auto set = cuco::static_set{num_items, cuco::empty_key<T>{-1}};
  set.insert(first, last);
  return set.size();
}

/**
 * @brief A benchmark evaluating `cuco::distinct_count_estimator` end-to-end performance
 */
template <typename Estimator, typename Dist>
void distinct_count_estimator_e2e(nvbench::state& state, nvbench::type_list<Estimator, Dist>)
{
  using T = typename Estimator::value_type;

  auto const num_items = state.get_int64_or_default("NumInputs", 1ull << 30);

  thrust::device_vector<T> items(num_items);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), items.begin(), items.end());

  state.add_element_count(num_items);
  state.add_global_memory_reads<T>(num_items, "InputSize");

  Estimator estimator;
  estimator.add(items.begin(), items.end());

  double estimated_cardinality  = estimator.estimate();
  double const true_cardinality = exact_distinct_count<T>(items.begin(), items.end());
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

  auto const num_items = state.get_int64_or_default("NumInputs", 1ull << 30);

  thrust::device_vector<T> items(num_items);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), items.begin(), items.end());

  state.add_element_count(num_items);
  state.add_global_memory_reads<T>(num_items, "InputSize");

  Estimator estimator;
  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    estimator.clear_async({launch.get_stream()});

    timer.start();
    estimator.add_async(items.begin(), items.end(), {launch.get_stream()});
    timer.stop();
  });
}

using ESTIMATOR_RANGE = nvbench::type_list<cuco::distinct_count_estimator<nvbench::int32_t, 8>,
                                           cuco::distinct_count_estimator<nvbench::int32_t, 9>,
                                           cuco::distinct_count_estimator<nvbench::int32_t, 10>,
                                           cuco::distinct_count_estimator<nvbench::int32_t, 11>,
                                           cuco::distinct_count_estimator<nvbench::int32_t, 12>,
                                           cuco::distinct_count_estimator<nvbench::int32_t, 13>,
                                           cuco::distinct_count_estimator<nvbench::int64_t, 11>,
                                           cuco::distinct_count_estimator<nvbench::int64_t, 12>>;

NVBENCH_BENCH_TYPES(distinct_count_estimator_e2e,
                    NVBENCH_TYPE_AXES(ESTIMATOR_RANGE, nvbench::type_list<distribution::unique>))
  .set_name("distinct_count_estimator")
  .set_type_axes_names({"Estimator", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE);

NVBENCH_BENCH_TYPES(distinct_count_estimator_add,
                    NVBENCH_TYPE_AXES(ESTIMATOR_RANGE, nvbench::type_list<distribution::unique>))
  .set_name("distinct_count_estimator::add")
  .set_type_axes_names({"Estimator", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE);