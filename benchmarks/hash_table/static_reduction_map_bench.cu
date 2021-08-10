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

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <cuco/static_reduction_map.cuh>
#include <key_generator.hpp>
#include <nvbench/nvbench.cuh>
#include <util.hpp>

/**
 * @brief Enum representation for reduction operators
 */
enum class op_type { REDUCE_ADD, CUSTOM_OP, CUSTOM_OP_NO_BACKOFF };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  // Enum type:
  op_type,
  // Callable to generate input strings:
  // Short identifier used for tables, command-line args, etc.
  // Used when context is available to figure out the enum type.
  [](op_type o) {
    switch (o) {
      case op_type::REDUCE_ADD: return "REDUCE_ADD";
      case op_type::CUSTOM_OP: return "CUSTOM_OP";
      case op_type::CUSTOM_OP_NO_BACKOFF: return "CUSTOM_OP_NO_BACKOFF";
      default: return "ERROR";
    }
  },
  // Callable to generate descriptions:
  // If non-empty, these are used in `--list` to describe values.
  // Used when context may not be available to figure out the type from the
  // input string.
  // Just use `[](auto) { return std::string{}; }` if you don't want these.
  [](auto) { return std::string{}; })

/**
 * @brief Maps the enum value of a cuco reduction operator to its actual type
 */
template <op_type Op>
struct op_type_map {
};

// Sum reduction with atomic fetch-and-add
template <>
struct op_type_map<op_type::REDUCE_ADD> {
  template <typename T>
  using type = cuco::reduce_add<T>;
};

// Sum reduction with atomic compare-and-swap loop
// Note: default backoff strategy
template <>
struct op_type_map<op_type::CUSTOM_OP> {
  template <typename T>
  using type = cuco::custom_op<T, 0, thrust::plus<T>>;
};

// Sum reduction with atomic compare-and-swap loop
// Note: backoff strategy omitted
template <>
struct op_type_map<op_type::CUSTOM_OP_NO_BACKOFF> {
  template <typename T>
  using type = cuco::custom_op<T, 0, thrust::plus<T>, 0>;
};

/**
 * @brief A benchmark evaluating insert performance.
 */
template <typename Key, typename Value, op_type Op>
void nvbench_cuco_static_reduction_map_insert(
  nvbench::state& state, nvbench::type_list<Key, Value, nvbench::enum_type<Op>>)
{
  using map_type = cuco::static_reduction_map<typename op_type_map<Op>::type<Value>, Key, Value>;

  auto const num_elems    = state.get_int64("NumInputs");
  auto const occupancy    = state.get_float64("Occupancy");
  auto const dist         = state.get_string("Distribution");
  auto const multiplicity = state.get_int64_or_default("Multiplicity", 8);

  std::vector<Key> h_keys_in(num_elems);
  std::vector<Value> h_values_in(num_elems);

  if (not generate_keys<Key>(dist, h_keys_in.begin(), h_keys_in.end(), multiplicity)) {
    state.skip("Invalid distribution.");
    return;
  }

  // generate uniform random values
  generate_keys<Value>("UNIFORM", h_values_in.begin(), h_values_in.end(), 1);

  // the size of the hash table under a given target occupancy depends on the
  // number of unique keys in the input
  std::size_t const unique   = count_unique(h_keys_in.begin(), h_keys_in.end());
  std::size_t const capacity = std::ceil(SDIV(unique, occupancy));

  // alternative occupancy calculation based on the total number of inputs
  // std::size_t const capacity = num_elems / occupancy;

  thrust::device_vector<Key> d_keys_in(h_keys_in);
  thrust::device_vector<Value> d_values_in(h_values_in);

  auto d_pairs_in_begin =
    thrust::make_zip_iterator(thrust::make_tuple(d_keys_in.begin(), d_values_in.begin()));
  auto d_pairs_in_end = d_pairs_in_begin + num_elems;

  state.add_element_count(num_elems);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               map_type map{capacity, -1};

               timer.start();
               map.insert(d_pairs_in_begin, d_pairs_in_end, launch.get_stream());
               timer.stop();
             });
}

// type parameter dimensions for benchmark
using key_type_range   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using value_type_range = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using op_type_range =
  nvbench::enum_type_list<op_type::REDUCE_ADD, op_type::CUSTOM_OP, op_type::CUSTOM_OP_NO_BACKOFF>;

// benchmark setups

NVBENCH_BENCH_TYPES(nvbench_cuco_static_reduction_map_insert,
                    NVBENCH_TYPE_AXES(key_type_range, value_type_range, op_type_range))
  .set_name("cuco_static_reduction_map_insert_occupancy")
  .set_type_axes_names({"Key", "Value", "ReductionOp"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs
  .add_float64_axis("Occupancy", nvbench::range(0.5, 0.9, 0.1))  // occupancy range
  .add_int64_axis("Multiplicity", {8})  // only applies to uniform distribution
  .add_string_axis("Distribution", {"GAUSSIAN", "UNIFORM", "UNIQUE"});

// Distribution "SAME" does not work with CUSTOM_OP
NVBENCH_BENCH_TYPES(nvbench_cuco_static_reduction_map_insert,
                    NVBENCH_TYPE_AXES(key_type_range,
                                      value_type_range,
                                      nvbench::enum_type_list<op_type::REDUCE_ADD>))
  .set_name("cuco_static_reduction_map_insert_occupancy")
  .set_type_axes_names({"Key", "Value", "ReductionOp"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs
  .add_float64_axis("Occupancy", nvbench::range(0.5, 0.9, 0.1))  // occupancy range
  .add_int64_axis("Multiplicity", {8})  // only applies to uniform distribution
  .add_string_axis("Distribution", {"SAME"});

NVBENCH_BENCH_TYPES(nvbench_cuco_static_reduction_map_insert,
                    NVBENCH_TYPE_AXES(key_type_range, value_type_range, op_type_range))
  .set_name("cuco_static_reduction_map_insert_multiplicity")
  .set_type_axes_names({"Key", "Value", "ReductionOp"})
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs
  .add_float64_axis("Occupancy", nvbench::range(0.5, 0.9, 0.1))
  .add_int64_axis("Multiplicity",
                  {1, 10, 100, 1'000, 10'000, 100'000, 1'000'000})  // key multiplicity range
  .add_string_axis("Distribution", {"UNIFORM"});