/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include <test_utils.hpp>

#include <cuco/hash_functions.cuh>
#include <cuco/hyperloglog.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/device_vector.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>  // std::memcpy
#include <vector>

/**
 * @file spark_parity_test.cu
 * @brief Unit test to ensure parity with Spark's HLL implementation
 *
 * The following unit tests mimic Spark's unit tests which can be found here:
 * https://github.com/apache/spark/blob/d10dbaa31a44878df5c7e144f111e18261346531/sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/expressions/aggregate/HyperLogLogPlusPlusSuite.scala
 *
 */

// TODO implement this test once add_if is available
// TEST_CASE("hyperloglog: Spark parity: add nulls", "")

TEST_CASE("hyperloglog: Spark parity: deterministic cardinality estimation", "")
{
  using T              = int;
  using estimator_type = cuco::hyperloglog<T, cuda::thread_scope_device, cuco::xxhash_64<T>>;

  constexpr size_t repeats = 10;
  // This factor determines the error threshold for passing the test
  constexpr double tolerance_factor = 3.0;
  auto num_items          = GENERATE(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000);
  auto standard_deviation = GENERATE(0.1, 0.05, 0.025, 0.01, 0.001);

  auto expected_hll_precision = std::max(
    static_cast<int32_t>(4),
    static_cast<int32_t>(std::ceil(2.0 * std::log(1.106 / standard_deviation) / std::log(2.0))));
  auto expected_sketch_bytes = 4 * (1ull << expected_hll_precision);

  INFO("num_items" << num_items);
  INFO("standard_deviation=" << standard_deviation);
  INFO("expected_hll_precision=" << expected_hll_precision);
  INFO("expected_sketch_bytes=" << expected_sketch_bytes);

  auto sd = cuco::standard_deviation(standard_deviation);
  auto sb = cuco::sketch_size_kb(expected_sketch_bytes / 1024.0);

  // Validate sketch size calculation
  REQUIRE(estimator_type::sketch_bytes(sd) >= 64);
  REQUIRE(estimator_type::sketch_bytes(sd) == expected_sketch_bytes);
  REQUIRE(estimator_type::sketch_bytes(sd) == estimator_type::sketch_bytes(sb));

  auto items_begin =
    cuda::make_transform_iterator(cuda::make_counting_iterator<size_t>(0),
                                  cuda::proclaim_return_type<T>([repeats] __device__(auto i) {
                                    return static_cast<T>(i / repeats);
                                  }));

  estimator_type estimator{sd};

  REQUIRE(estimator.estimate() == 0);

  // Add all items to the estimator
  estimator.add(items_begin, items_begin + num_items);

  auto const estimate = estimator.estimate();

  double const relative_error =
    std::abs((static_cast<double>(estimate) / static_cast<double>(num_items / repeats)) - 1.0);
  // RSD for a given precision is given by the following formula
  double const expected_standard_deviation =
    1.04 / std::sqrt(static_cast<double>(1ull << expected_hll_precision));

  // Check if the error is acceptable
  REQUIRE(relative_error < expected_standard_deviation * tolerance_factor);
}

TEST_CASE("hyperloglog: Spark parity: regression for issue #696", "")
{
  using T              = int;
  using estimator_type = cuco::hyperloglog<T, cuda::thread_scope_device, cuco::xxhash_64<T>>;

  auto const standard_deviation   = cuco::standard_deviation{0.3};
  std::vector<T> const host_items = {
    434971005,  -1801141102, 1963272577,  -493001830,  -1087762159, 843441079,   959409252,
    252071729,  -1830233271, 820808802,   -1535782039, 1531475465,  1642188005,  552222160,
    -194998970, 2109544455,  1405026214,  1672131131,  1247840828,  -180033177,  -1286780806,
    933672832,  1401381638,  -241603026,  615622263,   -957425136,  -276735314,  -2009711680,
    -639722582, 974221725,   713012837,   -1402812678, -546850329,  -866141232,  848946484,
    -635203849, -1450175774, 844979905,   888971584,   1855780699,  -1268565561, -1185513673,
    1019479409, -1333229875, -1246182436, -2147483648, 900525526,   1006079044,  -698588704,
    -943987698, 27695788,    -84695147,   -1441291062, 397673504,   -392707402,  1290858625,
    1420750585, -1178564290, 1921246226,  188935376,   6560145,     -1928347973, 820364161,
    -401706971, -1118924186, 1759421546,  -1350108963, 2097517825,  -23883470,   -1221269093,
    1264159503, 97097882,    982791723,   638708040,   -349593807,  361658100,   341780548,
    -4171545,   1095633384,  -1694321873, 1777502952,  -1699998259, -1432813716, 1113816192,
    -966808405, 1583478695,  -650293396,  35500231,    -440874147,  995739986,   207692068,
    0,          -1243401007, -1576220155, 1868986580,  -87141217,   2108694405,  -251958436,
    2028975576, 1725957984,  -354115601,  888726314,   1032487345,  -1968749299, 1880817790,
    1113480821, 789387254,   -1724956749, -1201901245};
  thrust::device_vector<T> items = host_items;

  estimator_type estimator{standard_deviation, cuco::xxhash_64<T>{42}};
  estimator.add(items.begin(), items.end());

  REQUIRE(estimator.estimate() == 81);
}

// the following test is omitted since we refrain from doing randomized unit tests in cuco
// TEST_CASE("hyperloglog: Spark parity: random cardinality estimation", "")

TEST_CASE("hyperloglog: Spark parity: merging HLL instances", "")
{
  using T              = int;
  using estimator_type = cuco::hyperloglog<T, cuda::thread_scope_device, cuco::xxhash_64<T>>;

  auto num_items          = 1000000;
  auto standard_deviation = cuco::standard_deviation(0.05);

  auto items_begin = cuda::make_counting_iterator<T>(0);

  // count lower half of input
  estimator_type lower{standard_deviation};
  lower.add(items_begin, items_begin + num_items / 2);

  // count upper half of input
  estimator_type upper{standard_deviation};
  upper.add(items_begin + num_items / 2, items_begin + num_items);

  // merge upper into lower so lower has seen the entire input
  lower.merge(upper);

  auto reversed_items_begin = cuda::make_transform_iterator(
    items_begin, cuda::proclaim_return_type<T>([num_items] __device__(auto i) {
      return static_cast<T>(num_items - i);
    }));

  // count the entire input vector but in reversed order
  estimator_type entire{standard_deviation};
  entire.add(reversed_items_begin, reversed_items_begin + num_items);

  auto const entire_sketch = entire.sketch();
  auto const lower_sketch  = lower.sketch();

  // check if sketches are bitwise identical
  REQUIRE(cuco::test::equal(entire_sketch.data(),
                            entire_sketch.data() + entire_sketch.size(),
                            lower_sketch.data(),
                            cuda::std::equal_to{}));
}

/*
The following unit tests fail since xxhash_64 does not deduplicate different bit patterns for NaN
values and +-0.0. They are thus counted as distinct items.

TEST_CASE("hyperloglog: Spark parity: add 0.0 and -0.0", "")
{
  using T = double;
  using estimator_type =
    cuco::hyperloglog<T, cuda::thread_scope_device, cuco::xxhash_64<T>>;

  auto standard_deviation = cuco::standard_deviation(0.05);

  auto items = thrust::device_vector<T>({0.0, -0.0});

  estimator_type estimator{standard_deviation};
  estimator.add(items.begin(), items.end());

  REQUIRE(estimator.estimate() == 1);
}

TEST_CASE("hyperloglog: Spark parity: add NaN", "")
{
  using T = double;
  using estimator_type =
    cuco::hyperloglog<T, cuda::thread_scope_device, cuco::xxhash_64<T>>;

  auto standard_deviation = cuco::standard_deviation(0.05);

  // Define the special bit pattern for the NaN.
  uint64_t nan_bits = 0x7ff1234512345678ULL;
  double special_nan;
  std::memcpy(&special_nan, &nan_bits, sizeof(special_nan));

  auto items = thrust::device_vector<T>({0.0, special_nan});

  estimator_type estimator{standard_deviation};
  estimator.add(items.begin(), items.end());

  REQUIRE(estimator.estimate() == 1);
}
*/
