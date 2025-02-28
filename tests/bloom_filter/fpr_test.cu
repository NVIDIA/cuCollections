/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cuco/bloom_filter.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>
#include <exception>

using size_type = int32_t;

template <typename Filter>
void test_fpr(Filter& filter, size_type num_keys)
{
  using Key = typename Filter::key_type;

  // Generate keys
  thrust::device_vector<Key> keys(num_keys);
  thrust::sequence(thrust::device, keys.begin(), keys.end());

  size_type num_tp = num_keys * 0.5;  ///< Insert the first half keys into the filter.
  size_type num_tn = num_keys - num_tp;

  auto tp_begin = keys.begin();
  auto tp_end   = tp_begin + num_tp;
  auto tn_begin = tp_end;
  auto tn_end   = keys.end();

  filter.add(tp_begin, tp_end);

  thrust::device_vector<bool> tp_result(num_tp, false);
  thrust::device_vector<bool> tn_result(num_keys - num_tp, false);

  // Query the filter for the previously inserted keys.
  // This should result in a true-positive rate of TPR=1.
  filter.contains(tp_begin, tp_end, tp_result.begin());

  // Query the filter for the keys that are not present in the filter.
  // Since bloom filters are probalistic data structures, the filter
  // exhibits a false-positive rate FPR>0 depending on the number of bits in
  // the filter and the number of hashes used per key.
  filter.contains(tn_begin, tn_end, tn_result.begin());

  float tp_rate =
    float(thrust::count(thrust::device, tp_result.begin(), tp_result.end(), true)) / float(num_tp);
  float fp_rate =
    float(thrust::count(thrust::device, tn_result.begin(), tn_result.end(), true)) / float(num_tn);

  SECTION("True-positive rate must be 1.") { REQUIRE(tp_rate == 1.0f); }

  SECTION("Fals-positive rate should be close to the theoretical value.")
  {
    auto expected_fpr = filter.expected_false_positive_rate(num_tp);
    INFO("expected_fpr = " << expected_fpr << ", fp_rate = " << fp_rate);

    // If the expected FPR is zero, then we expect fp_rate to be zero.
    if (expected_fpr == 0.0) {
      REQUIRE(fp_rate == 0.0f);
    } else {
      // Only fail if fp_rate exceeds expected_fpr by more than 3%
      float relative_excess =
        (fp_rate > expected_fpr) ? (fp_rate - expected_fpr) / expected_fpr : 0.0f;
      REQUIRE(relative_excess <= 0.03f);
    }
  }
}

TEMPLATE_TEST_CASE_SIG(
  "bloom_filter false-positive rate tests",
  "",
  ((class Key, class Policy), Key, Policy),
  (int32_t, cuco::default_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 1>),
  (int32_t, cuco::default_filter_policy<cuco::xxhash_64<int32_t>, uint32_t, 8>),
  (int32_t, cuco::default_filter_policy<cuco::xxhash_64<int32_t>, uint64_t, 1>),
  (int32_t, cuco::default_filter_policy<cuco::xxhash_64<int32_t>, uint64_t, 8>))
{
  using filter_type =
    cuco::bloom_filter<Key, cuco::extent<size_t>, cuda::thread_scope_device, Policy>;
  constexpr size_type num_keys{400};

  uint32_t pattern_bits = Policy::words_per_block + GENERATE(0, 1, 2, 3, 4);

  // some parameter combinations might be invalid so we skip them
  try {
    [[maybe_unused]] auto policy = Policy{pattern_bits};
  } catch (std::exception const& e) {
    SKIP(e.what());
  }

  auto filter = filter_type{1000, {}, {pattern_bits}};

  test_fpr(filter, num_keys);
}
