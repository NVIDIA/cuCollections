/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cuco/detail/utility/cuda.hpp>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/probing_scheme.cuh>

#include <cuda/std/functional>
#include <thrust/device_vector.h>

#include <cooperative_groups.h>

#include <catch2/catch_template_test_macros.hpp>

#include <cstddef>
#include <cstdint>

template <int32_t BucketSize, class ProbingScheme, class Key, class Extent, class OutputIt>
__global__ void generate_scalar_probing_sequence(Key key,
                                                 Extent upper_bound,
                                                 size_t seq_length,
                                                 OutputIt out_seq)
{
  auto constexpr cg_size = ProbingScheme::cg_size;
  static_assert(cg_size == 1, "Invalid CG size");

  auto const tid      = blockIdx.x * blockDim.x + threadIdx.x;
  auto probing_scheme = ProbingScheme{};

  if (tid == 0) {
    auto iter = probing_scheme.template make_iterator<BucketSize>(key, upper_bound);

    for (size_t i = 0; i < seq_length; ++i) {
      out_seq[i] = *iter;
      ++iter;
    }
  }
}

template <int32_t BucketSize, class ProbingScheme, class Key, class Extent, class OutputIt>
__global__ void generate_cg_probing_sequence(Key key,
                                             Extent upper_bound,
                                             size_t seq_length,
                                             OutputIt out_seq)
{
  auto constexpr cg_size = ProbingScheme::cg_size;

  auto const tid      = blockIdx.x * blockDim.x + threadIdx.x;
  auto probing_scheme = ProbingScheme{};

  if (tid < cg_size) {
    auto const tile =
      cooperative_groups::tiled_partition<cg_size, cooperative_groups::thread_block>(
        cooperative_groups::this_thread_block());

    auto iter = probing_scheme.template make_iterator<BucketSize>(tile, key, upper_bound);

    for (size_t i = tile.thread_rank(); i < seq_length; ++i) {
      out_seq[i] = *iter;
      ++iter;
    }
  }
}

// Walks each lane's probe iterator and records, at every step, the probe distance reported for the
// slot that lane is visiting. Because the resident under test is `key` itself, the slot visited at
// step `i` is at probe distance `i` from `key`'s home — for every lane. Recording one column per
// lane lets the host check both that `probe_distance` inverts `make_iterator` and that it strips
// the per-lane intra-stride offset (so all lanes at a given step agree).
template <int32_t BucketSize, class ProbingScheme, class Key, class Extent, class OutputIt>
__global__ void generate_cg_probe_distance_sequence(Key key,
                                                    Extent upper_bound,
                                                    size_t seq_length,
                                                    OutputIt out_seq)
{
  auto constexpr cg_size = ProbingScheme::cg_size;

  auto const tid      = blockIdx.x * blockDim.x + threadIdx.x;
  auto probing_scheme = ProbingScheme{};

  if (tid < cg_size) {
    auto const tile =
      cooperative_groups::tiled_partition<cg_size, cooperative_groups::thread_block>(
        cooperative_groups::this_thread_block());

    auto iter = probing_scheme.template make_iterator<BucketSize>(tile, key, upper_bound);

    for (size_t i = 0; i < seq_length; ++i) {
      out_seq[i * cg_size + tile.thread_rank()] =
        probing_scheme.template probe_distance<BucketSize>(key, *iter, upper_bound);
      ++iter;
    }
  }
}

TEMPLATE_TEST_CASE_SIG(
  "utility probing_scheme tests",
  "",
  ((typename Key, cuco::test::probe_sequence Probe, int32_t BucketSize), Key, Probe, BucketSize),
  (int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, cuco::test::probe_sequence::linear_probing, 2))
{
  using probing_scheme_t = cuco::linear_probing<1, cuco::default_hash_function<int>>;
  auto const upper_bound = cuco::make_valid_extent<probing_scheme_t, cuco::storage<BucketSize>>(
    cuco::extent<std::size_t>{10});
  constexpr size_t seq_length{8};
  constexpr Key key{42};

  using probe = std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                                   cuco::linear_probing<1, cuco::default_hash_function<Key>>,
                                   cuco::double_hashing<1, cuco::default_hash_function<Key>>>;

  thrust::device_vector<size_t> scalar_seq(seq_length);
  generate_scalar_probing_sequence<BucketSize, probe>
    <<<1, 1>>>(key, upper_bound, seq_length, scalar_seq.begin());
  thrust::device_vector<size_t> cg_seq(seq_length);
  generate_cg_probing_sequence<BucketSize, probe>
    <<<1, 1>>>(key, upper_bound, seq_length, cg_seq.begin());

  REQUIRE(cuco::test::equal(
    scalar_seq.begin(), scalar_seq.end(), cg_seq.begin(), cuda::std::equal_to<std::size_t>{}));
}

TEMPLATE_TEST_CASE_SIG(
  "utility robin_hood probe_distance inverts make_iterator",
  "",
  ((typename Key, int32_t CGSize, int32_t BucketSize), Key, CGSize, BucketSize),
  (int32_t, 1, 1),
  (int32_t, 4, 1),
  (int32_t, 8, 1),
  (int32_t, 8, 2),
  (int64_t, 4, 1),
  (int64_t, 8, 2))
{
  // Robin Hood wraps a linear probe sequence; `probe_distance` is its inverse. For `key`'s own
  // probe sequence, the slot visited at step `i` must report probe distance `i`.
  using probe =
    cuco::robin_hood_probing<cuco::linear_probing<CGSize, cuco::default_hash_function<Key>>>;

  // A deliberately small capacity, so the probe sequence wraps around the table within
  // `seq_length` steps — this exercises the modular-subtraction (wrap) path in `probe_distance`.
  auto const upper_bound =
    cuco::make_valid_extent<probe, cuco::storage<BucketSize>>(cuco::extent<std::size_t>{64});

  // Probe distance is measured in whole probing steps and lives in `[0, num_buckets)`, where one
  // step spans the full `cg_size * bucket_size` stride. Taking `seq_length` past `num_buckets`
  // guarantees the walk wraps at least once.
  auto const capacity    = static_cast<std::size_t>(upper_bound);
  auto const num_buckets = capacity / (CGSize * BucketSize);
  auto const seq_length  = num_buckets + 3;
  constexpr Key key{42};

  thrust::device_vector<std::size_t> distances(seq_length * CGSize);
  generate_cg_probe_distance_sequence<BucketSize, probe>
    <<<1, CGSize>>>(key, upper_bound, seq_length, distances.begin());

  // Under wrap, the slot visited at step `i` sits at probe distance `i mod num_buckets`.
  for (std::size_t i = 0; i < seq_length; ++i) {
    for (std::int32_t r = 0; r < CGSize; ++r) {
      REQUIRE(distances[i * CGSize + r] == i % num_buckets);
    }
  }
}
