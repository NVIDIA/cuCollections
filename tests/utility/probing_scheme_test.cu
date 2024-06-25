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

#include <test_utils.hpp>

#include <cuco/detail/utility/cuda.hpp>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/probing_scheme.cuh>

#include <thrust/device_vector.h>

#include <cooperative_groups.h>

#include <catch2/catch_template_test_macros.hpp>

#include <cstddef>
#include <cstdint>

template <class ProbingScheme, class Key, class Extent, class OutputIt>
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
    auto iter = probing_scheme(key, upper_bound);

    for (size_t i = 0; i < seq_length; ++i) {
      out_seq[i] = *iter;
      iter++;
    }
  }
}

template <class ProbingScheme, class Key, class Extent, class OutputIt>
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
      cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());

    auto iter = probing_scheme(tile, key, upper_bound);

    for (size_t i = tile.thread_rank(); i < seq_length; ++i) {
      out_seq[i] = *iter;
      iter++;
    }
  }
}

TEMPLATE_TEST_CASE_SIG(
  "probing_scheme scalar vs CGSize 1 test",
  "",
  ((typename Key, cuco::test::probe_sequence Probe, int32_t WindowSize), Key, Probe, WindowSize),
  (int32_t, cuco::test::probe_sequence::double_hashing, 1),
  (int32_t, cuco::test::probe_sequence::double_hashing, 2),
  (int64_t, cuco::test::probe_sequence::double_hashing, 1),
  (int64_t, cuco::test::probe_sequence::double_hashing, 2),
  (int32_t, cuco::test::probe_sequence::linear_probing, 1),
  (int32_t, cuco::test::probe_sequence::linear_probing, 2),
  (int64_t, cuco::test::probe_sequence::linear_probing, 1),
  (int64_t, cuco::test::probe_sequence::linear_probing, 2))
{
  auto const upper_bound = cuco::make_window_extent<1, WindowSize>(cuco::extent<std::size_t>{10});
  constexpr size_t seq_length{8};
  constexpr Key key{42};

  using probe = std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                                   cuco::linear_probing<1, cuco::default_hash_function<Key>>,
                                   cuco::double_hashing<1, cuco::default_hash_function<Key>>>;

  thrust::device_vector<size_t> scalar_seq(seq_length);
  generate_scalar_probing_sequence<probe>
    <<<1, 1>>>(key, upper_bound, seq_length, scalar_seq.begin());
  thrust::device_vector<size_t> cg_seq(seq_length);
  generate_cg_probing_sequence<probe><<<1, 1>>>(key, upper_bound, seq_length, cg_seq.begin());

  REQUIRE(cuco::test::equal(
    scalar_seq.begin(), scalar_seq.end(), cg_seq.begin(), thrust::equal_to<std::size_t>{}));
}