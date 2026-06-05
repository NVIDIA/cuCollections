/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#pragma once

#include <cuco/detail/error.hpp>
#include <cuco/operator.hpp>

#include <thrust/device_vector.h>

#include <catch2/catch_test_macros.hpp>

namespace cuco {
namespace test {

// Per-probe-step Robin Hood layout check. The unit is the *stride group* of `cg_size * bucket_size`
// contiguous slots that one probing step examines -- a single bucket for scalar probing, the whole
// cooperative-group window for CG probing. Within a stride group the slot order is free (the probe
// step distance is identical for every slot in it, so the intra-group offset cancels in all
// comparisons), so the invariant is only meaningful *between* groups. For each occupied group `g`
// (with predecessor `pg`), the resident probe-step distances ("ages") must satisfy:
//
//   (1) Contiguity. If `g` holds any overflowed resident (distance >= 1), `pg` must be full --
//       otherwise that resident would have stopped in `pg`'s free slot instead of probing past it.
//   (2) Balance. No resident of `pg` may be more than one probing step *richer* than the poorest
//       resident of `g` (`min_age(pg) >= max_age(g) - 1`) -- otherwise the poorest resident of `g`
//       should have displaced it. This is the property that distinguishes Robin Hood from plain
//       linear probing, and (via condition 1) it inductively forces the whole home-to-position run
//       to be full.
//
// `probe_distance` is reused here -- it is exercised independently by the utility probe-distance
// test, so a bug in *insert* (a layout that violates the invariant) is still caught.
template <typename Ref>
__global__ void robin_hood_invariant_kernel(Ref ref, int* violations)
{
  using size_type        = typename Ref::size_type;
  constexpr int bs       = Ref::bucket_size;
  constexpr int stride   = Ref::cg_size * Ref::bucket_size;
  auto const storage_ref = ref.storage_ref();
  auto const slots       = storage_ref.data();
  auto const num_groups  = storage_ref.capacity() / stride;
  auto const extent      = storage_ref.extent();
  auto const empty_key   = ref.empty_key_sentinel();
  auto const erased_key  = ref.erased_key_sentinel();
  auto const scheme      = ref.probing_scheme();

  for (size_type g = blockIdx.x * blockDim.x + threadIdx.x; g < num_groups;
       g += gridDim.x * blockDim.x) {
    int occupied_g      = 0;
    size_type max_age_g = 0;
    for (int s = 0; s < stride; ++s) {
      auto const slot = slots[g * stride + s];
      if (slot.first != empty_key) {  // tombstones count as residents (erase enabled => != empty)
        ++occupied_g;
        // A tombstone keeps its age in its payload; a live key's age is its probe distance.
        auto const age = (slot.first == erased_key)
                           ? static_cast<size_type>(slot.second)
                           : scheme.template probe_distance<bs>(
                               slot.first, static_cast<size_type>(g * stride + s), extent);
        if (age > max_age_g) { max_age_g = age; }
      }
    }
    if (occupied_g == 0) { continue; }

    size_type const pg  = (g + num_groups - 1) % num_groups;
    int occupied_p      = 0;
    size_type min_age_p = 0;
    for (int s = 0; s < stride; ++s) {
      auto const slot = slots[pg * stride + s];
      if (slot.first != empty_key) {
        auto const age = (slot.first == erased_key)
                           ? static_cast<size_type>(slot.second)
                           : scheme.template probe_distance<bs>(
                               slot.first, static_cast<size_type>(pg * stride + s), extent);
        if (occupied_p == 0 || age < min_age_p) { min_age_p = age; }
        ++occupied_p;
      }
    }

    if (max_age_g >= 1 && occupied_p < stride) { atomicAdd(violations, 1); }        // (1)
    if (occupied_p > 0 && min_age_p + 1 < max_age_g) { atomicAdd(violations, 1); }  // (2)
  }
}

// Asserts that a populated Robin Hood `map` satisfies the per-bucket layout invariant above. No-op
// to call only on Robin Hood maps -- `probe_distance` exists only on `robin_hood_probing`, so guard
// the call site with `cuco::is_robin_hood_probing<...>`.
template <typename Map>
void check_robin_hood_invariant(Map& map)
{
  auto const ref = map.ref(cuco::op::find);

  thrust::device_vector<int> d_violations(1, 0);
  auto constexpr block_size = 128;
  auto const grid_size      = (map.capacity() + block_size - 1) / block_size;
  robin_hood_invariant_kernel<<<grid_size, block_size>>>(
    ref, thrust::raw_pointer_cast(d_violations.data()));
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  REQUIRE(d_violations[0] == 0);
}

}  // namespace test
}  // namespace cuco
