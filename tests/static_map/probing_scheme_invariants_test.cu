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

#include <test_utils.hpp>

#include <cuco/static_map.cuh>

#include <cuda/std/functional>
#include <thrust/device_vector.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

namespace {

// Identity hash. `cuco::detail::sanitize_hash` is `to_positive`, so for a non-negative key `k` this
// puts its home bucket at slot `k % capacity` — letting the test hand-craft an exact Robin Hood
// layout instead of reverse-engineering a real hash function.
template <typename Key>
struct identity_hash {
  __host__ __device__ constexpr Key operator()(Key key) const noexcept { return key; }
};

}  // namespace

// Validates the Robin Hood read-path early-exit (`find` / `contains`) against a hand-seeded layout,
// before the displacing `insert` exists. A scalar (cg_size == 1) Robin Hood map over a linear probe
// sequence is seeded directly through the storage pointer with a known-valid Robin Hood cluster,
// then queried with keys chosen to exercise each lookup-termination rule.
TEST_CASE("static_map robin_hood read-path early-exit", "")
{
  using Key               = std::int32_t;
  using Value             = std::int32_t;
  using size_type         = std::int32_t;
  auto constexpr capacity = size_type{16};

  using extent_type = cuco::extent<size_type, capacity>;
  using probe_type  = cuco::robin_hood_probing<cuco::linear_probing<1, identity_hash<Key>>>;
  using map_type    = cuco::static_map<Key,
                                       Value,
                                       extent_type,
                                       cuda::thread_scope_device,
                                       cuda::std::equal_to<Key>,
                                       probe_type,
                                       cuco::cuda_allocator<cuda::std::byte>,
                                       cuco::storage<1>>;
  using value_type  = typename map_type::value_type;  // cuco::pair<Key, Value>

  auto map = map_type{extent_type{}, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};
  // Ensure the constructor's slot initialization (empty sentinels) has completed before we seed.
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  // Hand-seed a valid Robin Hood layout (identity hash => home(k) = k % capacity):
  //   slot 0: key 0   home 0, distance 0
  //   slot 1: key 16  home 0, distance 1   (displaced past key 0)
  //   slot 2: key 2   home 2, distance 0
  //   slots 3..15:    empty
  // This is exactly what inserting {0, 16, 2} would produce, so it satisfies the Robin Hood
  // invariant. We only write the three occupied (and contiguous) slots; the rest stay empty.
  // The ref is used purely to reach the storage pointer; the bulk queries below build their own.
  auto const ref = map.ref(cuco::op::find);
  std::vector<value_type> const seed{value_type{0, 0}, value_type{16, 16}, value_type{2, 2}};
  REQUIRE(cudaMemcpy(ref.storage_ref().data(),
                     seed.data(),
                     seed.size() * sizeof(value_type),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  // Probe keys chosen to exercise every lookup-termination rule:
  //   0  : present, found immediately at its home (distance 0).
  //   16 : present at distance 1 — found only if we do NOT early-exit at slot 0, where the resident
  //        distance (0) equals our probe step (0). Guards the strict `<` (vs `<=`) richer rule.
  //   2  : present at its home.
  //   32 : home 0, absent — terminates via the richer-resident early-exit at slot 2 (key 2 sits at
  //        distance 0 < our probe step 2), before reaching the empty slot 3.
  //   1  : home 1, absent — also terminates via the early-exit at slot 2.
  //   3  : home 3, absent — terminates on the empty slot 3.
  std::vector<Key> const probe_keys{0, 16, 2, 32, 1, 3};
  std::vector<bool> const expected_contained{true, true, true, false, false, false};

  thrust::device_vector<Key> const d_keys(probe_keys.begin(), probe_keys.end());

  thrust::device_vector<bool> d_contained(probe_keys.size());
  map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());
  for (std::size_t i = 0; i < probe_keys.size(); ++i) {
    INFO("contains, probe key = " << probe_keys[i]);
    REQUIRE(static_cast<bool>(d_contained[i]) == expected_contained[i]);
  }

  // `find` must return the stored value for present keys and the empty-value sentinel otherwise.
  thrust::device_vector<Value> d_values(probe_keys.size());
  map.find(d_keys.begin(), d_keys.end(), d_values.begin());
  std::vector<Value> const expected_values{0, 16, 2, -1, -1, -1};
  for (std::size_t i = 0; i < probe_keys.size(); ++i) {
    INFO("find, probe key = " << probe_keys[i]);
    REQUIRE(d_values[i] == expected_values[i]);
  }
}
