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

#include <cuco/detail/error.hpp>
#include <cuco/detail/open_addressing/robin_hood/open_addressing_ref_impl.cuh>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/operator.hpp>
#include <cuco/probing_scheme.cuh>
#include <cuco/static_map.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

#include <cooperative_groups.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>

namespace {

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
// `probe_distance` is reused here -- it is exercised independently by the probe-distance test
// below, so a bug in *insert* (a layout that violates the invariant) is still caught.
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
        auto const age =
          (slot.first == erased_key)
            ? static_cast<size_type>(slot.second)
            : cuco::detail::robin_hood::probe_distance<bs>(
                scheme, slot.first, static_cast<size_type>(g * stride + s), extent);
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
        auto const age =
          (slot.first == erased_key)
            ? static_cast<size_type>(slot.second)
            : cuco::detail::robin_hood::probe_distance<bs>(
                scheme, slot.first, static_cast<size_type>(pg * stride + s), extent);
        if (occupied_p == 0 || age < min_age_p) { min_age_p = age; }
        ++occupied_p;
      }
    }

    if (max_age_g >= 1 && occupied_p < stride) { atomicAdd(violations, 1); }        // (1)
    if (occupied_p > 0 && min_age_p + 1 < max_age_g) { atomicAdd(violations, 1); }  // (2)
  }
}

// Asserts that a populated Robin Hood `map` satisfies the per-bucket layout invariant above. The
// `find` ref reaches the live storage pointer, the probing scheme, and the sentinels that the
// kernel needs.
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

// Walks each lane's probe iterator and records, at every step, the probe distance reported for the
// slot that lane is visiting. Because the resident under test is `key` itself, the slot visited at
// step `i` is at probe distance `i` from `key`'s home -- for every lane. Recording one column per
// lane lets the host check both that `probe_distance` inverts `make_iterator` and that it strips
// the per-lane intra-stride offset (so all lanes at a given step agree).
template <int32_t BucketSize, class ProbingScheme, class Key, class Extent, class OutputIt>
__global__ void generate_cg_probe_distance_sequence(Key key,
                                                    Extent upper_bound,
                                                    std::size_t seq_length,
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

    for (std::size_t i = 0; i < seq_length; ++i) {
      out_seq[i * cg_size + tile.thread_rank()] =
        cuco::detail::robin_hood::probe_distance<BucketSize>(probing_scheme, key, *iter, upper_bound);
      ++iter;
    }
  }
}

}  // namespace

TEMPLATE_TEST_CASE_SIG(
  "static_map robin_hood probe_distance inverts make_iterator",
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
  using probe = cuco::linear_probing<CGSize, cuco::default_hash_function<Key>>;

  // A deliberately small capacity, so the probe sequence wraps around the table within
  // `seq_length` steps -- this exercises the modular-subtraction (wrap) path in `probe_distance`.
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
  CUCO_CUDA_TRY(cudaDeviceSynchronize());

  // Under wrap, the slot visited at step `i` sits at probe distance `i mod num_buckets`.
  for (std::size_t i = 0; i < seq_length; ++i) {
    for (std::int32_t r = 0; r < CGSize; ++r) {
      REQUIRE(distances[i * CGSize + r] == i % num_buckets);
    }
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_map robin_hood high-load-factor invariant",
  "",
  ((typename Key, typename Value, int CGSize, int BucketSize), Key, Value, CGSize, BucketSize),
  (int32_t, int32_t, 1, 1),
  (int32_t, int32_t, 1, 2)
#if defined(CUCO_HAS_128BIT_ATOMICS)
  ,
  (int64_t, int64_t, 1, 1),
  (int64_t, int64_t, 1, 2)
#endif
)
{
  // Robin Hood is most meaningfully exercised when the table is nearly full: the displacement
  // chains are long and the layout invariant becomes load-bearing. Size the table for ~95% load.
  using size_type = std::int32_t;

  constexpr size_type num_keys = 100'000;

  using extent_type = cuco::extent<size_type>;
  using probe       = cuco::linear_probing<CGSize, cuco::murmurhash3_32<Key>>;
  using map_type    = cuco::static_map<Key,
                                       Value,
                                       extent_type,
                                       cuda::thread_scope_device,
                                       cuda::std::equal_to<Key>,
                                       probe,
                                       cuco::cuda_allocator<cuda::std::byte>,
                                       cuco::storage<BucketSize>>;

  // High load factor: size the table for ~95% occupancy so Robin Hood is exercised near-full.
  auto map = map_type{extent_type{num_keys},
                      0.95,
                      cuco::empty_key<Key>{-1},
                      cuco::empty_value<Value>{-1}};

  auto keys_begin  = cuda::counting_iterator<Key>{0};
  auto pairs_begin = cuda::make_transform_iterator(
    cuda::make_counting_iterator<size_type>(0),
    cuda::proclaim_return_type<cuco::pair<Key, Value>>(
      [] __device__(auto i) { return cuco::pair<Key, Value>{i, i}; }));

  map.insert(pairs_begin, pairs_begin + num_keys);
  REQUIRE(map.size() == num_keys);

  // The hand-built Robin Hood layout must be structurally valid after a near-full insert.
  check_robin_hood_invariant(map);

  // Every inserted unique key must be found and contained.
  thrust::device_vector<bool> d_contained(num_keys);
  map.contains(keys_begin, keys_begin + num_keys, d_contained.begin());
  REQUIRE(cuco::test::all_of(d_contained.begin(), d_contained.end(), cuda::std::identity{}));

  thrust::device_vector<Value> d_values(num_keys);
  map.find(keys_begin, keys_begin + num_keys, d_values.begin());
  auto zip = thrust::make_zip_iterator(cuda::std::tuple{d_values.begin(), keys_begin});
  REQUIRE(cuco::test::all_of(
    zip, zip + num_keys, cuda::proclaim_return_type<bool>([] __device__(auto const& p) {
      return cuda::std::get<0>(p) == cuda::std::get<1>(p);
    })));
}
