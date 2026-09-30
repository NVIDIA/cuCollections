/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/bucket_storage.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/operator.hpp>
#include <cuco/pair.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/static_map.cuh>
#include <cuco/static_set.cuh>

#include <cuda/atomic>
#include <cuda/functional>

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>

namespace {

using key_type    = std::int32_t;
using mapped_type = std::int32_t;
using probe_type  = cuco::linear_probing<1, cuco::default_hash_function<key_type>>;
using map_type    = cuco::static_map<key_type,
                                     mapped_type,
                                     cuco::extent<std::size_t>,
                                     cuda::thread_scope_device,
                                     cuda::std::equal_to<key_type>,
                                     probe_type>;

struct zero_hash {
  template <typename T>
  __device__ std::uint32_t operator()(T const&) const noexcept
  {
    return 0;
  }
};

using hetero_probe_type = cuco::linear_probing<1, zero_hash>;
using hetero_map_type   = cuco::static_map<key_type,
                                           mapped_type,
                                           cuco::extent<std::size_t>,
                                           cuda::thread_scope_device,
                                           cuda::std::equal_to<>,
                                           hetero_probe_type>;
using tiny_set_type     = cuco::static_set<key_type,
                                           cuco::extent<std::size_t>,
                                           cuda::thread_scope_device,
                                           cuda::std::equal_to<key_type>,
                                           probe_type>;
template <typename Ref>
__global__ void insert_one(Ref ref, key_type key, mapped_type value)
{
  ref.insert(cuco::pair{key, value});
}

template <typename Ref, typename ProbeKey>
__global__ void probe_one(Ref ref, ProbeKey key, bool* result)
{
  auto const found = ref.contains(key);
  if (result != nullptr) { *result = found; }
}

template <typename Ref>
__global__ void fill_map(Ref ref)
{
  for (std::size_t i = 0; i < ref.capacity(); ++i) {
    ref.insert(cuco::pair{static_cast<key_type>(i), static_cast<mapped_type>(i)});
  }
}

template <typename Ref>
__global__ void overflow_map(Ref ref)
{
  for (std::size_t i = 0; i <= ref.capacity(); ++i) {
    ref.insert(cuco::pair{static_cast<key_type>(i), static_cast<mapped_type>(i)});
  }
}
template <typename StorageRef>
__global__ void access_oob(StorageRef ref)
{
  static_cast<void>(ref[ref.capacity()]);
}

template <typename Ref>
__global__ void retrieve_from_empty_set(Ref ref,
                                        key_type* input,
                                        key_type* stencil,
                                        key_type* output_probe,
                                        key_type* output_match,
                                        cuda::atomic<int, cuda::thread_scope_device>* counter)
{
  auto const block = cooperative_groups::this_thread_block();
  auto const pred  = [] __device__(key_type) { return true; };

  ref.template retrieve_if<32>(
    block, input, input + 1, stencil, pred, output_probe, output_match, *counter);
}

map_type* make_map()
{
  return new map_type{cuco::extent<std::size_t>{8},
                      cuco::empty_key<key_type>{-1},
                      cuco::empty_value<mapped_type>{-1},
                      cuda::std::equal_to<key_type>{},
                      probe_type{}};
}
hetero_map_type* make_hetero_map()
{
  return new hetero_map_type{cuco::extent<std::size_t>{8},
                             cuco::empty_key<key_type>{-1},
                             cuco::empty_value<mapped_type>{-1},
                             cuda::std::equal_to<>{},
                             hetero_probe_type{zero_hash{}}};
}

void require_device_assert() { REQUIRE(cudaDeviceSynchronize() == cudaErrorAssert); }

void require_cuda_success() { REQUIRE(cudaDeviceSynchronize() == cudaSuccess); }

}  // namespace

TEST_CASE("CUCO_DEBUG rejects empty key insertion")
{
  auto* map = make_map();
  insert_one<<<1, 1>>>(map->ref(cuco::op::insert), -1, 1);
  require_device_assert();
}

TEST_CASE("CUCO_DEBUG rejects empty payload insertion")
{
  auto* map = make_map();
  insert_one<<<1, 1>>>(map->ref(cuco::op::insert), 1, -1);
  require_device_assert();
}

TEST_CASE("CUCO_DEBUG rejects probing for an empty key sentinel")
{
  auto* map = make_map();
  probe_one<<<1, 1>>>(map->ref(cuco::op::contains), -1, nullptr);
  require_device_assert();
}

TEST_CASE("CUCO_DEBUG detects insertion probe exhaustion")
{
  auto* map = make_map();
  overflow_map<<<1, 1>>>(map->ref(cuco::op::insert));
  require_device_assert();
}

TEST_CASE("CUCO_DEBUG detects out-of-bounds bucket access")
{
  using storage_ref_type = cuco::bucket_storage_ref<key_type, 2, cuco::extent<std::size_t>>;
  auto const ref         = storage_ref_type{cuco::extent<std::size_t>{2}, nullptr};
  access_oob<<<1, 1>>>(ref);
  require_device_assert();
}
TEST_CASE("CUCO_DEBUG permits a full-table lookup miss")
{
  auto* map = make_map();
  fill_map<<<1, 1>>>(map->ref(cuco::op::insert));
  require_cuda_success();

  bool* found{};
  REQUIRE(cudaMallocManaged(&found, sizeof(bool)) == cudaSuccess);
  *found = true;

  probe_one<<<1, 1>>>(map->ref(cuco::op::contains), key_type{123456}, found);
  require_cuda_success();
  REQUIRE_FALSE(*found);

  REQUIRE(cudaFree(found) == cudaSuccess);
}

TEST_CASE("CUCO_DEBUG does not narrow heterogeneous probe keys")
{
  auto* map = make_hetero_map();

  bool* found{};
  REQUIRE(cudaMallocManaged(&found, sizeof(bool)) == cudaSuccess);
  *found = true;

  probe_one<<<1, 1>>>(map->ref(cuco::op::contains), std::int64_t{4294967295LL}, found);
  require_cuda_success();
  REQUIRE_FALSE(*found);

  REQUIRE(cudaFree(found) == cudaSuccess);
}

TEST_CASE("CUCO_DEBUG retrieval may finish before probe wraparound")
{
  auto* set = new tiny_set_type{cuco::extent<std::size_t>{1},
                                cuco::empty_key<key_type>{-1},
                                cuda::std::equal_to<key_type>{},
                                probe_type{}};

  key_type* input{};
  key_type* stencil{};
  key_type* output_probe{};
  key_type* output_match{};
  cuda::atomic<int, cuda::thread_scope_device>* counter{};

  REQUIRE(cudaMalloc(&input, sizeof(key_type)) == cudaSuccess);
  REQUIRE(cudaMalloc(&stencil, sizeof(key_type)) == cudaSuccess);
  REQUIRE(cudaMalloc(&output_probe, sizeof(key_type)) == cudaSuccess);
  REQUIRE(cudaMalloc(&output_match, sizeof(key_type)) == cudaSuccess);
  REQUIRE(cudaMalloc(&counter, sizeof(*counter)) == cudaSuccess);

  key_type const input_value   = 7;
  key_type const stencil_value = 1;
  REQUIRE(cudaMemcpy(input, &input_value, sizeof(key_type), cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(stencil, &stencil_value, sizeof(key_type), cudaMemcpyHostToDevice) ==
          cudaSuccess);
  REQUIRE(cudaMemset(counter, 0, sizeof(*counter)) == cudaSuccess);

  retrieve_from_empty_set<<<1, 32>>>(
    set->ref(cuco::op::retrieve), input, stencil, output_probe, output_match, counter);
  require_cuda_success();

  int count{-1};
  REQUIRE(cudaMemcpy(&count, counter, sizeof(count), cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(count == 0);

  REQUIRE(cudaFree(input) == cudaSuccess);
  REQUIRE(cudaFree(stencil) == cudaSuccess);
  REQUIRE(cudaFree(output_probe) == cudaSuccess);
  REQUIRE(cudaFree(output_match) == cudaSuccess);
  REQUIRE(cudaFree(counter) == cudaSuccess);
}
