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

#include <cuda/functional>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

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

template <typename Ref>
__global__ void insert_one(Ref ref, key_type key, mapped_type value)
{
  ref.insert(cuco::pair{key, value});
}

template <typename Ref>
__global__ void probe_one(Ref ref, key_type key)
{
  static_cast<void>(ref.contains(key));
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

map_type* make_map()
{
  return new map_type{cuco::extent<std::size_t>{8},
                      cuco::empty_key<key_type>{-1},
                      cuco::empty_value<mapped_type>{-1},
                      cuda::std::equal_to<key_type>{},
                      probe_type{}};
}

int expect_device_assert()
{
  auto const status = cudaDeviceSynchronize();
  return status == cudaErrorAssert ? 0 : 1;
}

int test_empty_key()
{
  auto* map = make_map();
  insert_one<<<1, 1>>>(map->ref(cuco::op::insert), -1, 1);
  return expect_device_assert();
}

int test_empty_payload()
{
  auto* map = make_map();
  insert_one<<<1, 1>>>(map->ref(cuco::op::insert), 1, -1);
  return expect_device_assert();
}

int test_probe_empty_key()
{
  auto* map = make_map();
  probe_one<<<1, 1>>>(map->ref(cuco::op::contains), -1);
  return expect_device_assert();
}

int test_probe_exhaustion()
{
  auto* map = make_map();
  overflow_map<<<1, 1>>>(map->ref(cuco::op::insert));
  return expect_device_assert();
}

int test_storage_bounds()
{
  using storage_ref_type = cuco::bucket_storage_ref<key_type, 2, cuco::extent<std::size_t>>;
  auto const ref         = storage_ref_type{cuco::extent<std::size_t>{2}, nullptr};
  access_oob<<<1, 1>>>(ref);
  return expect_device_assert();
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc != 2) { return 2; }
  if (std::strcmp(argv[1], "empty-key") == 0) { return test_empty_key(); }
  if (std::strcmp(argv[1], "empty-payload") == 0) { return test_empty_payload(); }
  if (std::strcmp(argv[1], "probe-empty-key") == 0) { return test_probe_empty_key(); }
  if (std::strcmp(argv[1], "probe-exhaustion") == 0) { return test_probe_exhaustion(); }
  if (std::strcmp(argv[1], "storage-bounds") == 0) { return test_storage_bounds(); }
  return 2;
}
