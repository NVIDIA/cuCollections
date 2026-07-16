/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/static_set.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/device_vector.h>

#include <catch2/catch_test_macros.hpp>

using T    = int32_t;
using Hash = uint32_t;
using Key  = cuco::pair<Hash, T>;

struct hasher {
  __device__ Hash operator()(Key const& k) const { return k.first; }
};

struct always_not_equal {
  __device__ constexpr bool operator()(Key const&, Key const&) const noexcept
  {
    // All build table keys are distinct thus `false` no matter what
    return false;
  }
};

class build_fn {
 public:
  __device__ __forceinline__ auto operator()(T i) const noexcept { return cuco::pair{_hash(i), i}; }

 private:
  cuco::default_hash_function<T> _hash{};
};

// This test exercise is designed to replicate a Spark runtime failure scenario
// https://github.com/NVIDIA/spark-rapids/issues/12586 and
// https://github.com/rapidsai/cudf/issues/18587
// that is not addressed by the current test suite. It will result in a runtime
// crash if the CCCL atomic storage is not managed correctly.
TEST_CASE("atomic_storage_test", "")
{
  using probe = cuco::linear_probing<1, hasher>;

  auto const num_keys = 100'000;

  auto set = cuco::static_set{cuco::extent<int>{num_keys},
                              0.5,
                              cuco::empty_key<Key>{Key{std::numeric_limits<Hash>::max(), -1}},
                              always_not_equal{},
                              probe{},
                              {},
                              cuco::storage<1>{}};

  auto keys_begin = cuda::make_transform_iterator(cuda::counting_iterator{0},
                                                  cuda::proclaim_return_type<Key>(build_fn{}));

  set.insert_async(keys_begin, keys_begin + num_keys);
  auto const count = set.size();

  REQUIRE(count == num_keys);
}
