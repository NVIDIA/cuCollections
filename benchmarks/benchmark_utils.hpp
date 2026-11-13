/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/__config>
#include <cuco/detail/error.hpp>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/functional>
#include <cuda/iterator>

#include <nv/target>

namespace cuco::benchmark {

template <typename Dist>
auto dist_from_state(nvbench::state const& state)
{
  if constexpr (std::is_same_v<Dist, cuco::utility::distribution::unique>) {
    return Dist{};
  } else if constexpr (std::is_same_v<Dist, cuco::utility::distribution::uniform>) {
    auto const multiplicity = state.get_int64("Multiplicity");
    return Dist{multiplicity};
  } else if constexpr (std::is_same_v<Dist, cuco::utility::distribution::gaussian>) {
    auto const skew = state.get_float64("Skew");
    return Dist{skew};
  } else {
    CUCO_FAIL("Unexpected distribution type");
  }
}

template <typename T, typename NewType>
struct rebind_unary_hasher;

template <template <typename> class Template, typename OldType, typename NewType>
struct rebind_unary_hasher<Template<OldType>, NewType> {
  using type = Template<NewType>;
};

template <typename T, typename NewType>
struct rebind_hasher : rebind_unary_hasher<T, NewType> {};

template <typename OldType, cuda::hash_algorithm Algorithm, typename NewType>
struct rebind_hasher<cuda::hash<OldType, Algorithm>, NewType> {
  using type = cuda::hash<NewType, Algorithm>;
};

template <typename T, typename NewType>
using rebind_hasher_t = typename rebind_hasher<T, NewType>::type;

template <class OutputIt>
struct lazy_discard {
  OutputIt it;

  using index_type = typename cuda::std::iterator_traits<OutputIt>::difference_type;
  using value_type = typename cuda::std::iterator_traits<OutputIt>::value_type;

  __device__ void device_dispatch(index_type index, value_type const& value) const
  {
    // pick some predicate that is always false, but depends on the runtime value
    if (threadIdx.x > 2025 + *reinterpret_cast<char const*>(&value) + static_cast<int>(index)) {
      *(it + index) = value;
    }
  }
  __host__ __device__ void operator()(index_type index, value_type const& value) const
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 this->device_dispatch(index, value);)  // we don't care about the host path for now
  }
};

/**
 * @brief An output iterator similar to `cuda::discard_iterator` but prevents the write from being
 * optimized out by the compiler.
 */
template <class OutputIt>
auto make_lazy_discard_iterator(OutputIt it)
{
  return cuda::make_tabulate_output_iterator(lazy_discard<OutputIt>{it});
}

}  // namespace cuco::benchmark

NVBENCH_DECLARE_TYPE_STRINGS(cuco::utility::distribution::unique, "UNIQUE", "distribution::unique");
NVBENCH_DECLARE_TYPE_STRINGS(cuco::utility::distribution::uniform,
                             "UNIFORM",
                             "distribution::uniform");
NVBENCH_DECLARE_TYPE_STRINGS(cuco::utility::distribution::gaussian,
                             "GAUSSIAN",
                             "distribution::gaussian");

namespace nvbench {

template <cuda::hash_algorithm Algorithm>
struct type_strings<cuda::hash<char, Algorithm>> {
  static std::string input_string()
  {
    switch (Algorithm) {
      case cuda::hash_algorithm::xxhash_32: return "xxhash_32";
      case cuda::hash_algorithm::xxhash_64: return "xxhash_64";
      case cuda::hash_algorithm::murmurhash3_32: return "murmurhash3_32";
      case cuda::hash_algorithm::murmurhash3_x86_128: return "murmurhash3_x86_128";
      case cuda::hash_algorithm::murmurhash3_x64_128: return "murmurhash3_x64_128";
    }
    CUCO_FAIL("Unexpected hash algorithm");
  }

  static std::string description()
  {
    return "cuda::hash<char, cuda::hash_algorithm::" + input_string() + ">";
  }
};

}  // namespace nvbench

#if defined(CUCO_HAS_128BIT_ATOMICS)
NVBENCH_DECLARE_TYPE_STRINGS(__int128_t, "I128", "__int128_t");
#endif
