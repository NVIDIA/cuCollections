/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark_defaults.hpp>

#include <cuco/detail/utility/cuda.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/functional>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <thrust/device_vector.h>

#include <cstdint>

// repeat hash computation n times
static constexpr auto n_repeats = 100;

template <int32_t Words>
struct large_key {
  constexpr __host__ __device__ large_key(int32_t seed) noexcept
  {
    for (int32_t i = 0; i < Words; ++i) {
      data_[i] = seed;
    }
  }

 private:
  int32_t data_[Words];
};

template <typename Hash>
struct hash_key {
  using type = typename Hash::argument_type;
};

template <typename Key, cuda::hash_algorithm Algorithm>
struct hash_key<cuda::hash<Key, Algorithm>> {
  using type = Key;
};

template <typename Hash>
using hash_key_t = typename hash_key<Hash>::type;

template <typename Hash>
using hash_result_t = cuda::std::invoke_result_t<Hash, hash_key_t<Hash>>;

template <int32_t BlockSize, typename Hasher, typename OutputIt>
__global__ void hash_bench_kernel(Hasher hash,
                                  cuco::detail::index_type n,
                                  OutputIt out,
                                  bool materialize_result)
{
  cuco::detail::index_type const gid         = BlockSize * blockIdx.x + threadIdx.x;
  cuco::detail::index_type const loop_stride = gridDim.x * BlockSize;
  cuco::detail::index_type idx               = gid;
  hash_result_t<Hasher> agg                  = {};

  while (idx < n) {
    hash_key_t<Hasher> key(idx);
    for (int32_t i = 0; i < n_repeats; ++i) {  // execute hash func n times
      agg += hash(key);
    }
    idx += loop_stride;
  }

  if (materialize_result) { out[gid] = agg; }
}

/**
 * @brief A benchmark evaluating performance of various hash functions
 */
template <typename Hash>
void hash_eval(nvbench::state& state, nvbench::type_list<Hash>)
{
  bool const materialize_result = false;
  constexpr auto block_size     = 128;
  auto const num_keys           = state.get_int64("NumInputs");
  auto const grid_size          = (num_keys + block_size * 16 - 1) / block_size * 16;

  thrust::device_vector<hash_result_t<Hash>> hash_values((materialize_result) ? num_keys : 1);

  state.add_element_count(num_keys);

  state.exec([&](nvbench::launch& launch) {
    hash_bench_kernel<block_size><<<grid_size, block_size, 0, launch.get_stream()>>>(
      Hash{}, num_keys, hash_values.begin(), materialize_result);
  });
}

template <int32_t BlockSize, typename Hasher, typename InputIt, typename OutputIt>
__global__ void string_hash_bench_kernel(
  Hasher hash, InputIt in, cuco::detail::index_type n, OutputIt out, bool materialize_result)
{
  cuco::detail::index_type const gid         = BlockSize * blockIdx.x + threadIdx.x;
  cuco::detail::index_type const loop_stride = gridDim.x * BlockSize;
  cuco::detail::index_type idx               = gid;
  hash_result_t<Hasher> agg                  = {};

  while (idx < n) {
    auto const key = thrust::raw_reference_cast(*(in + idx));
    for (int32_t i = 0; i < n_repeats; ++i) {  // execute hash func n times
      agg += hash(key);
    }
    idx += loop_stride;
  }

  if (materialize_result) { out[gid] = agg; }
}

/**
 * @brief A benchmark evaluating performance of various hash functions on random strings with
 * variable length and alignment
 */
template <typename Hash>
void string_hash_eval(nvbench::state& state, nvbench::type_list<Hash>)
{
  static_assert(cuda::std::is_same_v<hash_key_t<Hash>, cuda::std::byte>,
                "Argument type must be cuda::std::byte");

  bool const materialize_result = false;
  constexpr auto block_size     = 128;
  auto const num_keys           = state.get_int64("NumInputs");
  auto const min_length         = state.get_int64("MinLength");
  auto const max_length         = state.get_int64("MaxLength");
  auto const grid_size          = (num_keys + block_size * 16 - 1) / block_size * 16;

  if (min_length > max_length) {
    state.skip("MinLength > MaxLength");
    return;
  }

  // auto const [keys, storage] = ... (can't capture structured bindings into lambdas in C++17)
  auto const sequences =
    cuco::utility::generate_random_byte_sequences(num_keys, min_length, max_length);
  auto const& keys = sequences.first;
  // auto const& storage = sequences.second;

  thrust::device_vector<hash_result_t<Hash>> hash_values((materialize_result) ? num_keys : 1);

  state.add_element_count(num_keys);
  // state.add_global_memory_reads<cuda::std::byte>(storage.size() * n_repeats);

  state.exec([&](nvbench::launch& launch) {
    string_hash_bench_kernel<block_size><<<grid_size, block_size, 0, launch.get_stream()>>>(
      Hash{}, keys.begin(), num_keys, hash_values.begin(), materialize_result);
  });
}

using hash_types =
  nvbench::type_list<cuda::hash<nvbench::int32_t, cuda::hash_algorithm::murmurhash3_32>,
                     cuda::hash<nvbench::int64_t, cuda::hash_algorithm::murmurhash3_32>,
                     cuda::hash<large_key<32>, cuda::hash_algorithm::murmurhash3_32>,  // 32*4bytes
                     cuda::hash<nvbench::int32_t, cuda::hash_algorithm::xxhash_32>,
                     cuda::hash<nvbench::int64_t, cuda::hash_algorithm::xxhash_32>,
                     cuda::hash<large_key<32>, cuda::hash_algorithm::xxhash_32>,
                     cuda::hash<nvbench::int32_t, cuda::hash_algorithm::xxhash_64>,
                     cuda::hash<nvbench::int64_t, cuda::hash_algorithm::xxhash_64>,
                     cuda::hash<large_key<32>, cuda::hash_algorithm::xxhash_64>
#if defined(CUCO_HAS_INT128)
                     ,
                     cuda::hash<nvbench::int32_t, cuda::hash_algorithm::murmurhash3_x86_128>,
                     cuda::hash<nvbench::int64_t, cuda::hash_algorithm::murmurhash3_x86_128>,
                     cuda::hash<large_key<32>, cuda::hash_algorithm::murmurhash3_x86_128>,
                     cuda::hash<nvbench::int32_t, cuda::hash_algorithm::murmurhash3_x64_128>,
                     cuda::hash<nvbench::int64_t, cuda::hash_algorithm::murmurhash3_x64_128>,
                     cuda::hash<large_key<32>, cuda::hash_algorithm::murmurhash3_x64_128>
#endif
                     >;

NVBENCH_BENCH_TYPES(hash_eval, NVBENCH_TYPE_AXES(hash_types))
  .set_name("hash_function_eval")
  .set_type_axes_names({"Hash"})
  .add_int64_axis("NumInputs", {cuco::benchmark::defaults::N * 10});

using string_hash_types =
  nvbench::type_list<cuda::hash<cuda::std::byte, cuda::hash_algorithm::murmurhash3_32>,
                     cuda::hash<cuda::std::byte, cuda::hash_algorithm::xxhash_32>,
                     cuda::hash<cuda::std::byte, cuda::hash_algorithm::xxhash_64>
#if defined(CUCO_HAS_INT128)
                     ,
                     cuda::hash<cuda::std::byte, cuda::hash_algorithm::murmurhash3_x86_128>,
                     cuda::hash<cuda::std::byte, cuda::hash_algorithm::murmurhash3_x64_128>
#endif
                     >;

NVBENCH_BENCH_TYPES(string_hash_eval, NVBENCH_TYPE_AXES(string_hash_types))
  .set_name("string_hash_function_eval")
  .set_type_axes_names({"Hash"})
  .add_int64_axis("NumInputs", {cuco::benchmark::defaults::N / 4})
  .add_int64_axis("MinLength", {1, 4})
  .add_int64_axis("MaxLength", {4, 32, 64});
