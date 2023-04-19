/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <defaults.hpp>
#include <utils.hpp>

#include <cuco/detail/utils.hpp>
#include <cuco/hash_functions.cuh>

#include <nvbench/nvbench.cuh>

#include <cstdint>

using namespace cuco::benchmark;
using namespace cuco::utility;

template <int32_t Words>
struct large_key {
  constexpr __host__ __device__ large_key(int32_t seed) noexcept
  {
#pragma unroll Words
    for (int32_t i = 0; i < Words; ++i) {
      data_[i] = seed;
    }
  }

 private:
  int32_t data_[Words];
};

template <int32_t BlockSize, typename Hasher>
__global__ void hash_bench_kernel(Hasher hash, cuco::detail::index_type n)
{
  cuco::detail::index_type const loop_stride = gridDim.x * BlockSize;
  cuco::detail::index_type idx               = BlockSize * blockIdx.x + threadIdx.x;

  while (idx < n) {
    volatile auto hash_value = hash(idx);
    idx += loop_stride;
  }
}

/**
 * @brief A benchmark evaluating performance of various hash functions
 */
template <typename Hash>
void hash_eval(nvbench::state& state, nvbench::type_list<Hash>)
{
  constexpr auto block_size = 128;
  auto const num_keys       = state.get_int64_or_default("NumInputs", defaults::N);
  auto const grid_size =
    state.get_int64_or_default("GridSize", SDIV(num_keys + 4 * block_size, 4 * block_size));

  state.add_element_count(num_keys);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    hash_bench_kernel<block_size>
      <<<grid_size, block_size, 0, launch.get_stream()>>>(Hash{}, num_keys);
  });
}

NVBENCH_BENCH_TYPES(
  hash_eval,
  NVBENCH_TYPE_AXES(nvbench::type_list<cuco::murmurhash3_32<nvbench::int32_t>,
                                       cuco::murmurhash3_32<nvbench::int64_t>,
                                       cuco::murmurhash3_32<large_key<64>>,  // 64*4bytes
                                       cuco::xxhash_64<nvbench::int32_t>,
                                       cuco::xxhash_64<nvbench::int64_t>,
                                       cuco::xxhash_64<large_key<64>>>))
  .set_name("hash_function_eval")
  .set_type_axes_names({"Hash"})
  .set_max_noise(defaults::MAX_NOISE);