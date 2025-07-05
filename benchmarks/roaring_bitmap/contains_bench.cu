/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <benchmark_defaults.hpp>
#include <benchmark_utils.hpp>

#include <cuco/roaring_bitmap.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

void roaring_bitmap_contains(nvbench::state& state)
{
  namespace fs = std::filesystem;

  // Get the path of the current source file
  fs::path source_file_path = __FILE__;
  fs::path source_dir       = source_file_path.parent_path();

  fs::path path      = source_dir / "../../examples/roaring_bitmap/bitmapwithoutruns.bin";
  fs::path full_path = path.lexically_normal();

  std::ifstream file(full_path, std::ios::binary);
  if (!file.is_open()) { state.skip("Failed to open bitmap file"); }

  // Get file size
  file.seekg(0, std::ios::end);
  std::streamsize file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  char* buffer;
  CUCO_CUDA_TRY(cudaMallocHost(&buffer, file_size));

  file.read(buffer, file_size);
  file.close();

  cuda::std::span<cuda::std::byte const> bitmap(reinterpret_cast<cuda::std::byte const*>(buffer),
                                                file_size);
  cuco::roaring_bitmap<cuda::std::uint32_t> roaring_bitmap(bitmap);

  std::vector<cuda::std::uint32_t> keys;
  for (cuda::std::uint32_t k = 0; k < 100000; k += 1000) {
    keys.push_back(k);
  }
  for (cuda::std::uint32_t k = 100000; k < 200000; ++k) {
    keys.push_back(3 * k);
  }
  for (cuda::std::uint32_t k = 700000; k < 800000; ++k) {
    keys.push_back(k);
  }

  // multiply the keys for more accurate benchmark numbers
  for (int i = 0; i < 13; i++) {
    keys.insert(keys.end(), keys.begin(), keys.end());
  }

  thrust::device_vector<cuda::std::uint32_t> keys_d(keys.begin(), keys.end());
  thrust::device_vector<bool> contained(keys.size(), false);

  state.add_element_count(keys.size());
  state.add_global_memory_reads<cuda::std::uint32_t>(keys.size(), "InputSize");

  state.exec([&](nvbench::launch& launch) {
    roaring_bitmap.contains_async(
      keys_d.begin(), keys_d.end(), contained.begin(), {launch.get_stream()});
  });

  CUCO_CUDA_TRY(cudaFreeHost(buffer));
}

NVBENCH_BENCH(roaring_bitmap_contains)
  .set_name("roaring_bitmap_contains")
  .set_max_noise(cuco::benchmark::defaults::MAX_NOISE);