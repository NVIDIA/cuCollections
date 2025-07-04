#include <cuco/detail/error.hpp>
#include <cuco/roaring_bitmap.cuh>

#include <cuda/std/span>
#include <thrust/logical.h>
#include <thrust/universal_vector.h>

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <bitmap_file_path>" << std::endl;
    return -1;
  }

  // Open file
  std::ifstream file(argv[1], std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open " << argv[1] << std::endl;
    return -1;
  }

  // Get file size
  file.seekg(0, std::ios::end);
  std::streamsize file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  // Allocate pinned host memory using cudaMallocHost
  char* buffer;
  CUCO_CUDA_TRY(cudaMallocHost(&buffer, file_size));

  // Read file into memory
  file.read(buffer, file_size);
  file.close();

  cuda::std::span<cuda::std::byte const> bitmap(reinterpret_cast<cuda::std::byte const*>(buffer),
                                                file_size);
  cuco::roaring_bitmap<cuda::std::uint32_t> roaring_bitmap(bitmap);

  std::vector<cuda::std::uint32_t> keys;
  for (cuda::std::uint32_t k = 0; k < 100000; k += 1000) {
    keys.push_back(k);
  }
  for (int k = 100000; k < 200000; ++k) {
    keys.push_back(3 * k);
  }
  for (int k = 700000; k < 800000; ++k) {
    keys.push_back(k);
  }

  thrust::universal_vector<cuda::std::uint32_t> keys_d(keys.begin(), keys.end());
  thrust::universal_vector<bool> contained(keys.size(), false);

  roaring_bitmap.contains(keys_d.begin(), keys_d.end(), contained.begin());

  for (size_t i = 0; i < keys.size(); i++) {
    if (not contained[i]) {
      std::cout << "Error: " << keys_d[i] << " is not contained" << std::endl;
    }
  }

  // check if all elements are contained
  bool all_contained = thrust::all_of(contained.begin(), contained.end(), ::cuda::std::identity{});
  std::cout << "all_contained: " << all_contained << std::endl;

  // Free the allocated memory
  CUCO_CUDA_TRY(cudaFreeHost(buffer));

  return 0;
}