
#include <cstdint>
#include <cuco/hash_functions.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <random>

#define cudaCheckError(call)                                                              \
  {                                                                                       \
    cudaError_t err = call;                                                               \
    if (err != cudaSuccess) {                                                             \
      std::cerr << "CUDA error in file '" << __FILE__ << "' in line " << __LINE__ << ": " \
                << cudaGetErrorString(err) << "." << std::endl;                           \
      exit(EXIT_FAILURE);                                                                 \
    }                                                                                     \
  }

template <typename Hasher>
void init_keys(cuco::detail::index_type n, typename Hasher::argument_type* keys)
{
  for (cuco::detail::index_type idx = 0; idx < n; idx++) {
    keys[idx] = (typename Hasher::argument_type)idx;
  }
}

template <int32_t BlockSize, typename Hasher>
__global__ void hash_keys(Hasher hash,
                          cuco::detail::index_type n,
                          typename Hasher::argument_type* in,
                          typename Hasher::result_type* hashes,
                          bool* passed)
{
  cuco::detail::index_type const gid         = BlockSize * blockIdx.x + threadIdx.x;
  cuco::detail::index_type const loop_stride = gridDim.x * BlockSize;
  cuco::detail::index_type idx               = gid;

  while (idx < n) {
    typename Hasher::argument_type key = in[idx];
    hashes[idx]                        = hash(key);
    if (hashes[idx] != in[idx]) *passed = false;
    idx += loop_stride;
  }
}

/**
 * @brief A benchmark evaluating performance of various hash functions
 */
template <typename Hash>
void hash_eval()
{
  constexpr int32_t block_size               = 128;
  typename cuco::detail::index_type num_keys = 100000;
  auto const grid_size                       = (num_keys + block_size * 16 - 1) / block_size * 16;

  typename Hash::result_type* hashes;
  typename Hash::argument_type* keys;
  bool* passed;
  cudaCheckError(cudaMallocManaged(&hashes, sizeof(typename Hash::result_type) * num_keys));
  cudaCheckError(cudaMallocManaged(&keys, sizeof(typename Hash::argument_type) * num_keys));
  cudaCheckError(cudaMallocManaged(&passed, sizeof(bool)));
  *passed = true;

  init_keys<Hash>(num_keys, keys);

  hash_keys<block_size, Hash><<<grid_size, block_size, 0>>>(Hash{}, num_keys, keys, hashes, passed);

  cudaCheckError(cudaDeviceSynchronize());

  int samples = 20;
  std::cout << samples << " samples of input and output" << std::endl;
  std::random_device rd;   // Obtain a random number from hardware
  std::mt19937 gen(rd());  // Seed the generator
  std::uniform_int_distribution<> distr(0, num_keys);
  for (int i = 0; i < 20; i++) {
    cuco::detail::index_type sample = distr(gen);
    std::cout << "idx: " << sample << ",\tIn: " << keys[sample] << ",\tOut: " << hashes[sample]
              << std::endl;
  }

  std::cout << "Identity Test: " << (*passed ? "PASSED" : "FAILED") << std::endl;

  cudaCheckError(cudaFree(hashes));
  cudaCheckError(cudaFree(keys));
  cudaCheckError(cudaFree(passed));
}

int main()
{
  hash_eval<cuco::identityhash_32<std::uint32_t>>();
  hash_eval<cuco::identityhash_64<std::uint64_t>>();
}
