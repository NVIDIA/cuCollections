#ifndef CPU_INTERFACE
#define CPU_INTERFACE

#include <memory>
#include <vector>

#include <thrust/device_vector.h>

#include <hash/concurrent_unordered_map.cuh>

// TODO: Move to utils
#define DIV_AND_CEIL(x, y) (x + y - 1) / y
#define CHK_CUDA(expression)                                                  \
  {                                                                           \
    cudaError_t status = (expression);                                        \
    if (status != cudaSuccess) {                                              \
      std::cerr << "Error in file: " << __FILE__ << ", on line: " << __LINE__ \
                << ": " << cudaGetErrorString(status) << std::endl;           \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  }

namespace cuCollections {
namespace details {

template <typename Map, typename KeyType, typename ValType>
__global__ void insert(Map map, const size_t size, const KeyType* keys,
                       const ValType* values) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  while (idx < size) {
    map.insert(thrust::pair<KeyType, ValType>(keys[idx], values[idx]));
    idx += stride;
  }
}

template <typename Map, typename KeyType, typename ValType>
__global__ void find(Map map, const size_t size, const KeyType* keys,
                     ValType* results) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  while (idx < size) {
    auto found = map.find(keys[idx]);
    if (found != map.end()) {
      results[idx] = found->second;
    } else {
      results[idx] = map.get_unused_element();
    }
    idx += stride;
  }
}

}  // namespace details

// Lower case naming to follow std
template <typename KeyType, typename ValType>
class unordered_map {
 private:
  using SpecializedPair = thrust::pair<KeyType, ValType>;
  using Map = concurrent_unordered_map<KeyType, ValType, default_hash<KeyType>,
                                       equal_to<KeyType>,
                                       managed_allocator<SpecializedPair>>;
  const int kThreadBlockSize_ = 256;
  std::unique_ptr<Map, std::function<void(Map*)>> map_;
  size_t numElems_;

 public:
  unordered_map(const size_t numElems, const double loadFactor = 0.5)
      : numElems_(numElems) {
    size_t mapSize = (1 / loadFactor) * numElems;
    map_ = std::move(Map::create(mapSize));
  }

  void insert(const std::vector<KeyType>& keys,
              const std::vector<ValType>& vals) {
    if (keys.size() != vals.size()) {
      throw std::runtime_error("Number of keys and values must be equal.");
    }
    KeyType* dKeys;
    ValType* dVals;
    const size_t numBlocks = DIV_AND_CEIL(keys.size(), kThreadBlockSize_);

    CHK_CUDA(cudaMalloc((void**)&dKeys, keys.size() * sizeof(KeyType)));
    CHK_CUDA(cudaMalloc((void**)&dVals, vals.size() * sizeof(ValType)));

    CHK_CUDA(cudaMemcpy(dKeys, keys.data(), keys.size() * sizeof(KeyType),
                        cudaMemcpyHostToDevice));
    CHK_CUDA(cudaMemcpy(dVals, vals.data(), vals.size() * sizeof(ValType),
                        cudaMemcpyHostToDevice));

    details::insert<<<numBlocks, kThreadBlockSize_>>>(*map_.get(), keys.size(),
                                                      dKeys, dVals);
    CHK_CUDA(cudaFree(dKeys));
    CHK_CUDA(cudaFree(dVals));
  }

  std::vector<ValType> find(const std::vector<KeyType>& keys) {
    std::vector<ValType> hResults(keys.size());
    KeyType* dKeys;
    ValType* dResults;
    const size_t numBlocks = DIV_AND_CEIL(keys.size(), kThreadBlockSize_);

    CHK_CUDA(cudaMalloc((void**)&dKeys, keys.size() * sizeof(KeyType)));
    CHK_CUDA(cudaMalloc((void**)&dResults, keys.size() * sizeof(ValType)));

    CHK_CUDA(cudaMemcpy(dKeys, keys.data(), keys.size() * sizeof(KeyType),
                        cudaMemcpyHostToDevice));

    details::find<<<numBlocks, kThreadBlockSize_>>>(*map_.get(), keys.size(),
                                                    dKeys, dResults);

    CHK_CUDA(cudaMemcpy(hResults.data(), dResults,
                        keys.size() * sizeof(ValType), cudaMemcpyDeviceToHost));
    CHK_CUDA(cudaFree(dKeys));
    CHK_CUDA(cudaFree(dResults));
    return hResults;
  }
};
}  // namespace cuCollections

#endif