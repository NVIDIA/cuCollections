#ifndef GDF_UTILS_H
#define GDF_UTILS_H

#include <cuda_runtime_api.h>

#include <cassert>
#include <vector>

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE_CALLABLE __host__ __device__ inline
#define CUDA_DEVICE_CALLABLE __device__ inline
#else
#define CUDA_HOST_DEVICE_CALLABLE inline
#define CUDA_DEVICE_CALLABLE inline
#endif

inline bool isPtrManaged(cudaPointerAttributes attr) {
#if CUDART_VERSION >= 10000
  return (attr.type == cudaMemoryTypeManaged);
#else
  return attr.isManaged;
#endif
}

#endif  // GDF_UTILS_H
