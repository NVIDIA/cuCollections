#ifndef CC_UTILS_H
#define CC_UTILS_H

inline bool isPtrManaged(cudaPointerAttributes attr) {
#if CUDART_VERSION >= 10000
  return (attr.type == cudaMemoryTypeManaged);
#else
  return attr.isManaged;
#endif
}

#endif  // CC_UTILS_H
