/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
 */

#pragma once

#include <algorithm>
#include <iterator>

namespace cuco {

/**
 * @brief Registers the global memory region `[begin, end)` to
 * be permanently resident in L2 cache.
 *
 * @tparam Iterator Accessor of the memory region
 * @param[in, out] stream The CUDA stream this region is accessed through
 * @param[in] begin Start of the memory region to be mapped
 * @param[in] end End of the memory region
 * @param[in] hit_rate Probability for a sub-segment to be mapped in L2
 * @param[in] carve_out Fraction of total L2 space to be blocked for resident memory segments
 *
 * @note Only has effect on Ampere and above.
 * @note Assumes the memory region to be contiguous.
 */
template <typename Iterator>
void register_l2_persistence(
  cudaStream_t& stream, Iterator begin, Iterator end, float hit_rate = 0.6f, float carve_out = 1.0f)
{
  using value_type = typename std::iterator_traits<Iterator>::value_type;

  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);

  hit_rate  = std::clamp(hit_rate, 0.0f, 1.0f);
  carve_out = std::clamp(carve_out, 0.0f, 1.0f);
  // Must be less than cudaDeviceProp::accessPolicyMaxWindowSize
  auto const num_bytes = std::min(std::distance(begin, end) * sizeof(value_type),
                                  std::size_t(prop.accessPolicyMaxWindowSize));

  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, carve_out * prop.persistingL2CacheMaxSize);

  // Stream level attributes data structure
  cudaStreamAttrValue stream_attribute;
  // Global Memory data pointer
  stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(&begin[0]);
  // Number of bytes for persistence access.
  stream_attribute.accessPolicyWindow.num_bytes = num_bytes;
  // Hint for cache hit ratio
  stream_attribute.accessPolicyWindow.hitRatio = hit_rate;
  // Type of access property on cache hit
  stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  // Type of access property on cache miss.
  stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  // Set the attributes to a CUDA stream of type cudaStream_t
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
}

/**
 * @brief Globally removes all persistent cache lines from L2.
 *
 * @param[in, out] stream The CUDA stream the resident region has been accessed through
 *
 * @note Only has effect on Ampere and above.
 */
void unregister_l2_persistence(cudaStream_t& stream)
{
  cudaStreamAttrValue stream_attribute;
  // Setting the window size to 0 to disable it
  stream_attribute.accessPolicyWindow.num_bytes = 0;
  // Overwrite the access policy attribute of CUDA Stream
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
  // Remove any persistent lines in L2$
  cudaCtxResetPersistingL2Cache();
}

}  // namespace cuco