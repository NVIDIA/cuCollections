/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

namespace cuco {

void register_l2_residency(cudaStream_t& stream,
                           void* ptr,
                           std::size_t num_bytes,
                           float l2_hit_rate  = 0.6f,
                           float l2_carve_out = 1.0f)
{
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);

  l2_hit_rate  = std::clamp(l2_hit_rate, 0.0f, 1.0f);
  l2_carve_out = std::clamp(l2_carve_out, 0.0f, 1.0f);
  // Must be less than cudaDeviceProp::accessPolicyMaxWindowSize
  num_bytes = std::min(num_bytes, std::size_t(prop.accessPolicyMaxWindowSize));

  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_carve_out * prop.persistingL2CacheMaxSize);

  // Stream level attributes data structure
  cudaStreamAttrValue stream_attribute;
  // Global Memory data pointer
  stream_attribute.accessPolicyWindow.base_ptr = ptr;
  // Number of bytes for persistence access.
  stream_attribute.accessPolicyWindow.num_bytes = num_bytes;
  // Hint for cache hit ratio
  stream_attribute.accessPolicyWindow.hitRatio = l2_hit_rate;
  // Type of access property on cache hit
  stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  // Type of access property on cache miss.
  stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  // Set the attributes to a CUDA stream of type cudaStream_t
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
}

void unregister_l2_residency(cudaStream_t& stream)
{
  cudaStreamAttrValue stream_attribute;
  // Setting the window size to 0 disable it
  stream_attribute.accessPolicyWindow.num_bytes = 0;
  // Overwrite the access policy attribute to a CUDA Stream
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
  // Remove any persistent lines in L2
  cudaCtxResetPersistingL2Cache();
}

}  // namespace cuco