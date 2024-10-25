/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#pragma once

#include <cstdint>

namespace cuco {

template <class T>
__host__ __device__ constexpr arrow_bf_policy<T>::arrow_bf_policy(std::uint32_t num_blocks,
                                                                  hasher hash)
  : impl_{num_blocks, hash}
{
}

template <class T>
__device__ constexpr typename arrow_bf_policy<T>::hash_result_type arrow_bf_policy<T>::hash(
  typename arrow_bf_policy<T>::hash_argument_type const& key) const
{
  return impl_.hash(key);
}

template <class T>
template <class Extent>
__device__ constexpr auto arrow_bf_policy<T>::block_index(
  typename arrow_bf_policy<T>::hash_result_type hash, Extent num_blocks) const
{
  return impl_.block_index(hash, num_blocks);
}

template <class T>
__device__ constexpr typename arrow_bf_policy<T>::word_type arrow_bf_policy<T>::word_pattern(
  arrow_bf_policy<T>::hash_result_type hash, std::uint32_t word_index) const
{
  return impl_.word_pattern(hash, word_index);
}

}  // namespace cuco