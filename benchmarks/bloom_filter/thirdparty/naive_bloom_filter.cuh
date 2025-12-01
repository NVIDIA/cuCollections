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

#pragma once

#include <cuco/detail/utility/cuda.cuh>
#include <cuco/detail/utility/math.cuh>
#include <cuco/detail/utils.hpp>
#include <cuco/hash_functions.cuh>

#include <cstddef>
#include <cstdint>
#include <limits>

template <class InputIt, class Ref>
__global__ void add_kernel_vertical(InputIt first, typename Ref::size_type n, Ref ref)
{
  auto const tid = cuco::detail::global_thread_id();
  auto const r   = ref;
  if (tid < n) {
    auto const key       = *(first + tid);
    auto const base_hash = r.hash(key);
    auto const step_hash = r.hash2(key);
    for (int i = 0; i < r.k; ++i) {
      typename Ref::size_type const bit_index = (base_hash + i * step_hash) % r.num_bits;
      typename Ref::word_type const word      = typename Ref::word_type(1u)
                                           << (bit_index % Ref::word_bits);
      atomicOr(r.words + (bit_index / Ref::word_bits), word);
    }
  }
}

template <class InputIt, class Ref>
__global__ void add_kernel_horizontal(InputIt first, typename Ref::size_type n, Ref ref)
{
  auto const tid = cuco::detail::global_thread_id();
  auto const idx = tid / ref.k;
  auto const r   = ref;
  if (idx < n) {
    auto const key                          = *(first + idx);
    auto const base_hash                    = r.hash(key);
    auto const step_hash                    = r.hash2(key);
    typename Ref::size_type const bit_index = (base_hash + (tid % ref.k) * step_hash) % r.num_bits;
    typename Ref::word_type const word      = typename Ref::word_type(1u)
                                         << (bit_index % Ref::word_bits);
    atomicOr(r.words + (bit_index / Ref::word_bits), word);
  }
}

template <class InputIt, class OutputIt, class Ref>
__global__ void contains_kernel_vertical(InputIt first,
                                         typename Ref::size_type n,
                                         OutputIt output_begin,
                                         Ref ref)
{
  auto const tid = cuco::detail::global_thread_id();
  auto const r   = ref;
  if (tid < n) {
    auto const key       = *(first + tid);
    bool result          = true;
    auto const base_hash = r.hash(key);
    auto const step_hash = r.hash2(key);
    for (int i = 0; i < r.k; ++i) {
      typename Ref::size_type const bit_index = (base_hash + i * step_hash) % r.num_bits;
      typename Ref::word_type const word      = typename Ref::word_type(1u)
                                           << (bit_index % Ref::word_bits);
      if ((r.words[bit_index / Ref::word_bits] & word) != word) {
        result = false;
        break;
      }
    }
    *(output_begin + tid) = result;
  }
}

template <class Key, class Hash = cuco::xxhash_64<Key>>
class naive_bloom_filter {
 public:
  using key_type  = Key;
  using word_type = uint32_t;
  using size_type = size_t;

  static constexpr int word_bits               = std::numeric_limits<word_type>::digits;
  static constexpr size_type kernel_block_size = 256;

  struct ref_type {
    using key_type  = key_type;
    using word_type = word_type;
    using size_type = size_type;

    static constexpr int word_bits = std::numeric_limits<word_type>::digits;

    size_type num_bits;
    int k;
    Hash hash;
    size_type num_words;
    word_type* words;
    cuco::murmurhash3_32<key_type> hash2;
  } ref;

  __host__ explicit constexpr naive_bloom_filter(size_type bits,
                                                 int k,
                                                 Hash hash           = {},
                                                 cudaStream_t stream = nullptr)
    : ref{bits,
          k,
          hash,
          cuco::detail::int_div_ceil(bits, static_cast<size_type>(word_bits)),
          nullptr,
          {}}
  {
    CUCO_CUDA_TRY(cudaMalloc(&(ref.words), ref.num_words * sizeof(word_type)));
    this->clear(stream);
  }

  __host__ ~naive_bloom_filter() noexcept(false) { CUCO_CUDA_TRY(cudaFree(ref.words)); }

  __host__ void clear_async(cudaStream_t stream)
  {
    cudaMemsetAsync(ref.words, 0, ref.num_words * sizeof(word_type), stream);
  }

  __host__ void clear(cudaStream_t stream = nullptr)
  {
    this->clear_async(stream);
    CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  template <class InputIt>
  __host__ void add_async(InputIt first, InputIt last, cudaStream_t stream)
  {
    size_type const num_keys = cuda::std::distance(first, last);
    add_kernel_vertical<<<cuco::detail::int_div_ceil(num_keys, kernel_block_size),
                          kernel_block_size,
                          0,
                          stream>>>(first, num_keys, ref);
    // add_kernel_horizontal<<<cuco::detail::int_div_ceil(num_keys * ref.k, kernel_block_size),
    // kernel_block_size, 0, stream>>>(first, num_keys, ref);
  }

  template <class InputIt>
  __host__ void add(InputIt first, InputIt last, cudaStream_t stream = nullptr)
  {
    this->add_async(first, last, stream);
    CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  template <class InputIt, class OutputIt>
  __host__ void contains_async(InputIt first,
                               InputIt last,
                               OutputIt output_begin,
                               cudaStream_t stream) const
  {
    size_type const num_keys = cuda::std::distance(first, last);
    contains_kernel_vertical<<<cuco::detail::int_div_ceil(num_keys, kernel_block_size),
                               kernel_block_size,
                               0,
                               stream>>>(first, num_keys, output_begin, ref);
  }

  template <class InputIt, class OutputIt>
  __host__ void contains(InputIt first,
                         InputIt last,
                         OutputIt output_begin,
                         cudaStream_t stream = nullptr) const
  {
    this->contains_async(first, last, output_begin, stream);
    CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  }
};