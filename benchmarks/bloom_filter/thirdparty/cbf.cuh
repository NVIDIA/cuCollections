/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/utility/cuda.cuh>
#include <cuco/detail/utility/math.cuh>
#include <cuco/detail/utils.hpp>
#include <cuco/hash_functions.cuh>

#include <cuda/std/iterator>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cuco::benchmark {
namespace detail {

template <class InputIt, class Ref>
CUCO_KERNEL void cbf_add_kernel(InputIt first, typename Ref::size_type n, Ref ref)
{
  auto const tid = cuco::detail::global_thread_id();
  if (tid < n) {
    auto const key       = *(first + tid);
    auto const base_hash = ref.hash(key);
    auto const step_hash = ref.hash2(key);
    for (int i = 0; i < ref.num_hashes; ++i) {
      typename Ref::size_type const bit_index = (base_hash + i * step_hash) % ref.num_bits;
      typename Ref::word_type const word      = typename Ref::word_type{1}
                                           << (bit_index % Ref::word_bits);
      atomicOr(ref.words + (bit_index / Ref::word_bits), word);
    }
  }
}

template <class InputIt, class OutputIt, class Ref>
CUCO_KERNEL void cbf_contains_kernel(InputIt first,
                                     typename Ref::size_type n,
                                     OutputIt output_begin,
                                     Ref ref)
{
  auto const tid = cuco::detail::global_thread_id();
  if (tid < n) {
    auto const key       = *(first + tid);
    auto const base_hash = ref.hash(key);
    auto const step_hash = ref.hash2(key);
    bool result          = true;
    for (int i = 0; i < ref.num_hashes; ++i) {
      typename Ref::size_type const bit_index = (base_hash + i * step_hash) % ref.num_bits;
      typename Ref::word_type const word      = typename Ref::word_type{1}
                                           << (bit_index % Ref::word_bits);
      if ((ref.words[bit_index / Ref::word_bits] & word) != word) {
        result = false;
        break;
      }
    }
    *(output_begin + tid) = result;
  }
}

}  // namespace detail

template <class Key, class Hash = cuco::xxhash_64<Key>>
class cbf {
 public:
  using key_type  = Key;
  using word_type = std::uint32_t;
  using size_type = std::size_t;

  static constexpr int word_bits               = std::numeric_limits<word_type>::digits;
  static constexpr size_type kernel_block_size = 256;

  struct ref_type {
    using word_type = std::uint32_t;
    using size_type = std::size_t;

    static constexpr int word_bits = std::numeric_limits<word_type>::digits;

    size_type num_bits;
    int num_hashes;
    Hash hash;
    word_type* words;
    cuco::murmurhash3_32<key_type> hash2;
  };

  explicit cbf(size_type num_bits, int num_hashes, Hash hash = {}, cudaStream_t stream = nullptr)
    : ref_{num_bits, num_hashes, hash, nullptr, {}},
      num_words_{cuco::detail::int_div_ceil(num_bits, static_cast<size_type>(word_bits))}
  {
    CUCO_CUDA_TRY(cudaMalloc(&ref_.words, num_words_ * sizeof(word_type)));
    clear(stream);
  }

  cbf(cbf const&)            = delete;
  cbf& operator=(cbf const&) = delete;
  cbf(cbf&&)                 = delete;
  cbf& operator=(cbf&&)      = delete;

  ~cbf() noexcept(false) { CUCO_CUDA_TRY(cudaFree(ref_.words)); }

  void clear_async(cudaStream_t stream)
  {
    CUCO_CUDA_TRY(cudaMemsetAsync(ref_.words, 0, num_words_ * sizeof(word_type), stream));
  }

  void clear(cudaStream_t stream = nullptr)
  {
    clear_async(stream);
    CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  template <class InputIt>
  void add_async(InputIt first, InputIt last, cudaStream_t stream)
  {
    size_type const num_keys = static_cast<size_type>(cuda::std::distance(first, last));
    detail::cbf_add_kernel<<<cuco::detail::int_div_ceil(num_keys, kernel_block_size),
                             kernel_block_size,
                             0,
                             stream>>>(first, num_keys, ref_);
    CUCO_CUDA_TRY(cudaPeekAtLastError());
  }

  template <class InputIt>
  void add(InputIt first, InputIt last, cudaStream_t stream = nullptr)
  {
    add_async(first, last, stream);
    CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  template <class InputIt, class OutputIt>
  void contains_async(InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream) const
  {
    size_type const num_keys = static_cast<size_type>(cuda::std::distance(first, last));
    detail::cbf_contains_kernel<<<cuco::detail::int_div_ceil(num_keys, kernel_block_size),
                                  kernel_block_size,
                                  0,
                                  stream>>>(first, num_keys, output_begin, ref_);
    CUCO_CUDA_TRY(cudaPeekAtLastError());
  }

  template <class InputIt, class OutputIt>
  void contains(InputIt first,
                InputIt last,
                OutputIt output_begin,
                cudaStream_t stream = nullptr) const
  {
    contains_async(first, last, output_begin, stream);
    CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 private:
  ref_type ref_;
  size_type num_words_;
};

}  // namespace cuco::benchmark
