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

#include <cuco/detail/bloom_filter/kernels.cuh>
#include <cuco/detail/error.hpp>
#include <cuco/detail/utility/cuda.hpp>
#include <cuco/detail/utils.hpp>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cub/device/device_for.cuh>
#include <cuda/atomic>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/tuple>
#include <cuda/stream_ref>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>

#include <cooperative_groups.h>

#include <cstdint>
#include <type_traits>

namespace cuco::detail {

template <class Key, class Extent, cuda::thread_scope Scope, class Policy>
class bloom_filter_impl {
 public:
  using key_type    = Key;
  using extent_type = Extent;
  using size_type   = typename extent_type::value_type;
  using policy_type = Policy;
  using word_type =
    typename policy_type::word_type;  // TODO static_assert can use fetch_or() and load()

  static constexpr auto thread_scope    = Scope;  ///< CUDA thread scope
  static constexpr auto words_per_block = policy_type::words_per_block;

  __host__ __device__
  bloom_filter_impl(word_type* filter, Extent num_blocks, cuda_thread_scope<Scope>, Policy policy)
    : words_{filter}, num_blocks_{num_blocks}, policy_{policy}
  {
#ifndef __CUDA_ARCH__
    auto const alignment =
      1ull << cuda::std::countr_zero(reinterpret_cast<cuda::std::uintptr_t>(filter));
    CUCO_EXPECTS(alignment >= required_alignment(), "Invalid memory alignment", std::runtime_error);

    CUCO_EXPECTS(num_blocks_ > 0, "Number of blocks cannot be zero", std::runtime_error);
#endif
  }

  template <class CG>
  __device__ void clear(CG const& group)
  {
    for (int i = group.thread_rank(); num_blocks_ * words_per_block; i += group.size()) {
      words_[i] = 0;
    }
  }

  __host__ void clear(cuda::stream_ref stream)
  {
    this->clear_async(stream);
    stream.wait();
  }

  __host__ void clear_async(cuda::stream_ref stream)
  {
    CUCO_CUDA_TRY(cub::DeviceFor::ForEachN(
      words_,
      num_blocks_ * words_per_block,
      [] __device__(word_type & word) { word = 0; },
      stream.get()));
  }

  template <class ProbeKey>
  __device__ void add(ProbeKey const& key)
  {
    auto const hash_value = policy_.hash(key);
    auto const idx        = policy_.block_index(hash_value, num_blocks_);

#pragma unroll words_per_block
    for (uint32_t i = 0; i < words_per_block; ++i) {
      auto const word = policy_.word_pattern(hash_value, i);
      if (word != 0) {
        auto atom_word =
          cuda::atomic_ref<word_type, thread_scope>{*(words_ + (idx * words_per_block + i))};
        atom_word.fetch_or(word, cuda::memory_order_relaxed);
      }
    }
  }

  template <class ProbeKey>
  __device__ void add(cooperative_groups::thread_block_tile<words_per_block> const& tile,
                      ProbeKey const& key)
  {
    auto const hash_value = policy_.hash(key);
    auto const idx        = policy_.block_index(hash_value, num_blocks_);
    auto const rank       = tile.thread_rank();

    auto const word = policy_.word_pattern(hash_value, rank);
    if (word != 0) {
      auto atom_word =
        cuda::atomic_ref<word_type, thread_scope>{*(words_ + (idx * words_per_block + rank))};
      atom_word.fetch_or(word, cuda::memory_order_relaxed);
    }
  }

  // TODO
  // template <class CG, class InputIt>
  // __device__ void add(CG const& group, InputIt first, InputIt last);

  template <class InputIt>
  __host__ void add(InputIt first, InputIt last, cuda::stream_ref stream)
  {
    this->add_async(first, last, stream);
    stream.wait();
  }

  template <class InputIt>
  __host__ void add_async(InputIt first, InputIt last, cuda::stream_ref stream)
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    if constexpr (words_per_block == 1) {
      CUCO_CUDA_TRY(cub::DeviceFor::ForEachCopyN(
        first,
        num_keys,
        [*this] __device__(key_type const key) mutable { this->add(key); },
        stream.get()));
    } else {
      auto const always_true = thrust::constant_iterator<bool>{true};
      this->add_if_async(first, last, always_true, thrust::identity{}, stream);
    }
  }

  template <class InputIt, class StencilIt, class Predicate>
  __host__ void add_if(
    InputIt first, InputIt last, StencilIt stencil, Predicate pred, cuda::stream_ref stream)
  {
    this->add_if_async(first, last, stencil, pred, stream);
    stream.wait();
  }

  template <class InputIt, class StencilIt, class Predicate>
  __host__ void add_if_async(InputIt first,
                             InputIt last,
                             StencilIt stencil,
                             Predicate pred,
                             cuda::stream_ref stream) noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto constexpr block_size = cuco::detail::default_block_size();
    auto const grid_size      = cuco::detail::grid_size(
      num_keys, words_per_block, cuco::detail::default_stride(), block_size);

    detail::add_if_n<block_size>
      <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, stencil, pred, *this);
  }

  template <class ProbeKey>
  [[nodiscard]] __device__ bool contains(ProbeKey const& key) const
  {
    auto const hash_value = policy_.hash(key);

    auto const stored_pattern = this->vec_load_words<words_per_block>(
      policy_.block_index(hash_value, num_blocks_) * words_per_block);

#pragma unroll words_per_block
    for (uint32_t i = 0; i < words_per_block; ++i) {
      auto const expected_pattern = policy_.word_pattern(hash_value, i);
      if ((stored_pattern[i] & expected_pattern) != expected_pattern) { return false; }
    }

    return true;
  }

  // TODO
  // template <class CG, class ProbeKey>
  // [[nodiscard]] __device__ bool contains(CG const& group, ProbeKey const& key) const;

  // TODO
  // template <class CG, class InputIt, class OutputIt>
  // __device__ void contains(CG const& group, InputIt first, InputIt last, OutputIt output_begin)
  // const;

  template <class InputIt, class OutputIt>
  __host__ void contains(InputIt first,
                         InputIt last,
                         OutputIt output_begin,
                         cuda::stream_ref stream) const
  {
    this->contains_async(first, last, output_begin, stream);
    stream.wait();
  }

  template <class InputIt, class OutputIt>
  __host__ void contains_async(InputIt first,
                               InputIt last,
                               OutputIt output_begin,
                               cuda::stream_ref stream) const noexcept
  {
    auto const always_true = thrust::constant_iterator<bool>{true};
    this->contains_if_async(first, last, always_true, thrust::identity{}, output_begin, stream);
  }

  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  __host__ void contains_if(InputIt first,
                            InputIt last,
                            StencilIt stencil,
                            Predicate pred,
                            OutputIt output_begin,
                            cuda::stream_ref stream) const
  {
    this->contains_if_async(first, last, stencil, pred, output_begin, stream);
    stream.wait();
  }

  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  __host__ void contains_if_async(InputIt first,
                                  InputIt last,
                                  StencilIt stencil,
                                  Predicate pred,
                                  OutputIt output_begin,
                                  cuda::stream_ref stream) const noexcept
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto constexpr block_size = cuco::detail::default_block_size();
    auto const grid_size =
      cuco::detail::grid_size(num_keys, 1, cuco::detail::default_stride(), block_size);

    detail::contains_if_n<block_size><<<grid_size, block_size, 0, stream.get()>>>(
      first, num_keys, stencil, pred, output_begin, *this);
  }

  [[nodiscard]] __host__ __device__ word_type* data() noexcept { return words_; }

  [[nodiscard]] __host__ __device__ word_type const* data() const noexcept { return words_; }

  [[nodiscard]] __host__ __device__ extent_type block_extent() const noexcept
  {
    return num_blocks_;
  }

  // TODO
  // [[nodiscard]] __host__ double occupancy() const;
  // [[nodiscard]] __host__ double expected_false_positive_rate(size_t unique_keys) const
  // [[nodiscard]] __host__ __device__ static uint32_t optimal_pattern_bits(size_t num_blocks)
  // template <typename CG, cuda::thread_scope NewScope = thread_scope>
  // [[nodiscard]] __device__ constexpr auto make_copy(CG const& group, word_type* const
  // memory_to_use, cuda_thread_scope<NewScope> scope = {}) const noexcept;

 private:
  template <uint32_t NumWords>
  __device__ cuda::std::array<word_type, NumWords> vec_load_words(size_type index) const
  {
    return *reinterpret_cast<cuda::std::array<word_type, NumWords>*>(__builtin_assume_aligned(
      words_ + index, min(sizeof(word_type) * NumWords, required_alignment())));
  }

  __host__ __device__ static constexpr size_t required_alignment() noexcept
  {
    return sizeof(word_type) * words_per_block;  // TODO check if a maximum of 16byte is sufficient
  }

  word_type* words_;
  extent_type num_blocks_;
  policy_type policy_;
};

}  // namespace cuco::detail