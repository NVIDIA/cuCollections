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
#include <cuda/stream_ref>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>

#include <cooperative_groups.h>

#include <cstdint>
#include <type_traits>

namespace cuco::detail {

template <class Key,
          class Extent,
          cuda::thread_scope Scope,
          class Hash,
          uint32_t BlockWords,
          class Word>
class bloom_filter_impl {
 public:
  static constexpr auto thread_scope = Scope;  ///< CUDA thread scope
  static constexpr auto block_words  = BlockWords;
  // static constexpr auto alignment    = std::max(16, sizeof(Word) * BlockWords); // TODO include

  using key_type    = Key;
  using extent_type = Extent;
  using size_type   = typename extent_type::value_type;
  using hasher      = Hash;
  using word_type   = Word;  // TODO static_assert can use fetch_or() and load()

  static_assert(cuda::std::has_single_bit(BlockWords) and BlockWords <= 32,
                "Number of words per block must be a power-of-two and less than or equal to 32");

  __host__ __device__ bloom_filter_impl(word_type* filter,
                                        Extent num_blocks,
                                        uint32_t pattern_bits,
                                        cuda_thread_scope<Scope>,
                                        Hash const& hash)
    : words_{filter}, num_blocks_{num_blocks}, pattern_bits_{pattern_bits}, hash_{hash}
  {
#ifndef __CUDA_ARCH__
    auto const alignment =
      1ull << cuda::std::countr_zero(reinterpret_cast<cuda::std::uintptr_t>(filter));
    CUCO_EXPECTS(alignment >= required_alignment(), "Invalid memory alignment", std::runtime_error);

    CUCO_EXPECTS(this->num_blocks_ > 0, "Number of blocks cannot be zero", std::runtime_error);
#endif
  }

  template <class CG>
  __device__ void clear(CG const& group)
  {
    for (int i = group.thread_rank(); num_blocks_ * block_words; i += group.size()) {
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
      num_blocks_ * block_words,
      [] __device__(word_type & word) { word = 0; },
      stream.get()));
  }

  template <class ProbeKey>
  __device__ void add(ProbeKey const& key)
  {
    auto const hash_value = hash_(key);
    auto const idx        = this->block_idx(hash_value);

#pragma unroll block_words
    for (int32_t i = 0; i < block_words; ++i) {
      auto const word = this->pattern_word(hash_value, i);
      if (word != 0) {
        auto atom_word =
          cuda::atomic_ref<word_type, thread_scope>{*(words_ + (idx * block_words + i))};
        atom_word.fetch_or(word, cuda::memory_order_relaxed);
      }
    }
  }

  template <class ProbeKey>
  __device__ void add(cooperative_groups::thread_block_tile<block_words> const& tile,
                      ProbeKey const& key)
  {
    auto const hash_value = hash_(key);
    auto const idx        = this->block_idx(hash_value);
    auto const rank       = tile.thread_rank();

    auto const word = this->pattern_word(hash_value, rank);
    if (word != 0) {
      auto atom_word =
        cuda::atomic_ref<word_type, thread_scope>{*(words_ + (idx * block_words + rank))};
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

    if constexpr (block_words == 1) {
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
    auto const grid_size =
      cuco::detail::grid_size(num_keys, block_words, cuco::detail::default_stride(), block_size);

    detail::add_if_n<block_size>
      <<<grid_size, block_size, 0, stream.get()>>>(first, num_keys, stencil, pred, *this);
  }

  template <class ProbeKey>
  [[nodiscard]] __device__ bool test(ProbeKey const& key) const
  {
    auto const hash_value = hash_(key);
    auto const idx        = this->block_idx(hash_value);

    auto const stored_pattern =
      this->vec_load_words<block_words>(idx * block_words);  // vectorized load
    auto const expected_pattern = this->pattern(hash_value);

#pragma unroll block_words
    for (int32_t i = 0; i < block_words; ++i) {
      if ((stored_pattern[i] & expected_pattern[i]) != expected_pattern[i]) { return false; }
    }

    return true;
  }

  // TODO
  // template <class CG, class ProbeKey>
  // [[nodiscard]] __device__ bool test(CG const& group, ProbeKey const& key) const;

  // TODO
  // template <class CG, class InputIt, class OutputIt>
  // __device__ void test(CG const& group, InputIt first, InputIt last, OutputIt output_begin)
  // const;

  template <class InputIt, class OutputIt>
  __host__ void test(InputIt first,
                     InputIt last,
                     OutputIt output_begin,
                     cuda::stream_ref stream) const
  {
    this->test_async(first, last, output_begin, stream);
    stream.wait();
  }

  template <class InputIt, class OutputIt>
  __host__ void test_async(InputIt first,
                           InputIt last,
                           OutputIt output_begin,
                           cuda::stream_ref stream) const noexcept
  {
    auto const always_true = thrust::constant_iterator<bool>{true};
    this->test_if_async(first, last, always_true, thrust::identity{}, output_begin, stream);
  }

  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  __host__ void test_if(InputIt first,
                        InputIt last,
                        StencilIt stencil,
                        Predicate pred,
                        OutputIt output_begin,
                        cuda::stream_ref stream) const
  {
    this->test_if_async(first, last, stencil, pred, output_begin, stream);
    stream.wait();
  }

  template <class InputIt, class StencilIt, class Predicate, class OutputIt>
  __host__ void test_if_async(InputIt first,
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

    detail::test_if_n<block_size><<<grid_size, block_size, 0, stream.get()>>>(
      first, num_keys, stencil, pred, output_begin, *this);
  }

  [[nodiscard]] __host__ __device__ word_type* data() noexcept { return words_; }

  [[nodiscard]] __host__ __device__ word_type const* data() const noexcept { return words_; }

  [[nodiscard]] __host__ __device__ extent_type block_extent() const noexcept
  {
    return num_blocks_;
  }

  [[nodiscard]] __host__ __device__ hasher hash_function() const noexcept { return hash_; }

  // TODO
  // [[nodiscard]] __host__ double occupancy() const;
  // [[nodiscard]] __host__ double expected_false_positive_rate(size_t unique_keys) const
  // [[nodiscard]] __host__ __device__ static uint32_t optimal_pattern_bits(size_t num_blocks)
  // template <typename CG, cuda::thread_scope NewScope = thread_scope>
  // [[nodiscard]] __device__ constexpr auto make_copy(CG const& group, word_type* const
  // memory_to_use, cuda_thread_scope<NewScope> scope = {}) const noexcept;

 private:
  template <class HashValue>
  __device__ size_type block_idx(HashValue hash_value) const
  {
    // TODO use fast_int modulo
    return hash_value % num_blocks_;
  }

  // we use the LSB bits of the hash value to determine the pattern bits for each word
  template <class HashValue>
  __device__ auto pattern(HashValue hash_value) const
  {
    cuda::std::array<word_type, block_words> pattern{};
    auto constexpr word_bits           = sizeof(word_type) * CHAR_BIT;
    auto constexpr bit_index_width     = cuda::std::bit_width(word_bits - 1);
    word_type constexpr bit_index_mask = (word_type{1} << bit_index_width) - 1;

    auto const bits_per_word = pattern_bits_ / block_words;
    auto const remainder     = pattern_bits_ % block_words;

    uint32_t k = 0;
#pragma unroll block_words
    for (int32_t i = 0; i < block_words; ++i) {
      for (int32_t j = 0; j < bits_per_word + (i < remainder ? 1 : 0); ++j) {
        if (k++ >= pattern_bits_) { return pattern; }
        pattern[i] |= word_type{1} << (hash_value & bit_index_mask);
        hash_value >>= bit_index_width;
      }
    }

    return pattern;
  }

  template <class HashValue>
  __device__ word_type pattern_word(HashValue hash_value, uint32_t i) const
  {
    auto constexpr word_bits           = sizeof(word_type) * CHAR_BIT;
    auto constexpr bit_index_width     = cuda::std::bit_width(word_bits - 1);
    word_type constexpr bit_index_mask = (word_type{1} << bit_index_width) - 1;

    auto const bits_per_word = pattern_bits_ / block_words;
    auto const remainder     = pattern_bits_ % block_words;
    auto const bits_so_far   = bits_per_word * i + (i < remainder ? i : remainder);

    hash_value >>= bits_so_far * bit_index_width;

    // Compute the word at index i
    word_type word  = 0;
    int32_t j_limit = bits_per_word + (i < remainder ? 1 : 0);

    for (int32_t j = 0; j < j_limit; ++j) {
      word |= word_type{1} << (hash_value & bit_index_mask);
      hash_value >>= bit_index_width;
    }

    return word;
  }

  template <uint32_t NumWords>
  __device__ auto vec_load_words(size_type index) const
  {
    using vec_type = cuda::std::array<word_type, NumWords>;

    return *reinterpret_cast<vec_type*>(
      __builtin_assume_aligned(words_ + index, sizeof(word_type) * NumWords));
  }

  __host__ __device__ static constexpr size_t required_alignment() noexcept
  {
    return sizeof(word_type) * BlockWords;
  }

  word_type* words_;
  extent_type num_blocks_;
  uint32_t pattern_bits_;
  hasher hash_;
};
}  // namespace cuco::detail